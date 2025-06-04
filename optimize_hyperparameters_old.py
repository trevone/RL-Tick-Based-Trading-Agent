# optimize_hyperparameters.py
import optuna
import os
import json
import numpy as np
import traceback

# Assuming train_simple_agent.py is in the same directory
from train_simple_agent import train_agent, DEFAULT_TRAIN_CONFIG # Import the new train_agent function and its defaults
from base_env import DEFAULT_ENV_CONFIG # Import for hashing consistency
from utils import load_config, merge_configs, convert_to_native_types, get_relevant_config_for_hash, generate_config_hash

# --- Configuration for Optimization ---
# These are parameters specific to the Optuna optimization process itself.
OPTIMIZATION_CONFIG = {
    "n_trials": 50, # Number of optimization trials to run
    "timeout_seconds": None, # Max seconds for optimization (None for no limit)
    "sampler": "tpe", # 'tpe' (Tree-structured Parzen Estimator) or 'random'
    "pruner": "median", # 'median', 'hyperband', 'sha' (Successive Halving) or None
    "pruning_interval_steps": 10000, # How often to check for pruning during training steps
    "db_url": "sqlite:///optuna_study.db", # URL for the study database (for resuming, parallelization)
    "study_name": "ppo_trading_hyperparam_study", # Name for the Optuna study
    "optimization_log_level": "INFO", # Optuna's log level (INFO, WARNING, ERROR)
    "optimization_metric_direction": "maximize", # 'maximize' or 'minimize'
    "optimization_metric_name": "eval_reward", # The metric returned by train_agent
}

# --- Hyperparameter Search Space Definition ---
# Define the ranges and types for hyperparameters Optuna will explore.
# These will override values in the 'ppo_params' section of config.yaml
HYPERPARAMETER_SEARCH_SPACE = {
    "ppo_params": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}, # log-uniform scale
        "n_steps": {"type": "categorical", "choices": [256, 512, 1024, 2048, 4096]}, # powers of 2
        "gamma": {"type": "float", "low": 0.9, "high": 0.999, "log": False},
        "ent_coef": {"type": "float", "low": 0.0001, "high": 0.1, "log": True}, # log-uniform scale
        "clip_range": {"type": "float", "low": 0.1, "high": 0.4, "log": False},
        # "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]}, # Often depends on n_steps
        # "n_epochs": {"type": "int", "low": 5, "high": 20},
        # "policy_kwargs_net_arch": {"type": "categorical", "choices": ["{'net_arch': [64, 64]}", "{'net_arch': [128, 128]}"]},
    },
    # You can also add environment parameters here if you want to optimize them
    # "environment": {
    #     "kline_window_size": {"type": "int", "low": 10, "high": 50},
    #     "tick_feature_window_size": {"type": "int", "low": 20, "high": 100},
    # }
}

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    Each call to this function runs a single training trial.
    """
    print(f"\n--- Starting Optuna Trial {trial.number} ---")

    # Define hyperparameters to optimize
    ppo_hparams = {}
    for param_name, details in HYPERPARAMETER_SEARCH_SPACE["ppo_params"].items():
        if details["type"] == "float":
            if details.get("log", False):
                ppo_hparams[param_name] = trial.suggest_float(param_name, details["low"], details["high"], log=True)
            else:
                ppo_hparams[param_name] = trial.suggest_float(param_name, details["low"], details["high"])
        elif details["type"] == "int":
            if details.get("log", False):
                ppo_hparams[param_name] = trial.suggest_int(param_name, details["low"], details["high"], log=True)
            else:
                ppo_hparams[param_name] = trial.suggest_int(param_name, details["low"], details["high"])
        elif details["type"] == "categorical":
            ppo_hparams[param_name] = trial.suggest_categorical(param_name, details["choices"])
        else:
            raise ValueError(f"Unknown hyperparameter type: {details['type']}")

    # Apply some common sense constraints or dynamic suggestions
    # E.g., n_steps should be a multiple of batch_size
    # if "n_steps" in ppo_hparams and "batch_size" in ppo_hparams:
    #     ppo_hparams["batch_size"] = trial.suggest_categorical("batch_size_constrained", [
    #         bs for bs in HYPERPARAMETER_SEARCH_SPACE["ppo_params"]["batch_size"]["choices"]
    #         if ppo_hparams["n_steps"] % bs == 0
    #     ])

    # Important: Override total_timesteps for faster trials
    # Also, ensure policy_kwargs (net_arch) is parsed if it's a string from config
    base_ppo_params = DEFAULT_TRAIN_CONFIG["ppo_params"].copy()
    if isinstance(base_ppo_params["policy_kwargs"], str):
        try: base_ppo_params["policy_kwargs"] = eval(base_ppo_params["policy_kwargs"])
        except Exception: pass
        
    ppo_params_for_trial = {
        **base_ppo_params,
        **ppo_hparams
    }
    
    # Set a much smaller total_timesteps for optimization trials
    ppo_params_for_trial["total_timesteps"] = DEFAULT_TRAIN_CONFIG["ppo_params"]["total_timesteps"] // 100 # Example: 1/100th of full timesteps

    # Construct the config override for train_agent function
    config_override = {
        "ppo_params": ppo_params_for_trial,
        "run_settings": {
            "log_level": "none" # Suppress excessive logging during trials
        }
    }

    try:
        # Optuna's Pruning Callback (can be integrated with EvalCallback for more granular pruning)
        # For now, we'll return the final metric and let Optuna decide based on that.
        # If you want early stopping *during training* of a trial, you need to create a custom
        # SB3 callback that reports to `trial.report()` and raises `optuna.exceptions.TrialPruned`.

        print(f"Trial {trial.number}: Training with hyperparameters: {ppo_params_for_trial}")
        
        # Call the train_agent function. It will return the metric.
        metric_value = train_agent(
            config_override=config_override,
            log_to_file=False,
            metric_to_return=OPTIMIZATION_CONFIG["optimization_metric_name"]
        )
        
        if np.isinf(metric_value) or np.isnan(metric_value):
            raise optuna.TrialPruned(f"Trial {trial.number} pruned due to invalid metric value (inf or NaN).")

        # You can report intermediate values for pruning if you integrate a custom callback
        # trial.report(intermediate_value, step)
        # if trial.should_prune():
        #    raise optuna.TrialPruned()

        return metric_value

    except optuna.TrialPruned as e:
        print(f"Trial {trial.number} pruned: {e}")
        raise # Re-raise for Optuna to handle
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials

if __name__ == "__main__":
    # Set up Optuna logging level
    optuna.logging.set_verbosity(getattr(optuna.logging, OPTIMIZATION_CONFIG["optimization_log_level"].upper()))

    # --- Create or Load Optuna Study ---
    # Sampler and Pruner configuration
    sampler = optuna.samplers.TPESampler(seed=42) if OPTIMIZATION_CONFIG["sampler"] == "tpe" else optuna.samplers.RandomSampler(seed=42)
    
    pruner = None
    if OPTIMIZATION_CONFIG["pruner"] == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=OPTIMIZATION_CONFIG["pruning_interval_steps"] * 2,
            interval_steps=OPTIMIZATION_CONFIG["pruning_interval_steps"]
        )
    elif OPTIMIZATION_CONFIG["pruner"] == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    elif OPTIMIZATION_CONFIG["pruner"] == "sha": # Successive Halving Pruner
        pruner = optuna.pruners.SuccessiveHalvingPruner()

    study = optuna.create_study(
        study_name=OPTIMIZATION_CONFIG["study_name"],
        direction=OPTIMIZATION_CONFIG["optimization_metric_direction"],
        sampler=sampler,
        pruner=pruner, # Pass the pruner
        storage=OPTIMIZATION_CONFIG["db_url"], # Persist study to a database
        load_if_exists=True # Load existing study if it exists
    )

    print(f"Starting optimization for study '{OPTIMIZATION_CONFIG['study_name']}'...")
    print(f"Optimization direction: {OPTIMIZATION_CONFIG['optimization_metric_direction']} {OPTIMIZATION_CONFIG['optimization_metric_name']}")
    print(f"Number of existing trials: {len(study.trials)}")

    try:
        study.optimize(
            objective,
            n_trials=OPTIMIZATION_CONFIG["n_trials"],
            timeout=OPTIMIZATION_CONFIG["timeout_seconds"],
            gc_after_trial=True # Clean up memory after each trial
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during optimization: {e}")
        traceback.print_exc()

    print("\n--- Optimization Finished ---")
    print("\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print("  Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Optional: Save best parameters to a JSON file
    # Ensure the directory exists for the sqlite db first
    db_dir = os.path.dirname(OPTIMIZATION_CONFIG["db_url"].replace("sqlite:///", ""))
    os.makedirs(db_dir, exist_ok=True)
    best_params_path = os.path.join(db_dir, "best_hyperparameters.json")

    try:
        with open(best_params_path, "w") as f:
            json.dump(convert_to_native_types(study.best_params), f, indent=4)
        print(f"\nBest hyperparameters saved to: {best_params_path}")
    except Exception as e:
        print(f"Error saving best hyperparameters: {e}")
        traceback.print_exc()

    # Optional: Plotting optimization results (requires plotly)
    # pip install plotly
    try:
        import optuna.visualization as vis
        # Make sure you have Plotly installed for visualization
        if hasattr(vis, 'plot_optimization_history'):
            fig = vis.plot_optimization_history(study)
            fig.show()
        if hasattr(vis, 'plot_param_importances'):
            fig = vis.plot_param_importances(study)
            fig.show()
    except ImportError:
        print("Install plotly for visualization: pip install plotly")
    except Exception as e:
        print(f"Error generating Optuna plots: {e}")
        traceback.print_exc()