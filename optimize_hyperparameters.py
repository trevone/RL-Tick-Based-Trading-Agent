# optimize_hyperparameters.py
import optuna
import os
import json
import numpy as np
import traceback
import sys

# Import the train_agent function from the refactored train_simple_agent.py
# Also import its default configurations to properly construct config overrides
from train_simple_agent import (
    train_agent,
    DEFAULT_RUN_SETTINGS,
    DEFAULT_ENV_CONFIG,
    DEFAULT_PPO_PARAMS,
    DEFAULT_BINANCE_SETTINGS,
    DEFAULT_EVALUATION_DATA_SETTINGS,
    DEFAULT_HASH_CONFIG_KEYS
)
from utils import load_config, merge_configs, convert_to_native_types, get_relevant_config_for_hash, generate_config_hash

# Ensure matplotlib is imported before anything that might set a backend if plotting is desired
try:
    import matplotlib.pyplot as plt
    import optuna.visualization as vis
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: Plotly or Matplotlib not installed. Optimization plots will not be generated. Install with: pip install plotly matplotlib")


# --- Configuration for Optimization ---
# These are parameters specific to the Optuna optimization process itself.
OPTIMIZATION_CONFIG = {
    "n_trials": 50, # Number of optimization trials to run
    "timeout_seconds": None, # Max seconds for optimization (None for no limit)
    "sampler": "tpe", # 'tpe' (Tree-structured Parzen Estimator) or 'random'
    "pruner": "median", # 'median', 'hyperband', 'sha' (Successive Halving) or None
    "pruning_interval_steps": 10000, # How often to check for pruning during training steps (not directly used by current SB3 EvalCallback, but for context)
    "db_url": "sqlite:///optuna_study.db", # URL for the study database (for resuming, parallelization)
    "study_name": "ppo_trading_hyperparam_study", # Name for the Optuna study
    "optimization_log_level": "INFO", # Optuna's log level (INFO, WARNING, ERROR)
    "optimization_metric_direction": "maximize", # 'maximize' or 'minimize'
    "optimization_metric_name": "eval_reward", # The metric returned by train_agent (e.g., "eval_reward")
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
        "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]}, # Added common choices
        "n_epochs": {"type": "int", "low": 4, "high": 15}, # More reasonable range
        # Example for policy_kwargs. Remember this would need to be in the hash_config_keys as well.
        # "policy_kwargs": {"type": "categorical", "choices": [
        #     "dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))",
        #     "dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))",
        #     "dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))"
        # ]},
    },
    # You can also add environment parameters here if you want to optimize them
    # "environment": {
    #     "kline_window_size": {"type": "int", "low": 10, "high": 50},
    #     "tick_feature_window_size": {"type": "int", "low": 20, "high": 100},
    #     "initial_balance": {"type": "categorical", "choices": [10000, 50000]},
    # }
}

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    Each call to this function runs a single training trial.
    """
    print(f"\n--- Starting Optuna Trial {trial.number} ---")

    # Define hyperparameters to optimize based on the search space
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

    # Handle environment hyperparameters if they are part of the search space
    env_hparams = {}
    if "environment" in HYPERPARAMETER_SEARCH_SPACE:
        for param_name, details in HYPERPARAMETER_SEARCH_SPACE["environment"].items():
            if details["type"] == "int":
                env_hparams[param_name] = trial.suggest_int(param_name, details["low"], details["high"])
            elif details["type"] == "float":
                env_hparams[param_name] = trial.suggest_float(param_name, details["low"], details["high"])
            elif details["type"] == "categorical":
                env_hparams[param_name] = trial.suggest_categorical(param_name, details["choices"])

    # Important: Override total_timesteps for faster trials
    # The `train_agent` function already handles popping `total_timesteps` from `ppo_params`.
    # We set a smaller total_timesteps for optimization trials.
    # It's better to use a small fraction of the total_timesteps from DEFAULT_PPO_PARAMS
    # or define a specific `optimization_timesteps` in OPTIMIZATION_CONFIG.
    optimization_timesteps = DEFAULT_PPO_PARAMS["total_timesteps"] // 100 # Example: 1/100th of full timesteps

    # Construct the config override for train_agent function
    # The structure must match the top-level sections that `train_agent` expects.
    config_override = {
        "ppo_params": {
            **DEFAULT_PPO_PARAMS.copy(), # Start with default PPO params
            **ppo_hparams, # Apply trial's suggested PPO hparams
            "total_timesteps": optimization_timesteps # Override total timesteps for trial
        },
        "environment": {
            **DEFAULT_ENV_CONFIG.copy(), # Start with default env config
            **env_hparams # Apply trial's suggested env hparams
        },
        "run_settings": {
            "log_level": "none", # Suppress excessive logging during trials
            "model_name": f"optuna_trial_{trial.number}" # Unique name for this trial's model
        }
    }

    try:
        # Call the train_agent function. It will return the metric.
        # Ensure that `train_agent` function can accept the `config_override` parameter correctly.
        metric_value = train_agent(
            config_override=config_override,
            log_to_file=False, # Do not save logs for every trial
            metric_to_return=OPTIMIZATION_CONFIG["optimization_metric_name"]
        )
        
        # Check for invalid metric values (NaN, Inf) and prune if found
        if np.isinf(metric_value) or np.isnan(metric_value):
            raise optuna.TrialPruned(f"Trial {trial.number} pruned due to invalid metric value (inf or NaN): {metric_value}.")

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
            n_startup_trials=5, # Number of trials before pruning starts
            n_warmup_steps=OPTIMIZATION_CONFIG["pruning_interval_steps"] * 2, # Steps to wait before starting pruning
            interval_steps=OPTIMIZATION_CONFIG["pruning_interval_steps"] # Check for pruning every X steps
        )
    elif OPTIMIZATION_CONFIG["pruner"] == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    elif OPTIMIZATION_CONFIG["pruner"] == "sha": # Successive Halving Pruner
        pruner = optuna.pruners.SuccessiveHalvingPruner()

    try:
        study = optuna.create_study(
            study_name=OPTIMIZATION_CONFIG["study_name"],
            direction=OPTIMIZATION_CONFIG["optimization_metric_direction"],
            sampler=sampler,
            pruner=pruner, # Pass the pruner
            storage=OPTIMIZATION_CONFIG["db_url"], # Persist study to a database
            load_if_exists=True # Load existing study if it exists
        )
    except Exception as e:
        print(f"ERROR: Could not create or load Optuna study. Is the database URL '{OPTIMIZATION_CONFIG['db_url']}' valid and accessible? Error: {e}")
        sys.exit(1)

    print(f"Starting optimization for study '{OPTIMIZATION_CONFIG['study_name']}'...")
    print(f"Optimization direction: {OPTIMIZATION_CONFIG['optimization_metric_direction']} {OPTIMIZATION_CONFIG['optimization_metric_name']}")
    print(f"Number of existing trials: {len(study.trials)}")

    try:
        study.optimize(
            objective,
            n_trials=OPTIMIZATION_CONFIG["n_trials"] - len(study.trials), # Run remaining trials
            timeout=OPTIMIZATION_CONFIG["timeout_seconds"],
            gc_after_trial=True # Clean up memory after each trial
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during optimization: {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)

    print("\n--- Optimization Finished ---")
    if study.best_value is not None:
        print("\nBest trial:")
        print(f"  Value: {study.best_value:.4f}")
        print("  Parameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        # Optional: Save best parameters to a JSON file
        # Ensure the directory exists for the sqlite db first
        db_dir = os.path.dirname(OPTIMIZATION_CONFIG["db_url"].replace("sqlite:///", ""))
        if db_dir and not os.path.exists(db_dir): # Only create if not empty string (i.e., not in-memory)
            os.makedirs(db_dir, exist_ok=True)
        
        best_params_path = os.path.join(db_dir or ".", "best_hyperparameters.json") # Save to current dir if db_dir is empty

        try:
            with open(best_params_path, "w") as f:
                json.dump(convert_to_native_types(study.best_params), f, indent=4)
            print(f"\nBest hyperparameters saved to: {best_params_path}")
        except Exception as e:
            print(f"Error saving best hyperparameters: {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
    else:
        print("No best trial found (e.g., all trials pruned or failed).")


    # Optional: Plotting optimization results (requires plotly and matplotlib)
    if PLOTTING_AVAILABLE:
        try:
            if hasattr(vis, 'plot_optimization_history'):
                fig = vis.plot_optimization_history(study)
                fig.show()
            if hasattr(vis, 'plot_param_importances'):
                fig = vis.plot_param_importances(study)
                fig.show()
        except Exception as e:
            print(f"Error generating Optuna plots (ensure enough trials completed successfully for plots to render): {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)