import os
import optuna
import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime

# --- UPDATED IMPORTS ---
from src.agents.train_agent import train_agent
from src.data.config_loader import load_config, merge_configs, convert_to_native_types
# --- END UPDATED IMPORTS ---

try:
    from optuna.callbacks import TqdmCallback
except ImportError:
    try:
        from optuna.integration import TqdmCallback
    except ImportError:
        TqdmCallback = None
        print("WARNING: Optuna TqdmCallback not found. Progress bar for HPO will not be available.")

def load_default_configs_for_optimization(config_dir="configs/defaults") -> dict:
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "environment.yaml"),
        os.path.join(config_dir, "binance_settings.yaml"),
        os.path.join(config_dir, "evaluation_data.yaml"),
        os.path.join(config_dir, "hash_keys.yaml"),
        os.path.join(config_dir, "hyperparameter_optimization.yaml"),
        os.path.join(config_dir, "ppo_params.yaml"),
        os.path.join(config_dir, "sac_params.yaml"),
        os.path.join(config_dir, "ddpg_params.yaml"),
        os.path.join(config_dir, "a2c_params.yaml"),
        os.path.join(config_dir, "recurrent_ppo_params.yaml"),
        os.path.join(config_dir, "live_trader_settings.yaml"),
    ]
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)

def objective(trial: optuna.Trial, base_effective_config: dict) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    It suggests hyperparameters, constructs a new config, and calls train_agent.
    """
    # Use a copy of the base_effective_config for this trial to prevent contamination
    trial_config = base_effective_config.copy() # Shallow copy might be enough if nested dicts are replaced, not modified

    # Deep copy relevant sections to avoid modifying the original base_effective_config dicts
    # if they are further modified in this function (e.g. trial_config["ppo_params"] = base_config["ppo_params"].copy())
    for key in ["ppo_params", "sac_params", "ddpg_params", "a2c_params", "recurrent_ppo_params", "environment", "hyperparameter_optimization", "run_settings"]:
        if key in trial_config:
            trial_config[key] = trial_config[key].copy()


    optimizer_settings = trial_config.get("hyperparameter_optimization", {})
    run_settings_for_trial = trial_config.get("run_settings", {}) # Get run_settings for this trial

    agent_type = trial_config.get("agent_type", "PPO")

    if agent_type == "PPO":
        ppo_optim_params = optimizer_settings.get("ppo_optim_params", {})
        trial_config["ppo_params"]["learning_rate"] = trial.suggest_float("learning_rate", **ppo_optim_params.get("learning_rate", {"low": 1e-5, "high": 1e-3, "log": True}))
        trial_config["ppo_params"]["n_steps"] = trial.suggest_categorical("n_steps", **ppo_optim_params.get("n_steps", {"choices": [256, 512, 1024, 2048]}))
        trial_config["ppo_params"]["gamma"] = trial.suggest_float("gamma", **ppo_optim_params.get("gamma", {"low": 0.9, "high": 0.999}))
        net_arch_str = trial.suggest_categorical("policy_kwargs_net_arch", **ppo_optim_params.get("policy_kwargs_net_arch", {"choices": ["[64, 64]", "[128, 128]"]}))
        trial_config["ppo_params"]["policy_kwargs"] = {"net_arch": eval(net_arch_str)}
        trial_config["ppo_params"]["total_timesteps"] = optimizer_settings.get("trial_total_timesteps", 100000) # Ensure this is used

    elif agent_type == "SAC":
        sac_optim_params = optimizer_settings.get("sac_optim_params", {})
        trial_config["sac_params"]["learning_rate"] = trial.suggest_float("learning_rate", **sac_optim_params.get("learning_rate", {"low": 1e-5, "high": 1e-3, "log": True}))
        trial_config["sac_params"]["buffer_size"] = trial.suggest_categorical("buffer_size", **sac_optim_params.get("buffer_size", {"choices": [100000, 1000000]}))
        trial_config["sac_params"]["gamma"] = trial.suggest_float("gamma", **sac_optim_params.get("gamma", {"low": 0.9, "high": 0.999}))
        trial_config["sac_params"]["tau"] = trial.suggest_float("tau", **sac_optim_params.get("tau", {"low": 0.001, "high": 0.01}))
        trial_config["sac_params"]["gradient_steps"] = trial.suggest_categorical("gradient_steps", **sac_optim_params.get("gradient_steps", {"choices": [1, -1]}))
        net_arch_str = trial.suggest_categorical("sac_policy_kwargs_net_arch", **sac_optim_params.get("policy_kwargs_net_arch", {"choices": ["[256, 256]"]}))
        trial_config["sac_params"]["policy_kwargs"] = {"net_arch": eval(net_arch_str)}
        trial_config["sac_params"]["total_timesteps"] = optimizer_settings.get("trial_total_timesteps", 100000)

    env_optim_params = optimizer_settings.get("env_optim_params", {})
    if "kline_window_size" in env_optim_params:
        trial_config["environment"]["kline_window_size"] = trial.suggest_categorical("kline_window_size", **env_optim_params["kline_window_size"])
    if "tick_feature_window_size" in env_optim_params:
        trial_config["environment"]["tick_feature_window_size"] = trial.suggest_categorical("tick_feature_window_size", **env_optim_params["tick_feature_window_size"])

    # Ensure run_settings log_level is 'none' for Optuna trials to keep console clean
    # The objective function's prints (like the one below) are still useful for tracking.
    if "run_settings" not in trial_config: trial_config["run_settings"] = {}
    trial_config["run_settings"]["log_level"] = "none"


    print(f"Optuna Trial {trial.number} starting with agent '{agent_type}'. Params: {json.dumps(convert_to_native_types(trial.params), indent=2)}")

    try:
        metric_value = train_agent(config_override=trial_config, log_to_file=False) # log_to_file=False for Optuna trials

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        print(f"Optuna Trial {trial.number} finished. Metric: {metric_value}")
        return metric_value
    except optuna.exceptions.TrialPruned as e_prune:
        print(f"Optuna Trial {trial.number} pruned.")
        raise e_prune # Re-raise for Optuna to handle
    except Exception as e:
        print(f"Optuna Trial {trial.number} failed due to an exception in train_agent or objective: {e}")
        traceback.print_exc()
        return -np.inf # Return a very bad value for failed trials not caught by pruning

def main():
    print("--- Starting Hyperparameter Optimization with Optuna ---")

    base_effective_config = load_default_configs_for_optimization()
    optimizer_settings = base_effective_config.get("hyperparameter_optimization", {})
    run_settings = base_effective_config.get("run_settings", {}) # For TqdmCallback condition

    study_name = optimizer_settings.get("study_name", "trading_agent_optimization")
    db_file = optimizer_settings.get("db_file", "optuna_study.db")

    optuna_studies_dir = "optuna"
    os.makedirs(optuna_studies_dir, exist_ok=True)
    optuna_study_db_path = os.path.join(optuna_studies_dir, db_file)
    storage_url = f"sqlite:///{optuna_study_db_path}"

    print(f"Optuna Study Name: {study_name}")
    print(f"Optuna Study Database: {storage_url}")
    print(f"Number of trials: {optimizer_settings.get('n_trials', 50)}")

    sampler_type = optimizer_settings.get("sampler_type", "TPESampler")
    sampler_seed = optimizer_settings.get("seed", None) # Allow None seed for non-deterministic
    if sampler_type == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    elif sampler_type == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
    else:
        print(f"WARNING: Unknown sampler type '{sampler_type}'. Using default TPESampler.")
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)

    pruner_config = {
        "n_startup_trials": optimizer_settings.get("pruner_n_startup_trials", 5),
        "n_warmup_steps": optimizer_settings.get("pruner_n_warmup_steps", 0),
        "interval_steps": optimizer_settings.get("pruner_interval_steps", 1),
    }
    pruner = optuna.pruners.MedianPruner(**pruner_config) if optimizer_settings.get("use_pruner", True) else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=optimizer_settings.get("load_if_exists", True),
        direction=optimizer_settings.get("direction", "maximize"),
        sampler=sampler,
        pruner=pruner
    )

    func_to_optimize = lambda trial: objective(trial, base_effective_config)
    
    # Setup TqdmCallback conditionally
    optuna_callbacks = []
    if TqdmCallback and run_settings.get("log_level", "normal") != "none":
        optuna_callbacks.append(TqdmCallback())

    try:
        study.optimize(
            func_to_optimize,
            n_trials=optimizer_settings.get("n_trials", 50),
            timeout=optimizer_settings.get("timeout_seconds"), # Can be None
            callbacks=optuna_callbacks if optuna_callbacks else None # Pass list or None
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e: # Catch other exceptions during optimize call
        print(f"An unexpected error occurred during optimization: {e}")
        traceback.print_exc()

    print("\n--- Optimization Finished ---")
    try:
        # This block might fail if no trials completed successfully
        print(f"Number of finished trials: {len(study.trials)}")
        if study.best_trial: # Check if best_trial exists
            print(f"Best trial (number {study.best_trial.number}):")
            print(f"  Value: {study.best_trial.value}")
            print("  Params: ")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")

            best_params_path = os.path.join(optuna_studies_dir, f"best_hyperparameters_{study_name}.json")
            try:
                with open(best_params_path, 'w') as f:
                    json.dump(convert_to_native_types(study.best_trial.params), f, indent=4)
                print(f"Best hyperparameters saved to: {best_params_path}")
            except Exception as e_save:
                print(f"Error saving best hyperparameters: {e_save}")
        else:
            print("No best trial found (e.g., all trials failed or were pruned before completion).")

    except ValueError as e_study_access: # Catch errors like "Record does not exist"
        print(f"Could not retrieve study results (e.g., no trials completed successfully): {e_study_access}")
    except Exception as e_final:
        print(f"An error occurred while finalizing or reporting HPO results: {e_final}")
        traceback.print_exc()


if __name__ == "__main__":
    main()