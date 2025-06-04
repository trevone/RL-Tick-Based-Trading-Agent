# hyperparameter_optimization.py

import os
import optuna
import pandas as pd
import numpy as np
import json
import yaml
import traceback
from datetime import datetime

# Import the train_agent function and utility functions from the new structure
from src.agents.train_agent import train_agent
from src.data.utils import load_config, merge_configs, convert_to_native_types


# --- NEW: Function to load default configurations for optimization ---
def load_default_configs_for_optimization(config_dir="configs/defaults") -> dict:
    """Loads default configurations from the specified directory for hyperparameter optimization."""
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "environment.yaml"),
        os.path.join(config_dir, "binance_settings.yaml"),
        os.path.join(config_dir, "evaluation_data.yaml"),
        os.path.join(config_dir, "hash_keys.yaml"), # Needed for generating unique run IDs
        os.path.join(config_dir, "hyperparameter_optimization.yaml"), # New settings file for Optuna
        os.path.join(config_dir, "ppo_params.yaml"), # Include all algo params for potential optimization
        os.path.join(config_dir, "sac_params.yaml"),
        os.path.join(config_dir, "ddpg_params.yaml"),
        os.path.join(config_dir, "a2c_params.yaml"),
        os.path.join(config_dir, "recurrent_ppo_params.yaml"),
    ]
    
    # Use the new load_config from src.data.utils which merges multiple files
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)


def objective(trial: optuna.Trial, base_effective_config: dict) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    It suggests hyperparameters, constructs a new config, and calls train_agent.
    """
    # Use a copy of the base_effective_config for this trial to prevent contamination
    trial_config = base_effective_config.copy()
    
    optimizer_settings = trial_config.get("hyperparameter_optimization", {})
    
    agent_type = trial_config.get("agent_type", "PPO") # Get agent type from base config

    # Dynamically suggest hyperparameters based on the chosen agent_type
    # The ranges and choices are defined in the 'hyperparameter_optimization' section of the config.
    # The keys in the config (e.g., 'ppo_optim_params') map to the algorithm being optimized.

    # Example for PPO parameters:
    if agent_type == "PPO":
        ppo_optim_params = optimizer_settings.get("ppo_optim_params", {})
        
        # Suggest learning rate
        lr_low = ppo_optim_params.get("learning_rate", {}).get("low", 1e-5)
        lr_high = ppo_optim_params.get("learning_rate", {}).get("high", 1e-3)
        log_lr = ppo_optim_params.get("learning_rate", {}).get("log", True)
        trial_config["ppo_params"]["learning_rate"] = trial.suggest_float("learning_rate", lr_low, lr_high, log=log_lr)

        # Suggest n_steps (rollout buffer size)
        n_steps_choices = ppo_optim_params.get("n_steps", {}).get("choices", [256, 512, 1024, 2048])
        trial_config["ppo_params"]["n_steps"] = trial.suggest_categorical("n_steps", n_steps_choices)

        # Suggest gamma
        gamma_low = ppo_optim_params.get("gamma", {}).get("low", 0.9)
        gamma_high = ppo_optim_params.get("gamma", {}).get("high", 0.999)
        trial_config["ppo_params"]["gamma"] = trial.suggest_float("gamma", gamma_low, gamma_high)
        
        # Suggest network architecture (example: 2 layers, sizes)
        # Policy kwargs string must be eval-able
        net_arch_choices = ppo_optim_params.get("policy_kwargs_net_arch", {}).get("choices", ["[64, 64]", "[128, 128]"])
        trial_config["ppo_params"]["policy_kwargs"] = {"net_arch": eval(trial.suggest_categorical("policy_kwargs_net_arch", net_arch_choices))}

        # Adjust total_timesteps for trials if needed, usually fixed for optimization rounds
        # For Optuna trials, it's often set to a smaller value to speed up exploration
        trial_config["ppo_params"]["total_timesteps"] = optimizer_settings.get("trial_total_timesteps", 1000000)

    # Example for SAC parameters:
    elif agent_type == "SAC":
        sac_optim_params = optimizer_settings.get("sac_optim_params", {})
        
        trial_config["sac_params"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        trial_config["sac_params"]["buffer_size"] = trial.suggest_categorical("buffer_size", [100000, 500000, 1000000])
        trial_config["sac_params"]["gamma"] = trial.suggest_float("gamma", 0.9, 0.999)
        trial_config["sac_params"]["tau"] = trial.suggest_float("tau", 0.001, 0.01)
        trial_config["sac_params"]["gradient_steps"] = trial.suggest_categorical("gradient_steps", [1, -1])
        
        net_arch_choices = sac_optim_params.get("policy_kwargs_net_arch", {}).get("choices", ["[256, 256]", "[128, 128]"])
        trial_config["sac_params"]["policy_kwargs"] = {"net_arch": eval(trial.suggest_categorical("sac_policy_kwargs_net_arch", net_arch_choices))}

        trial_config["sac_params"]["total_timesteps"] = optimizer_settings.get("trial_total_timesteps", 500000)
    
    # Add similar blocks for DDPG, A2C, RecurrentPPO if you intend to optimize them.
    # For now, if a non-PPO/SAC agent type is chosen and not explicitly optimized, it will use its default parameters
    # from the base_effective_config.

    # You can also optimize environment parameters
    env_optim_params = optimizer_settings.get("env_optim_params", {})
    if "kline_window_size" in env_optim_params:
        kline_choices = env_optim_params["kline_window_size"].get("choices", [10, 20, 30])
        trial_config["environment"]["kline_window_size"] = trial.suggest_categorical("kline_window_size", kline_choices)
    
    if "tick_feature_window_size" in env_optim_params:
        tick_choices = env_optim_params["tick_feature_window_size"].get("choices", [20, 50, 100])
        trial_config["environment"]["tick_feature_window_size"] = trial.suggest_categorical("tick_feature_window_size", tick_choices)

    # Pass the modified config to train_agent.
    # Set log_to_file=False to avoid cluttering logs during optimization.
    print(f"Trial {trial.number} starting with parameters: {json.dumps(convert_to_native_types(trial.params), indent=2)}")
    
    try:
        # The train_agent function now returns a metric (e.g., best_mean_reward)
        metric_value = train_agent(config_override=trial_config, log_to_file=False)
        
        # Pruning: Stop trials that are performing poorly
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        print(f"Trial {trial.number} finished. Metric: {metric_value}")
        return metric_value
    except Exception as e:
        print(f"Trial {trial.number} failed due to an exception: {e}")
        traceback.print_exc()
        raise # Re-raise to let Optuna handle failed trials properly (e.g., mark as FAIL)


def main():
    print("--- Starting Hyperparameter Optimization with Optuna ---")

    # Load the base configuration once
    base_effective_config = load_default_configs_for_optimization()
    optimizer_settings = base_effective_config.get("hyperparameter_optimization", {})
    run_settings = base_effective_config.get("run_settings", {})

    study_name = optimizer_settings.get("study_name", "trading_agent_optimization")
    db_file = optimizer_settings.get("db_file", "optuna_study.db")
    
    # Path to the SQLite database
    optuna_study_db_path = os.path.join("optuna_studies", db_file)
    os.makedirs("optuna_studies", exist_ok=True)
    
    storage_url = f"sqlite:///{optuna_study_db_path}"

    print(f"Optuna Study Name: {study_name}")
    print(f"Optuna Study Database: {storage_url}")
    print(f"Number of trials: {optimizer_settings.get('n_trials', 50)}")

    sampler_type = optimizer_settings.get("sampler_type", "TPESampler")
    if sampler_type == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=optimizer_settings.get("seed", 42))
    elif sampler_type == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=optimizer_settings.get("seed", 42))
    else:
        print(f"WARNING: Unknown sampler type '{sampler_type}'. Using default TPESampler.")
        sampler = optuna.samplers.TPESampler(seed=optimizer_settings.get("seed", 42))

    # Create a study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=optimizer_settings.get("load_if_exists", True),
        direction=optimizer_settings.get("direction", "maximize"), # Maximize reward
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=optimizer_settings.get("pruner_n_startup_trials", 5),
            n_warmup_steps=optimizer_settings.get("pruner_n_warmup_steps", 0),
            interval_steps=optimizer_settings.get("pruner_interval_steps", 1),
        ) if optimizer_settings.get("use_pruner", True) else optuna.pruners.NopPruner()
    )

    # Wrap the objective function to pass the base_effective_config
    func_to_optimize = lambda trial: objective(trial, base_effective_config)

    # Run the optimization
    try:
        study.optimize(
            func_to_optimize,
            n_trials=optimizer_settings.get("n_trials", 50),
            timeout=optimizer_settings.get("timeout_seconds"),
            callbacks=[optuna.callbacks.TqdmCallback()] if run_settings.get("log_level", "normal") != "none" else None
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during optimization: {e}")
        traceback.print_exc()

    # Print results
    print("\n--- Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial (number {study.best_trial.number}):")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters to a JSON file
    best_params_path = os.path.join("optuna_studies", f"best_hyperparameters_{study_name}.json")
    try:
        with open(best_params_path, 'w') as f:
            json.dump(convert_to_native_types(study.best_trial.params), f, indent=4)
        print(f"Best hyperparameters saved to: {best_params_path}")
    except Exception as e:
        print(f"Error saving best hyperparameters: {e}")
        traceback.print_exc()

    # You can also print / save all trials results
    # trials_df = study.trials_dataframe()
    # trials_df.to_csv(os.path.join("optuna_studies", f"{study_name}_trials_results.csv"), index=False)
    # print(f"All trials results saved to {study_name}_trials_results.csv")

if __name__ == "__main__":
    main()