# train_simple_agent.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd
import numpy as np
import json
import traceback

# Assuming utils.py, base_env.py, and custom_wrappers.py are in the same directory or accessible via PYTHONPATH
from utils import load_config, fetch_and_cache_kline_data, fetch_continuous_aggregate_trades, merge_configs, generate_config_hash, get_relevant_config_for_hash, convert_to_native_types
from base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG # Import DEFAULT_ENV_CONFIG for merging
from custom_wrappers import FlattenAction # Import your new custom wrapper

# Default configuration (used if config.yaml is not found or incomplete)
# This should ideally mirror the structure of your config.yaml
DEFAULT_TRAIN_CONFIG = {
    "run_settings": {
        "log_dir_base": "./logs/ppo_trading/",
        "model_name": "tick_trading_agent",
        "log_level": "normal",
        "live_log_dir": "./logs/live_trading/", # Default for live log dir
        "model_path": None,
        "refresh_interval_sec": 1,
        "data_processing_interval_ms": 100
    },
    "environment": DEFAULT_ENV_CONFIG, # Use the default from base_env.py
    "ppo_params": {
        "total_timesteps": 100000, # Reduced for quick testing for training
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "policy_kwargs": "{'net_arch': [64, 64]}" # Python dict here
    },
    "binance_settings": {
        "default_symbol": "BTCUSDT",
        "historical_interval": "1h",
        "historical_cache_dir": "./binance_data_cache/",
        "start_date_kline_data": "2024-01-01",
        "end_date_kline_data": "2024-01-03",
        "start_date_tick_data": "2024-01-01 00:00:00",
        "end_date_tick_data": "2024-01-03 00:00:00",
        "api_key": os.environ.get("BINANCE_API_KEY"),
        "api_secret": os.environ.get("BINANCE_API_SECRET"),
        "testnet": False,
        "api_request_delay_seconds": 0.2
    },
    "binance_api_client": { # Default for API client settings
        "timeout_seconds": 10,
        "recv_window_ms": 5000
    },
    "hash_config_keys": { # Ensure this is consistent with config.yaml
        "environment": [
            "kline_window_size", "tick_feature_window_size", "initial_balance", "commission_pct",
            "base_trade_amount_ratio", "catastrophic_loss_threshold_pct", "min_profit_target_low",
            "min_profit_target_high", "reward_open_buy_position", "penalty_buy_insufficient_balance",
            "penalty_buy_position_already_open", "reward_sell_profit_base", "reward_sell_profit_factor",
            "penalty_sell_loss_factor", "penalty_sell_loss_base", "penalty_sell_no_position",
            "reward_hold_profitable_position", "penalty_hold_losing_position", "penalty_hold_flat_position",
            "penalty_catastrophic_loss", "reward_eof_sell_factor", "reward_sell_meets_target_bonus",
            "penalty_sell_profit_below_target",
            "kline_price_features", # Added to hash for TA features
            "tick_features_to_use" # Added for completeness
        ],
        "ppo_params": [
            "total_timesteps", "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
            "target_kl", "policy_kwargs"
        ],
        "binance_settings": [
            "default_symbol", "historical_interval", 
            # Not including start/end dates for hashing unless specific data periods define different experiments
            # "start_date_kline_data", "end_date_kline_data", "start_date_tick_data", "end_date_tick_data",
            "testnet", "api_request_delay_seconds"
        ],
    }
}

# --- New Function: `train_agent` ---
def train_agent(
    config_override: dict = None,
    log_to_file: bool = True,
    metric_to_return: str = "eval_reward" # Can be "eval_reward", "final_equity_pct" etc.
) -> float:
    """
    Trains the PPO agent based on the provided configuration.
    Can be called directly or by an optimizer like Optuna.

    Args:
        config_override (dict): A dictionary of configuration parameters to override
                                the default or loaded config.
        log_to_file (bool): If True, logs are saved to disk. If False (e.g., for
                            Optuna trials), logs are suppressed or minimal.
        metric_to_return (str): The name of the metric to return for optimization.

    Returns:
        float: The value of the chosen metric (e.g., best evaluation reward).
               Returns -np.inf if the trial fails.
    """
    print("--- Starting Trading Agent Training Process ---")

    # Load configuration
    try:
        loaded_config = load_config("config.yaml")
        config = merge_configs(DEFAULT_TRAIN_CONFIG, loaded_config)
        if config_override:
            config = merge_configs(config, config_override) # Apply overrides
    except Exception as e:
        print(f"Error loading or merging configuration: {e}. Using default training config.")
        config = DEFAULT_TRAIN_CONFIG.copy()
        if config_override: # Apply overrides even if default is used
            config = merge_configs(config, config_override)

    # Convert policy_kwargs string to dict if loaded from YAML
    if isinstance(config["ppo_params"]["policy_kwargs"], str):
        try:
            config["ppo_params"]["policy_kwargs"] = eval(config["ppo_params"]["policy_kwargs"])
        except Exception as e:
            print(f"Warning: Could not parse policy_kwargs string '{config['ppo_params']['policy_kwargs']}': {e}. Using default.")
            config["ppo_params"]["policy_kwargs"] = DEFAULT_TRAIN_CONFIG["ppo_params"]["policy_kwargs"]

    run_settings = config["run_settings"]
    env_config = config["environment"]
    ppo_params = config["ppo_params"]
    binance_settings = config["binance_settings"]
    
    # Adjust log level based on log_to_file
    current_log_level = run_settings.get("log_level", "normal")
    if not log_to_file: # Suppress excessive logging during optimization trials
        current_log_level = "none"
        env_config["custom_print_render"] = "none" # Ensure no rendering

    # Generate a unique run ID based on hashed configuration
    # Only hash if logging to file, otherwise it might create many useless directories
    run_id = "optuna_trial" # Default for Optuna trials
    log_dir = "./optuna_temp_logs" # Default temp log dir for Optuna
    if log_to_file:
        relevant_config_for_hash = get_relevant_config_for_hash(
            config, DEFAULT_TRAIN_CONFIG, DEFAULT_ENV_CONFIG
        )
        config_hash = generate_config_hash(relevant_config_for_hash)
        run_id = f"{config_hash}_{run_settings['model_name']}"
        log_dir = os.path.join(run_settings['log_dir_base'], run_id)
        os.makedirs(log_dir, exist_ok=True)
    
    # Override tensorboard_log if not logging to file
    tensorboard_log_dir = log_dir if log_to_file else None


    print(f"Training run ID: {run_id} (Log Level: {current_log_level})")
    if log_to_file:
        print(f"Logs and models will be saved to: {log_dir}")
        # Save the effective configuration for this run
        with open(os.path.join(log_dir, "effective_config.json"), "w") as f:
            json.dump(convert_to_native_types(config), f, indent=4)
        if current_log_level != "none":
            print("Effective configuration saved to effective_config.json")
            if current_log_level == "detailed":
                print("\n--- Effective Configuration for this run ---")
                print(json.dumps(convert_to_native_types(config), indent=2, sort_keys=True))
                print("-------------------------------------------\n")
    else:
        print(f"Running Optuna trial. Logs are suppressed.")

    # --- Data Fetching ---
    if current_log_level != "none": print("\n--- Fetching and preparing K-line data ---")
    kline_df = pd.DataFrame() # Initialize
    try:
        kline_df = fetch_and_cache_kline_data(
            symbol=binance_settings["default_symbol"],
            interval=binance_settings["historical_interval"],
            start_date_str=binance_settings["start_date_kline_data"],
            end_date_str=binance_settings["end_date_kline_data"],
            cache_dir=binance_settings["historical_cache_dir"],
            price_features_to_add=env_config["kline_price_features"], # Pass features including TAs
            api_key=binance_settings["api_key"],
            api_secret=binance_settings["api_secret"],
            testnet=binance_settings["testnet"],
            cache_file_type=binance_settings.get("cache_file_type", "parquet"),
            log_level=current_log_level,
            api_request_delay_seconds=binance_settings.get("api_request_delay_seconds", 0.2)
        )
        if kline_df.empty:
            raise ValueError("K-line data not loaded. Cannot proceed with training.")
        if current_log_level != "none": print(f"K-line data loaded: {kline_df.shape} from {kline_df.index.min()} to {kline_df.index.max()}")
    except Exception as e:
        print(f"ERROR: K-line data not loaded. Cannot proceed with training. Details: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials

    if current_log_level != "none": print(f"\n--- Fetching and preparing Tick data from {binance_settings['start_date_tick_data']} to {binance_settings['end_date_tick_data']} ---")
    tick_df = pd.DataFrame() # Initialize
    try:
        tick_df = fetch_continuous_aggregate_trades(
            symbol=binance_settings["default_symbol"],
            start_date_str=binance_settings["start_date_tick_data"],
            end_date_str=binance_settings["end_date_tick_data"],
            cache_dir=binance_settings["historical_cache_dir"],
            api_key=binance_settings["api_key"],
            api_secret=binance_settings["api_secret"],
            testnet=binance_settings["testnet"],
            cache_file_type=binance_settings.get("cache_file_type", "parquet"),
            log_level=current_log_level,
            api_request_delay_seconds=binance_settings.get("api_request_delay_seconds", 0.2)
        )
        if tick_df.empty:
            raise ValueError("Tick data not loaded. Cannot proceed with training.")
        if current_log_level != "none": print(f"Tick data loaded: {tick_df.shape} from {tick_df.index.min()} to {tick_df.index.max()}")
    except Exception as e:
        print(f"ERROR: Tick data not loaded. Cannot proceed with training. Details: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials

    # --- Environment Setup ---
    # Ensure env_config log_level matches run_settings log_level for consistency
    env_config["log_level"] = current_log_level
    env_config["custom_print_render"] = "none"

    env = None
    eval_env_for_callback = None # To ensure it's closed in finally block
    try:
        def env_fn():
            # Create your base environment
            base_env = SimpleTradingEnv(tick_df=tick_df, kline_df_with_ta=kline_df, config=env_config)
            # Apply the FlattenAction wrapper to your base environment
            wrapped_env = FlattenAction(base_env) # Apply the wrapper here
            return wrapped_env

        vec_env = make_vec_env(env_fn, n_envs=1, seed=0)
        # Wrap the *single underlying environment* (which is now your FlattenAction wrapped env) with Monitor
        env = Monitor(vec_env.envs[0], filename=os.path.join(log_dir, "monitor.csv") if log_to_file else None)
        
        if current_log_level != "none": print("\nEnvironment created successfully.")
    except Exception as e:
        print(f"ERROR: Failed to create training environment. Details: {e}")
        traceback.print_exc()
        return -np.inf

    # --- Agent Training ---
    if current_log_level != "none": print("\n--- Initializing and Training PPO Agent ---")
    model = None
    try:
        # Extract total_timesteps before passing ppo_params to PPO constructor
        # Ensure it's a copy so the original ppo_params (from config) is not modified for hashing
        ppo_params_for_init = ppo_params.copy()
        total_timesteps_for_learn = ppo_params_for_init.pop("total_timesteps") # FIX: Remove total_timesteps from dict here
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1 if current_log_level in ["normal", "detailed"] else 0,
            tensorboard_log=tensorboard_log_dir,
            **ppo_params_for_init # FIX: Unpack modified dictionary here
        )

        callbacks = []
        # Checkpoint callback only if logging to file
        if log_to_file:
            checkpoint_callback = CheckpointCallback(save_freq=max(1, total_timesteps_for_learn // 5), save_path=log_dir, name_prefix="rl_model")
            callbacks.append(checkpoint_callback)
        
        # EvalCallback for saving the best model and evaluating
        # Crucial for Optuna: Use a separate eval env and get the best reward from it.
        # eval_env_for_callback will be closed in the finally block
        eval_env_for_callback = Monitor(env_fn(), filename=os.path.join(log_dir, "eval_monitor.csv") if log_to_file else None)
        
        eval_callback = EvalCallback(
            eval_env_for_callback,
            best_model_save_path=os.path.join(log_dir, "best_model") if log_to_file else None,
            log_path=log_dir if log_to_file else None,
            eval_freq=max(1000, total_timesteps_for_learn // 10), # Use the correct total_timesteps here
            deterministic=True,
            render=False,
            # Pass Optuna trial for pruning if needed (optional for this current setup)
            # callback_after_eval=lambda model, evaluation_env, callback: trial.report(callback.best_mean_reward, callback.num_timesteps) if 'trial' in locals() else None
        )
        callbacks.append(eval_callback)


        model.learn(total_timesteps=total_timesteps_for_learn, callback=callbacks) # FIX: Pass total_timesteps to .learn()
        if current_log_level != "none": print("\nTraining completed.")

        if log_to_file:
            final_model_path = os.path.join(log_dir, "trained_model_final.zip")
            model.save(final_model_path)
            if current_log_level != "none": print(f"Final model saved to {final_model_path}")

        # Return the best evaluation reward found by EvalCallback
        # EvalCallback saves the best reward in its `best_mean_reward` attribute
        if metric_to_return == "eval_reward" and eval_callback.best_mean_reward is not None:
            return float(eval_callback.best_mean_reward) # Ensure float type
        elif metric_to_return == "final_equity_pct":
            # This would require running an evaluation at the end of the trial
            # and getting the final equity from it. For now, we stick to eval_reward.
            print("WARNING: 'final_equity_pct' metric not yet implemented for direct return in train_agent.")
            return float(eval_callback.best_mean_reward) if eval_callback.best_mean_reward is not None else -np.inf
        else:
            print("WARNING: EvalCallback did not record a best mean reward or unknown metric requested. Returning negative infinity.")
            return -np.inf

    except Exception as e:
        print(f"ERROR: An error occurred during agent training. Details: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials
    finally:
        if 'env' in locals() and env is not None:
            env.close()
            if current_log_level != "none": print("Training environment closed.")
        if 'eval_env_for_callback' in locals() and eval_env_for_callback is not None:
            eval_env_for_callback.close()
            if current_log_level != "none": print("Evaluation environment for callback closed.")

# Ensure that the main execution block is guarded so Optuna can import `train_agent` directly.
if __name__ == "__main__":
    # When running train_simple_agent.py directly, log to file
    final_reward = train_agent(log_to_file=True)
    print(f"\n--- Training Script Finished (Final Metric: {final_reward:.2f}) ---")