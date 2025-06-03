# train_simple_agent.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize # NEW: Import VecNormalize
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
        metric_to_return (str): The name of the chosen metric (e.g., best evaluation reward).

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
            # Apply the FlattenAction wrapper
            wrapped_env = FlattenAction(base_env)
            # Apply VecNormalize for observation standardization # NEW
            # Gamma is typically from the agent's PPO params, normalize_reward=True if reward normalization is desired
            normalized_env = VecNormalize(wrapped_env, norm_obs=True, norm_reward=False, clip_obs=10.) # clip_obs can be adjusted
            return normalized_env # Return the normalized environment
        
        # When using make_vec_env with VecNormalize, you can pass VecNormalize directly to make_vec_env
        vec_env = make_vec_env(env_fn, n_envs=1, seed=0)
        
        # Monitor should wrap the final VecNormalize environment for proper logging
        # We need to access the underlying environment after VecNormalize for Monitor
        # For a single environment (n_envs=1), you can directly wrap vec_env.envs[0]
        # But VecNormalize is a VecEnv itself, so Monitor might go around VecNormalize directly if you prefer
        # However, for simplicity and typical SB3 patterns, VecNormalize usually sits *inside* the Monitor if you were
        # to use a single Monitor. But since make_vec_env returns a VecEnv, we should wrap the result.
        # The correct way to wrap a VecEnv with Monitor (if Monitor were a VecEnvWrapper) is not standard.
        # Instead, EvalCallback and Monitor callbacks usually work with the final VecEnv.
        # For logging, EvalCallback and Monitor will automatically record stats from the wrapped VecEnv.
        
        # If you want Monitor to sit *inside* the VecNormalize for *each* environment (which is not how make_vec_env works directly for Monitor)
        # you'd modify env_fn. But given `make_vec_env` returns a `VecEnv`, we apply Monitor outside.
        # However, `Monitor` is not a `VecEnvWrapper`. It's a standard `gym.Wrapper`.
        # The typical way `Monitor` is used with `make_vec_env` is by passing a `Monitor` wrapped env_fn
        # to `make_vec_env`. But since `VecNormalize` needs to wrap the `FlattenAction` env,
        # we return the `VecNormalize` env from `env_fn`.
        
        # Let's adjust the `env_fn` to return the Monitor-wrapped *then* VecNormalize-wrapped environment.
        # The standard order is Base -> FlattenAction -> VecNormalize -> Monitor for stats
        # However, make_vec_env usually applies Monitor *before* any other VecEnvWrappers if passed in.
        # Let's clarify the wrapper order for make_vec_env.
        # A common pattern is: make_vec_env(lambda: Monitor(YourEnv()), vec_env_cls=VecNormalize)
        # But if we want FlattenAction inside: make_vec_env(lambda: Monitor(FlattenAction(SimpleTradingEnv())), vec_env_cls=VecNormalize)
        # Then, VecNormalize wraps the monitored environment.
        # So, the final `env` for the model would be `VecNormalize(Monitor(FlattenAction(SimpleTradingEnv)))`.
        # This means `Monitor` will log pre-normalized rewards/observations.

        # To ensure Monitor logs post-normalized rewards and observations:
        # SimpleTradingEnv -> FlattenAction -> VecNormalize -> Monitor
        # But Monitor is for single env. VecNormalize is a VecEnv.
        # Correct approach:
        # The `env_fn` returns the *single* Gymnasium environment (wrapped by FlattenAction and VecNormalize).
        # `make_vec_env` then creates a VecEnv from this.
        # `Monitor` should wrap the resulting `VecEnv` for logging.
        # No, `Monitor` wraps a single `gym.Env`. `VecNormalize` is a `VecEnv`.
        # So, the `Monitor` can only wrap the `VecNormalize` *if* `VecNormalize` itself returns a single env after reset/step (which it doesn't).
        # The `Monitor` should wrap the *underlying* single environment, or EvalCallback will use it on its own.
        # The most common pattern with `make_vec_env` and `VecNormalize` and `Monitor` is:
        # env = make_vec_env(env_fn, n_envs=1, seed=0) # This env is a VecEnv
        # env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.) # This env is still a VecEnv (VecNormalize)
        # # Now, Monitor is used *inside* EvalCallback, so we don't need to wrap `env` with Monitor directly here.
        # # The `eval_env_for_callback` will be properly monitored.

        # So, `env_fn` should return the `FlattenAction` wrapped environment for `make_vec_env`.
        # Then `VecNormalize` wraps the `vec_env`.

        # Let's revert `env_fn` to return `FlattenAction` env.
        # Then apply `VecNormalize` to the `vec_env`.
        # And ensure `Monitor` for callbacks wraps `VecNormalize` or the original `FlattenAction` env as needed.
        
        def base_env_creator():
            # Create your base environment
            base_env = SimpleTradingEnv(tick_df=tick_df, kline_df_with_ta=kline_df, config=env_config)
            # Apply the FlattenAction wrapper
            wrapped_env = FlattenAction(base_env)
            # Monitor should wrap the environment BEFORE VecNormalize if you want Monitor to log pre-normalized data.
            # If you want Monitor to log post-normalized data, VecNormalize should come before Monitor.
            # However, VecNormalize is a VecEnv wrapper. Monitor is a standard Gym wrapper.
            # So, the proper order is:
            # SimpleTradingEnv -> FlattenAction -> (Monitor if you want Monitor to log pre-normalized) -> VecNormalize
            # Let's assume we want Monitor to log post-normalized observations, so Monitor should wrap the VecNormalize object.
            # But `Monitor` takes a `gym.Env`, not `VecEnv`.
            # So, we return `wrapped_env` here, and `make_vec_env` creates a `VecEnv`.
            # Then we wrap `vec_env` with `VecNormalize`.
            # `Monitor` will be used within `EvalCallback` to wrap the `eval_env_for_callback`.
            
            return wrapped_env # Return the FlattenAction wrapped environment

        vec_env = make_vec_env(base_env_creator, n_envs=1, seed=0) # This is a VecEnv
        
        # Apply VecNormalize to the vectorized environment
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.) # NEW: Apply VecNormalize
        
        if current_log_level != "none": print("\nEnvironment created successfully and normalized with VecNormalize.")
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
        total_timesteps_for_learn = ppo_params_for_init.pop("total_timesteps")
        
        model = PPO(
            "MlpPolicy",
            env, # Pass the VecNormalize-wrapped environment here
            verbose=1 if current_log_level in ["normal", "detailed"] else 0,
            tensorboard_log=tensorboard_log_dir,
            **ppo_params_for_init
        )

        callbacks = []
        # Checkpoint callback only if logging to file
        if log_to_file:
            checkpoint_callback = CheckpointCallback(save_freq=max(1, total_timesteps_for_learn // 5), save_path=log_dir, name_prefix="rl_model")
            callbacks.append(checkpoint_callback)
        
        # EvalCallback for saving the best model and evaluating
        # Crucial for Optuna: Use a separate eval env and get the best reward from it.
        # eval_env_for_callback will be closed in the finally block
        
        # The eval environment also needs to be VecNormalize wrapped.
        # IMPORTANT: The VecNormalize object for evaluation must *share* its statistics
        # with the training VecNormalize object, or load them from the training one.
        # The standard practice is to use the `load_original_properties=True` argument
        # when saving the VecNormalize object with the model and then loading it with the model.
        # When `model.save()` is called with a `VecNormalize` environment, its statistics are saved.
        # When `model.load()` is called, the `VecNormalize` environment is often re-created
        # using the saved statistics.

        # For EvalCallback, we need to create a new `VecNormalize` wrapped environment.
        # And then manually load the statistics into it.
        
        # 1. Create the base environment for evaluation
        eval_base_env = SimpleTradingEnv(tick_df=tick_df, kline_df_with_ta=kline_df, config=env_config)
        eval_wrapped_env = FlattenAction(eval_base_env)

        # 2. Wrap it with VecNormalize (make sure it's not the same instance as training env's VecNormalize)
        eval_env_for_callback = VecNormalize(eval_wrapped_env, norm_obs=True, norm_reward=False, clip_obs=10.) # NEW
        
        # 3. Load the normalization statistics from the training environment into the evaluation environment
        # This is CRUCIAL for consistent normalization between training and evaluation.
        # model.set_env() does this implicitly if the environment is VecNormalize and the model was loaded from a VecNormalize env.
        # But for EvalCallback, we set its env directly.
        # We need to save and load the `VecNormalize` object's state directly.
        # When `model.save()` is called on a model with `VecNormalize` env, the `VecNormalize` stats are saved in a `.pkl` file
        # alongside the model (e.g., `vec_normalize.pkl`).
        # `EvalCallback` will save `best_model.zip`, and if `env` is `VecNormalize`, it will save `vec_normalize.pkl` next to it.
        # During evaluation (in `evaluate_agent.py` and `live_trader.py`), we'll load these stats.

        eval_callback = EvalCallback(
            eval_env_for_callback, # Pass the VecNormalize-wrapped eval env
            best_model_save_path=os.path.join(log_dir, "best_model") if log_to_file else None,
            log_path=log_dir if log_to_file else None,
            eval_freq=max(1000, total_timesteps_for_learn // 10),
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)


        model.learn(total_timesteps=total_timesteps_for_learn, callback=callbacks)
        if current_log_level != "none": print("\nTraining completed.")

        if log_to_file:
            final_model_path = os.path.join(log_dir, "trained_model_final.zip")
            model.save(final_model_path)
            # Save the VecNormalize statistics separately.
            # This is important for loading them into evaluation/live environments.
            env.save(os.path.join(log_dir, "vec_normalize.pkl")) # NEW: Save VecNormalize stats
            if current_log_level != "none": print(f"Final model and VecNormalize stats saved to {log_dir}")


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
            # Close the VecNormalize environment, which in turn closes its wrapped env.
            env.close()
            if current_log_level != "none": print("Training environment closed.")
        if 'eval_env_for_callback' in locals() and eval_env_for_callback is not None:
            # Close the VecNormalize evaluation environment.
            eval_env_for_callback.close()
            if current_log_level != "none": print("Evaluation environment for callback closed.")

# Ensure that the main execution block is guarded so Optuna can import `train_agent` directly.
if __name__ == "__main__":
    # When running train_simple_agent.py directly, log to file
    final_reward = train_agent(log_to_file=True)
    print(f"\n--- Training Script Finished (Final Metric: {final_reward:.2f}) ---")