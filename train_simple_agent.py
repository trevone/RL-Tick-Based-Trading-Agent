# train_simple_agent.py

import os
import json
import yaml
import warnings
from datetime import datetime
import traceback

import pandas as pd
import numpy as np

# Suppress pandas FutureWarnings related to 'pd.concat'
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # Keep both for flexibility
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize # Crucial for obs normalization

# Import your custom environment and utilities
from base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG
from utils import load_config, merge_configs, get_relevant_config_for_hash, generate_config_hash
from utils import load_tick_data_for_range, load_kline_data_for_range, convert_to_native_types
from custom_wrappers import FlattenAction

# --- Default Fallback Configurations ---
# These are used if config.yaml is missing or specific keys are not defined there.
# They are merged with config.yaml contents.

DEFAULT_RUN_SETTINGS = {
    "log_level": "normal", # "none", "normal", "detailed"
    "log_dir_base": "./logs/ppo_trading/",
    "model_name": "tick_trading_agent",
    "total_timesteps": 1000000,
    "eval_freq_episodes": 10, # How often to evaluate (in episodes)
    "n_evaluation_episodes": 5, # How many episodes to run during each evaluation
}

# The default environment config is loaded from base_env.py
# DEFAULT_ENV_CONFIG = DEFAULT_ENV_CONFIG # Already imported

DEFAULT_PPO_PARAMS = {
    "policy": "MlpPolicy",
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
    "use_sde": True,
    "sde_sample_freq": 4,
    "verbose": 1, # SB3 verbosity level (0: no output, 1: info, 2: debug)
    "total_timesteps": 1000000, # This will be popped and passed to model.learn()
    "policy_kwargs": "dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))" # Default network architecture
}

DEFAULT_BINANCE_SETTINGS = {
    "default_symbol": "BTCUSDT",
    "historical_interval": "1h",
    "historical_cache_dir": "./binance_data_cache/",
    "start_date_kline_data": "2024-01-01",
    "end_date_kline_data": "2024-03-31",
    "start_date_tick_data": "2024-01-01",
    "end_date_tick_data": "2024-03-31",
    "api_key": None, # Should be read from env var or config
    "api_secret": None, # Should be read from env var or config
    "testnet": True,
    "api_request_delay_seconds": 0.05,
}

DEFAULT_EVALUATION_DATA_SETTINGS = {
    "start_date_eval": "2024-04-01",
    "end_date_eval": "2024-04-07",
}

# Define which config keys contribute to the run hash (for reproducibility)
# This list must be comprehensive for any parameter that, if changed, should result in a new model.
DEFAULT_HASH_CONFIG_KEYS = {
    "environment": [
        "kline_window_size", "kline_price_features", "tick_feature_window_size",
        "tick_features_to_use", "initial_balance", "commission_pct",
        "base_trade_amount_ratio", "min_tradeable_unit", "catastrophic_loss_threshold_pct",
        "obs_clip_low", "obs_clip_high", "min_profit_target_low", "min_profit_target_high",
        "reward_open_buy_position", "penalty_buy_insufficient_balance", "penalty_buy_position_already_open",
        "reward_sell_profit_base", "reward_sell_profit_factor", "penalty_sell_loss_factor",
        "penalty_sell_loss_base", "penalty_sell_no_position", "reward_hold_profitable_position",
        "penalty_hold_losing_position", "penalty_hold_flat_position", "penalty_catastrophic_loss",
        "reward_eof_sell_factor", "reward_sell_meets_target_bonus", "penalty_sell_profit_below_target"
    ],
    "ppo_params": [
        "policy", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
        "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
        "use_sde", "sde_sample_freq", "policy_kwargs"
    ],
    "binance_settings": [
        "default_symbol", "historical_interval", # Data source details for the agent's observation space
        # NOTE: Date ranges are typically NOT included in the hash unless changing them fundamentally
        # changes the experiment (e.g., training on bull vs. bear market).
        # Including them would create a new hash for every new daily batch of data, which is usually not desired.
    ]
}

# --- New Function: `train_agent` ---
def train_agent(
    config_override: dict = None,
    log_to_file: bool = True,
    metric_to_return: str = "eval_reward" # Can be "eval_reward", "final_equity_pct" etc. for Optuna
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

    # 1. Load and Merge Configurations
    try:
        loaded_config = load_config("config.yaml")
        # Start with a full merge of all default sections
        config = {
            "run_settings": merge_configs(DEFAULT_RUN_SETTINGS, loaded_config.get("run_settings")),
            "environment": merge_configs(DEFAULT_ENV_CONFIG, loaded_config.get("environment")),
            "ppo_params": merge_configs(DEFAULT_PPO_PARAMS, loaded_config.get("ppo_params")),
            "binance_settings": merge_configs(DEFAULT_BINANCE_SETTINGS, loaded_config.get("binance_settings")),
            "evaluation_data": merge_configs(DEFAULT_EVALUATION_DATA_SETTINGS, loaded_config.get("evaluation_data")),
            "hash_config_keys": merge_configs(DEFAULT_HASH_CONFIG_KEYS, loaded_config.get("hash_config_keys")),
        }
        if config_override:
            # Apply config_override to the already merged config structure
            config = merge_configs(config, config_override)
    except Exception as e:
        print(f"Error loading or merging configuration: {e}. Using default training config.")
        # Fallback to all defaults if loading fails
        config = {
            "run_settings": DEFAULT_RUN_SETTINGS.copy(),
            "environment": DEFAULT_ENV_CONFIG.copy(),
            "ppo_params": DEFAULT_PPO_PARAMS.copy(),
            "binance_settings": DEFAULT_BINANCE_SETTINGS.copy(),
            "evaluation_data": DEFAULT_EVALUATION_DATA_SETTINGS.copy(),
            "hash_config_keys": DEFAULT_HASH_CONFIG_KEYS.copy(),
        }
        if config_override: # Apply overrides even if default is used
            config = merge_configs(config, config_override)


    run_settings = config["run_settings"]
    env_config = config["environment"]
    ppo_params = config["ppo_params"]
    binance_settings = config["binance_settings"]
    evaluation_data_config = config["evaluation_data"] # Get eval data settings for EvalCallback

    # Adjust log level based on log_to_file
    current_log_level = run_settings.get("log_level", "normal")
    if not log_to_file: # Suppress excessive logging during optimization trials
        current_log_level = "none"
        env_config["custom_print_render"] = "none" # Ensure no rendering

    # 2. Generate a unique run ID based on hashed configuration and Setup Logging Directory
    run_id = "optuna_trial" # Default for Optuna trials
    log_dir = "./optuna_temp_logs" # Default temp log dir for Optuna if not logging to file
    
    if log_to_file:
        relevant_config_for_hash = get_relevant_config_for_hash(
            config, # Pass the fully merged config as the 'full_yaml_config'
            config, # Pass the fully merged config as the 'train_script_fallback_config'
            env_config # Pass the final env config as 'env_script_fallback_config'
        )
        config_hash = generate_config_hash(relevant_config_for_hash)
        run_id = f"{config_hash}_{run_settings['model_name']}"
        log_dir = os.path.join(run_settings['log_dir_base'], run_id)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "best_model"), exist_ok=True) # For EvalCallback

    # Override tensorboard_log if not logging to file
    tensorboard_log_dir = run_settings.get("log_dir_base") if log_to_file else None # TensorBoard base directory
    tb_log_name = run_id if log_to_file else None # Specific run name in TensorBoard

    print(f"Training run ID: {run_id} (Log Level: {current_log_level})")
    if log_to_file:
        print(f"Logs and models will be saved to: {log_dir}")
        # Save the effective configuration for this run
        with open(os.path.join(log_dir, "effective_config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        with open(os.path.join(log_dir, "relevant_config_for_hash.json"), 'w') as f:
            json.dump(convert_to_native_types(relevant_config_for_hash), f, indent=2)
        if current_log_level != "none":
            print("Effective configuration saved to effective_config.yaml")
            if current_log_level == "detailed":
                print("\n--- Effective Configuration for this run (detailed) ---")
                print(json.dumps(convert_to_native_types(config), indent=2, sort_keys=True))
                print("-------------------------------------------\n")
    else:
        print(f"Running Optuna trial. Logs are suppressed.")


    # 3. Data Fetching for Training
    train_start_date_kline = binance_settings["start_date_kline_data"]
    train_end_date_kline = binance_settings["end_date_kline_data"]
    train_start_date_tick = binance_settings["start_date_tick_data"]
    train_end_date_tick = binance_settings["end_date_tick_data"]
    symbol = binance_settings["default_symbol"]
    interval = binance_settings["historical_interval"]
    kline_features = env_config["kline_price_features"]
    tick_features = env_config["tick_features_to_use"]
    cache_dir = binance_settings["historical_cache_dir"]

    if current_log_level != "none": print(f"\n--- Fetching and preparing K-line training data ({symbol}, {interval}, {train_start_date_kline} to {train_end_date_kline}) ---")
    kline_df_train = pd.DataFrame() # Initialize
    try:
        kline_df_train = load_kline_data_for_range(
            symbol=symbol,
            start_date_str=train_start_date_kline,
            end_date_str=train_end_date_kline,
            interval=interval,
            price_features=kline_features,
            cache_dir=cache_dir
        )
        if kline_df_train.empty:
            raise ValueError("K-line training data is empty. Cannot proceed with training.")
        if current_log_level != "none": print(f"K-line training data loaded: {kline_df_train.shape} from {kline_df_train.index.min()} to {kline_df_train.index.max()}")
    except Exception as e:
        print(f"ERROR: K-line training data not loaded. Cannot proceed with training. Details: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials

    if current_log_level != "none": print(f"\n--- Fetching and preparing Tick training data ({symbol}, {train_start_date_tick} to {train_end_date_tick}) ---")
    tick_df_train = pd.DataFrame() # Initialize
    try:
        tick_df_train = load_tick_data_for_range(
            symbol=symbol,
            start_date_str=train_start_date_tick,
            end_date_str=train_end_date_tick,
            cache_dir=cache_dir
        )
        if tick_df_train.empty:
            raise ValueError("Tick training data is empty. Cannot proceed with training.")
        if current_log_level != "none": print(f"Tick training data loaded: {tick_df_train.shape} from {tick_df_train.index.min()} to {tick_df_train.index.max()}")
    except Exception as e:
        print(f"ERROR: Tick training data not loaded. Cannot proceed with training. Details: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials
    
    # Check for overlapping time ranges, if any for context
    train_min_tick_time = tick_df_train.index.min()
    train_max_tick_time = tick_df_train.index.max()
    train_min_kline_time = kline_df_train.index.min()
    train_max_kline_time = kline_df_train.index.max()

    required_kline_start = pd.to_datetime(train_min_tick_time, utc=True) - pd.Timedelta(hours=(env_config["kline_window_size"] + 5) * 2) # Rough buffer
    if train_min_kline_time > required_kline_start:
        if current_log_level != "none": print(f"WARNING: Training K-line data starts at {train_min_kline_time}, which might be too late for the tick data starting at {train_min_tick_time} given kline_window_size. Consider extending start_date_kline_data.")


    # 4. Create Training Environment
    env = None # Initialize outside try for finally block
    eval_env_for_callback = None # Initialize outside try for finally block
    try:
        def create_env_fn(tick_data: pd.DataFrame, kline_data: pd.DataFrame, env_config: dict, monitor_filepath: str = None):
            def _init_env():
                base_env = SimpleTradingEnv(tick_df=tick_data.copy(), kline_df_with_ta=kline_data.copy(), config=env_config)
                wrapped_env = FlattenAction(base_env)
                monitored_env = Monitor(wrapped_env, filename=monitor_filepath) # Monitor can take None for no file logging
                return monitored_env
            return _init_env

        # Ensure env_config log_level matches run_settings log_level for consistency
        env_config["log_level"] = current_log_level
        env_config["custom_print_render"] = "none"

        # Create vectorized environment for training
        train_monitor_path = os.path.join(log_dir, "monitor.csv") if log_to_file else None
        vec_env = make_vec_env(
            create_env_fn(tick_df_train, kline_df_train, env_config, train_monitor_path),
            n_envs=1, # For simplicity, start with 1 env. Can increase for parallel training.
            seed=0 # For reproducibility
        )

        # Apply VecNormalize for observation standardization. Important: save stats after training!
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.) # norm_reward=False as rewards are not normalized in this env

        if current_log_level != "none": print(f"\nTraining Environment created: Observation Space {env.observation_space.shape}, Action Space {env.action_space.shape}")
        
    except Exception as e:
        print(f"ERROR: Failed to create training environment: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials

    # 5. Data Fetching for Evaluation (for EvalCallback)
    eval_start_date_kline = evaluation_data_config["start_date_eval"]
    eval_end_date_kline = evaluation_data_config["end_date_eval"]
    eval_start_date_tick = evaluation_data_config["start_date_eval"]
    eval_end_date_tick = evaluation_data_config["end_date_eval"]

    if current_log_level != "none": print(f"\n--- Fetching and preparing K-line evaluation data ({symbol}, {interval}, {eval_start_date_kline} to {eval_end_date_kline}) ---")
    kline_df_eval = pd.DataFrame() # Initialize
    try:
        kline_df_eval = load_kline_data_for_range(
            symbol=symbol,
            start_date_str=eval_start_date_kline,
            end_date_str=eval_end_date_kline,
            interval=interval,
            price_features=kline_features,
            cache_dir=cache_dir
        )
        if kline_df_eval.empty:
            if current_log_level != "none": print("WARNING: Evaluation K-line data is empty. EvalCallback might not function correctly.")

    except Exception as e:
        print(f"WARNING: K-line evaluation data not loaded. Details: {e}. EvalCallback might be impacted.")
        traceback.print_exc()

    if current_log_level != "none": print(f"\n--- Fetching and preparing Tick evaluation data ({symbol}, {eval_start_date_tick} to {eval_end_date_tick}) ---")
    tick_df_eval = pd.DataFrame() # Initialize
    try:
        tick_df_eval = load_tick_data_for_range(
            symbol=symbol,
            start_date_str=eval_start_date_tick,
            end_date_str=eval_end_date_tick,
            cache_dir=cache_dir
        )
        if tick_df_eval.empty:
            if current_log_level != "none": print("WARNING: Evaluation tick data is empty. EvalCallback might not function correctly.")
    except Exception as e:
        print(f"WARNING: Tick evaluation data not loaded. Details: {e}. EvalCallback might be impacted.")
        traceback.print_exc()

    # Create Evaluation Environment for EvalCallback
    eval_callback = None
    if not kline_df_eval.empty and not tick_df_eval.empty:
        try:
            # Eval environment should use 'none' log level for cleaner output
            eval_env_config = env_config.copy()
            eval_env_config['log_level'] = "none"
            eval_env_config['custom_print_render'] = "none"

            eval_monitor_path = os.path.join(log_dir, "eval_monitor.csv") if log_to_file else None
            eval_vec_env = make_vec_env(
                create_env_fn(tick_df_eval, kline_df_eval, eval_env_config, eval_monitor_path),
                n_envs=1, # EvalCallback expects a VecEnv
                seed=0
            )
            # Apply VecNormalize to the evaluation environment, linking it to the training env's stats
            eval_env_for_callback = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)
            # Initially, eval_env_for_callback has its own normalization, but it will be updated by EvalCallback
            # with the training env's stats. (EvalCallback automatically handles this internally for its eval_env)

            stop_train_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=run_settings.get("eval_freq_episodes", 10) * 2,
                min_evals=run_settings.get("eval_freq_episodes", 10) * 2,
                verbose=1 if current_log_level in ["normal", "detailed"] else 0
            )
            
            # eval_freq: The number of steps between evaluations.
            # In SimpleTradingEnv, each step is a tick. So, we multiply by the max steps per episode (end_step)
            # to get evaluation frequency in terms of full episodes.
            eval_freq_steps = run_settings.get("eval_freq_episodes") * (tick_df_train.shape[0] - env_config["tick_feature_window_size"])
            
            eval_callback = EvalCallback(
                eval_env_for_callback, # Pass the VecNormalize wrapped eval env
                best_model_save_path=os.path.join(log_dir, "best_model") if log_to_file else None,
                log_path=log_dir if log_to_file else None,
                eval_freq=max(1, eval_freq_steps), # Ensure at least 1 step
                n_eval_episodes=run_settings.get("n_evaluation_episodes"),
                deterministic=True,
                render=False,
                callback_after_eval=stop_train_callback,
                verbose=1 if current_log_level in ["normal", "detailed"] else 0
            )
            if current_log_level != "none": print(f"\nEvalCallback set up to run every {run_settings.get('eval_freq_episodes')} episodes (~{eval_freq_steps} steps).")
        except Exception as e:
            print(f"WARNING: Failed to set up EvalCallback: {e}. Training will proceed without it.")
            traceback.print_exc()
            eval_callback = None
            if eval_env_for_callback: eval_env_for_callback.close() # Close if creation failed
            eval_env_for_callback = None # Ensure it's None so finally block doesn't try to close again
    else:
        print("Not enough valid data for evaluation environment. EvalCallback will be skipped.")

    # 6. Create PPO Agent
    # Extract total_timesteps before passing ppo_params to PPO constructor
    # Ensure it's a copy so the original ppo_params (from config) is not modified for hashing
    ppo_params_for_init = ppo_params.copy()
    total_timesteps_for_learn = ppo_params_for_init.pop("total_timesteps")
    
    # Handle policy_kwargs if it's a string (e.g., from YAML)
    if "policy_kwargs" in ppo_params_for_init and isinstance(ppo_params_for_init["policy_kwargs"], str):
        try:
            ppo_params_for_init["policy_kwargs"] = eval(ppo_params_for_init["policy_kwargs"])
        except Exception as e:
            if current_log_level != "none": print(f"WARNING: Could not parse policy_kwargs string '{ppo_params_for_init['policy_kwargs']}': {e}. Using default.")
            if "policy_kwargs" in ppo_params_for_init: del ppo_params_for_init["policy_kwargs"] # Remove potentially invalid policy_kwargs
            # Fallback to a safe default if needed, or let SB3 use its default

    try:
        model = PPO(
            "MlpPolicy", # policy_type
            env, # vectorized environment
            verbose=ppo_params_for_init.pop("verbose", 1), # Pop verbose to avoid passing twice
            tensorboard_log=tensorboard_log_dir,
            **ppo_params_for_init # Unpack remaining PPO parameters
        )
        if current_log_level != "none": print(f"\nPPO Agent created with parameters:\n{json.dumps(convert_to_native_types(ppo_params_for_init), indent=2)}")
        if current_log_level != "none": print(f"Total timesteps to learn: {total_timesteps_for_learn}")

    except Exception as e:
        print(f"ERROR: Failed to create PPO agent: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials

    # 7. Train the Agent
    if current_log_level != "none": print("\n--- Starting PPO Training ---")
    final_return_metric = -np.inf # Initialize with a low value for failed trials
    
    callbacks_list = []
    if log_to_file:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, total_timesteps_for_learn // 5), # Save 5 checkpoints
            save_path=log_dir,
            name_prefix="rl_model"
        )
        callbacks_list.append(checkpoint_callback)
    if eval_callback:
        callbacks_list.append(eval_callback)

    try:
        model.learn(
            total_timesteps=total_timesteps_for_learn,
            callback=callbacks_list,
            progress_bar=True,
            tb_log_name=tb_log_name # Name of the run in TensorBoard
        )
        if current_log_level != "none": print("\nTraining completed.")
        
        # Save VecNormalize statistics
        if log_to_file:
            vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
            env.save(vec_normalize_path) # Save VecNormalize object
            if current_log_level != "none": print(f"VecNormalize statistics saved to: {vec_normalize_path}")

        if log_to_file:
            final_model_path = os.path.join(log_dir, "trained_model_final.zip")
            model.save(final_model_path)
            if current_log_level != "none": print(f"Final model saved to {final_model_path}")

        # Return the best evaluation reward found by EvalCallback for Optuna
        if eval_callback and eval_callback.best_mean_reward is not None:
            final_return_metric = float(eval_callback.best_mean_reward)
            if current_log_level != "none": print(f"Best mean reward from EvalCallback: {final_return_metric:.2f}")
        else:
            if current_log_level != "none": print("WARNING: EvalCallback did not record a best mean reward. Returning -inf.")

    except KeyboardInterrupt:
        if current_log_level != "none": print("\nTraining interrupted by user.")
        # Attempt to save model even if interrupted
        if log_to_file and model:
            interrupted_model_path = os.path.join(log_dir, "trained_model_interrupted.zip")
            model.save(interrupted_model_path)
            if current_log_level != "none": print(f"Model saved to {interrupted_model_path} after interruption.")
    except Exception as e:
        print(f"ERROR: An error occurred during training: {e}")
        traceback.print_exc()
        final_return_metric = -np.inf
    finally:
        # Ensure environments are closed properly
        if 'env' in locals() and env is not None:
            env.close()
            if current_log_level != "none": print("Training environment closed.")
        if 'eval_env_for_callback' in locals() and eval_env_for_callback is not None:
            eval_env_for_callback.close()
            if current_log_level != "none": print("Evaluation environment for callback closed.")
    
    print(f"\n--- Training Script Execution Completed (Metric: {final_return_metric:.2f}) ---")
    return final_return_metric

# Ensure that the main execution block is guarded so Optuna can import `train_agent` directly.
if __name__ == "__main__":
    # When running train_simple_agent.py directly, log to file
    final_reward = train_agent(log_to_file=True)
    print(f"\nFinal Training Metric: {final_reward:.2f}")