# train_agent.py

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

# Import Stable Baselines3 algorithms dynamically
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # Keep both for flexibility
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize # Crucial for obs normalization

# For RecurrentPPO (if available and chosen)
try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    print("WARNING: sb3_contrib (for RecurrentPPO) not found. RecurrentPPO will not be available.")


# Import your custom environment and utilities from new paths
from src.environments.base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG # Keep DEFAULT_ENV_CONFIG for merging with specific env config
from src.environments.custom_wrappers import FlattenAction
from src.data.utils import load_config, merge_configs, get_relevant_config_for_hash, generate_config_hash
from src.data.utils import load_tick_data_for_range, load_kline_data_for_range, convert_to_native_types, DATA_CACHE_DIR


# --- NEW: Function to load default configurations from files ---
def load_default_configs_for_training(config_dir="configs/defaults") -> dict:
    """Loads default configurations from the specified directory."""
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "environment.yaml"), # This contains DEFAULT_ENV_CONFIG
        os.path.join(config_dir, "ppo_params.yaml"),
        os.path.join(config_dir, "sac_params.yaml"),
        os.path.join(config_dir, "ddpg_params.yaml"),
        os.path.join(config_dir, "a2c_params.yaml"),
        os.path.join(config_dir, "recurrent_ppo_params.yaml"),
        os.path.join(config_dir, "binance_settings.yaml"),
        os.path.join(config_dir, "evaluation_data.yaml"),
        os.path.join(config_dir, "hash_keys.yaml"),
    ]
    
    # Use the new load_config from src.data.utils which merges multiple files
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)


# --- New Function: `train_agent` ---
def train_agent(
    config_override: dict = None,
    log_to_file: bool = True,
    metric_to_return: str = "eval_reward" # Can be "eval_reward", "final_equity_pct" etc. for Optuna
) -> float:
    """
    Trains the RL agent based on the provided configuration.
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
        # Load all default configs and then merge with config.yaml
        effective_config = load_default_configs_for_training()
        
        if config_override:
            # Apply config_override to the already merged effective config
            effective_config = merge_configs(effective_config, config_override)
    except Exception as e:
        print(f"Error loading or merging configuration: {e}. Using minimal fallback.")
        traceback.print_exc()
        # Fallback to minimal if loading fails
        effective_config = {
            "run_settings": {"log_level": "normal", "log_dir_base": "./logs/training/", "model_name": "fallback_agent"},
            "environment": DEFAULT_ENV_CONFIG.copy(),
            "ppo_params": {"total_timesteps": 100000, "policy_kwargs": "{'net_arch': [64, 64]}"}, # Minimal PPO
            "binance_settings": {"default_symbol": "BTCUSDT", "historical_interval": "1h", "historical_cache_dir": DATA_CACHE_DIR,
                                 "start_date_kline_data": "2024-01-01", "end_date_kline_data": "2024-01-03",
                                 "start_date_tick_data": "2024-01-01 00:00:00", "end_date_tick_data": "2024-01-03 00:00:00",
                                 "testnet": True, "api_request_delay_seconds": 0.2},
            "evaluation_data": {"start_date_eval": "2024-01-04", "end_date_eval": "2024-01-04"},
            "hash_config_keys": {"environment": [], "agent_params": {"PPO": []}, "binance_settings": []} # Empty hash for fallback
        }
        if config_override: # Apply overrides even if fallback is used
            effective_config = merge_configs(effective_config, config_override)


    run_settings = effective_config["run_settings"]
    env_config = effective_config["environment"]
    binance_settings = effective_config["binance_settings"]
    evaluation_data_config = effective_config["evaluation_data"] # Get eval data settings for EvalCallback
    
    agent_type = effective_config.get("agent_type", "PPO") # Default to PPO if not specified
    algo_params = effective_config.get(f"{agent_type.lower()}_params", {}) # Get algorithm-specific params

    # Adjust log level based on log_to_file
    current_log_level = run_settings.get("log_level", "normal")
    if not log_to_file: # Suppress excessive logging during optimization trials
        current_log_level = "none"
        env_config["custom_print_render"] = "none" # Ensure no rendering

    # 2. Generate a unique run ID based on hashed configuration and Setup Logging Directory
    run_id = "optuna_trial" # Default for Optuna trials
    log_dir_base = "./logs/training/" # Base for all training logs
    log_dir = "./optuna_temp_logs" # Default temp log dir for Optuna if not logging to file
    
    if log_to_file:
        relevant_config_for_hash = get_relevant_config_for_hash(effective_config)
        config_hash = generate_config_hash(relevant_config_for_hash)
        run_id = f"{config_hash}_{run_settings['model_name']}"
        log_dir = os.path.join(log_dir_base, run_id)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "best_model"), exist_ok=True) # For EvalCallback

    # TensorBoard logs go into logs/tensorboard_logs, with a subfolder for each run
    tensorboard_log_dir = os.path.join("logs", "tensorboard_logs") if log_to_file else None
    tb_log_name = run_id if log_to_file else None # Specific run name in TensorBoard

    print(f"Training run ID: {run_id} (Log Level: {current_log_level})")
    print(f"Agent Type: {agent_type}")
    if log_to_file:
        print(f"Logs and models will be saved to: {log_dir}")
        # Save the effective configuration for this run
        with open(os.path.join(log_dir, "effective_config.yaml"), "w") as f:
            yaml.dump(effective_config, f, default_flow_style=False)
        with open(os.path.join(log_dir, "relevant_config_for_hash.json"), 'w') as f:
            json.dump(convert_to_native_types(relevant_config_for_hash), f, indent=2)
        if current_log_level != "none":
            print("Effective configuration saved to effective_config.yaml")
            if current_log_level == "detailed":
                print("\n--- Effective Configuration for this run (detailed) ---")
                print(json.dumps(convert_to_native_types(effective_config), indent=2, sort_keys=True))
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
            cache_dir=cache_dir,
            binance_settings=binance_settings # Pass binance_settings for API keys/testnet
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
            cache_dir=cache_dir,
            binance_settings=binance_settings # Pass binance_settings for API keys/testnet
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
            cache_dir=cache_dir,
            binance_settings=binance_settings # Pass binance_settings for API keys/testnet
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
            cache_dir=cache_dir,
            binance_settings=binance_settings # Pass binance_settings for API keys/testnet
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
            # Approx steps per episode from tick_df_train:
            approx_steps_per_episode = tick_df_train.shape[0] - env_config["tick_feature_window_size"]
            if approx_steps_per_episode <= 0:
                print("WARNING: Not enough training data for even one episode. EvalCallback might fail.")
                eval_freq_steps = 1 # Fallback to minimal
            else:
                eval_freq_steps = run_settings.get("eval_freq_episodes") * approx_steps_per_episode
            
            eval_callback = EvalCallback(
                eval_env_for_callback, # Pass the VecNormalize wrapped eval env
                best_model_save_path=os.path.join(log_dir, "best_model") if log_to_file else None,
                log_path=log_dir if log_to_file else None,
                eval_freq=max(1, int(eval_freq_steps)), # Ensure at least 1 step and is int
                n_eval_episodes=run_settings.get("n_evaluation_episodes"),
                deterministic=True,
                render=False,
                callback_after_eval=stop_train_callback,
                verbose=1 if current_log_level in ["normal", "detailed"] else 0
            )
            if current_log_level != "none": print(f"\nEvalCallback set up to run every {run_settings.get('eval_freq_episodes')} episodes (~{int(eval_freq_steps)} steps).")
        except Exception as e:
            print(f"WARNING: Failed to set up EvalCallback: {e}. Training will proceed without it.")
            traceback.print_exc()
            eval_callback = None
            if eval_env_for_callback: eval_env_for_callback.close() # Close if creation failed
            eval_env_for_callback = None # Ensure it's None so finally block doesn't try to close again
    else:
        print("Not enough valid data for evaluation environment. EvalCallback will be skipped.")

    # 6. Create Agent (PPO, SAC, DDPG, A2C, RecurrentPPO)
    model = None
    # Extract total_timesteps before passing algo_params to constructor
    algo_params_for_init = algo_params.copy()
    total_timesteps_for_learn = algo_params_for_init.pop("total_timesteps", 100000) # Default if missing
    
    # Handle policy_kwargs if it's a string (e.g., from YAML)
    if "policy_kwargs" in algo_params_for_init and isinstance(algo_params_for_init["policy_kwargs"], str):
        try:
            algo_params_for_init["policy_kwargs"] = eval(algo_params_for_init["policy_kwargs"])
        except Exception as e:
            if current_log_level != "none": print(f"WARNING: Could not parse policy_kwargs string '{algo_params_for_init['policy_kwargs']}': {e}. Using default.")
            if "policy_kwargs" in algo_params_for_init: del algo_params_for_init["policy_kwargs"] # Remove potentially invalid policy_kwargs

    try:
        model_class = None
        if agent_type == "PPO":
            model_class = PPO
        elif agent_type == "SAC":
            model_class = SAC
        elif agent_type == "DDPG":
            model_class = DDPG
        elif agent_type == "A2C":
            model_class = A2C
        elif agent_type == "RecurrentPPO":
            if SB3_CONTRIB_AVAILABLE:
                model_class = RecurrentPPO
                # RecurrentPPO requires LstmPolicy. Ensure it's not overridden by policy_kwargs.
                algo_params_for_init["policy"] = "MlpLstmPolicy" # Hardcode for RecurrentPPO
            else:
                raise ImportError("RecurrentPPO requested but sb3_contrib is not installed.")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}. Supported: PPO, SAC, DDPG, A2C, RecurrentPPO.")

        model = model_class(
            "MlpPolicy" if agent_type != "RecurrentPPO" else "MlpLstmPolicy", # Policy type
            env, # vectorized environment
            verbose=1 if current_log_level in ["normal", "detailed"] else 0,
            tensorboard_log=tensorboard_log_dir,
            **algo_params_for_init # Unpack remaining algorithm parameters
        )
        if current_log_level != "none": print(f"\n{agent_type} Agent created with parameters:\n{json.dumps(convert_to_native_types(algo_params_for_init), indent=2)}")
        if current_log_level != "none": print(f"Total timesteps to learn: {total_timesteps_for_learn}")

    except Exception as e:
        print(f"ERROR: Failed to create {agent_type} agent: {e}")
        traceback.print_exc()
        return -np.inf # Return negative infinity for failed trials

    # 7. Train the Agent
    if current_log_level != "none": print(f"\n--- Starting {agent_type} Training ---")
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
    # When running train_agent.py directly, log to file
    final_reward = train_agent(log_to_file=True)
    print(f"\n--- Training Script Finished (Final Metric: {final_reward:.2f}) ---")