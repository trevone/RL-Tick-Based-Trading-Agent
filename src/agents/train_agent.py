# src/agents/train_agent.py

import os
import json
import yaml
import warnings
from datetime import datetime
import traceback
import time # For debug prints around model.learn()

import pandas as pd
import numpy as np

# Suppress pandas FutureWarnings related to 'pd.concat'
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Import Stable Baselines3 algorithms dynamically
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize 

try:
    from sb3_contrib import RecurrentPPO # This name will be patched in tests
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    # This print might still appear if sb3_contrib is not installed,
    # but SB3_CONTRIB_AVAILABLE will be patched to True in mock_sb3_models
    print("WARNING: sb3_contrib (for RecurrentPPO) not found. RecurrentPPO will not be available.")


from src.environments.base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG 
from src.environments.custom_wrappers import FlattenAction
from src.data.utils import load_config, merge_configs, get_relevant_config_for_hash, generate_config_hash
# These two are imported here and will be patched by the test's mock_data_loader
from src.data.utils import load_tick_data_for_range, load_kline_data_for_range 
from src.data.utils import convert_to_native_types, DATA_CACHE_DIR


def load_default_configs_for_training(config_dir="configs/defaults") -> dict:
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "environment.yaml"), 
        os.path.join(config_dir, "ppo_params.yaml"),
        os.path.join(config_dir, "sac_params.yaml"),
        os.path.join(config_dir, "ddpg_params.yaml"),
        os.path.join(config_dir, "a2c_params.yaml"),
        os.path.join(config_dir, "recurrent_ppo_params.yaml"),
        os.path.join(config_dir, "binance_settings.yaml"),
        os.path.join(config_dir, "evaluation_data.yaml"),
        os.path.join(config_dir, "hash_keys.yaml"),
    ]
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)


def train_agent(
    config_override: dict = None,
    log_to_file: bool = True,
    metric_to_return: str = "eval_reward" 
) -> float:
    print("--- Starting Trading Agent Training Process ---") 

    try:
        effective_config = load_default_configs_for_training()
        if config_override:
            effective_config = merge_configs(effective_config, config_override)
    except Exception as e:
        print(f"Error loading or merging configuration: {e}. Using minimal fallback.")
        traceback.print_exc()
        effective_config = { 
            "run_settings": {"log_level": "normal", "log_dir_base": "./logs/training/", "model_name": "fallback_agent"},
            "environment": DEFAULT_ENV_CONFIG.copy(),
            "ppo_params": {"total_timesteps": 100000, "policy_kwargs": "{'net_arch': [64, 64]}"},
            "binance_settings": {"default_symbol": "BTCUSDT", "historical_interval": "1h", "historical_cache_dir": DATA_CACHE_DIR,
                                 "start_date_kline_data": "2024-01-01 00:00:00", "end_date_kline_data": "2024-01-01 23:59:59",
                                 "start_date_tick_data": "2024-01-01 00:00:00", "end_date_tick_data": "2024-01-01 23:59:59",
                                 "testnet": True, "api_request_delay_seconds": 0.2},
            "evaluation_data": {"start_date_eval": "2024-01-02 00:00:00", "end_date_eval": "2024-01-02 23:59:59"},
            "hash_config_keys": {"environment": [], "agent_params": {"PPO": []}, "binance_settings": []}
        }
        if config_override: 
            effective_config = merge_configs(effective_config, config_override)

    run_settings = effective_config["run_settings"]
    env_config = effective_config["environment"]
    binance_settings = effective_config["binance_settings"]
    evaluation_data_config = effective_config["evaluation_data"]
    agent_type = effective_config.get("agent_type", "PPO")
    algo_params = effective_config.get(f"{agent_type.lower()}_params", {})

    current_log_level = run_settings.get("log_level", "normal")
    if not log_to_file: 
        current_log_level = "none" # This will be set by the test
        env_config["custom_print_render"] = "none"

    run_id = "optuna_trial" 
    log_dir_base = "./logs/training/" 
    log_dir = "./optuna_temp_logs" 
    
    if log_to_file:
        relevant_config_for_hash = get_relevant_config_for_hash(effective_config)
        config_hash = generate_config_hash(relevant_config_for_hash)
        run_id = f"{config_hash}_{run_settings['model_name']}"
        log_dir = os.path.join(log_dir_base, run_id)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "best_model"), exist_ok=True)

    tensorboard_log_dir = os.path.join("logs", "tensorboard_logs") if log_to_file else None
    tb_log_name = run_id if log_to_file else None

    print(f"Training run ID: {run_id} (Log Level: {current_log_level})") # Will be 'none' for test
    print(f"Agent Type: {agent_type}")
    # ... (config saving logic if log_to_file - won't run for test) ...

    train_start_date_kline = binance_settings["start_date_kline_data"]
    train_end_date_kline = binance_settings["end_date_kline_data"]
    train_start_date_tick = binance_settings["start_date_tick_data"]
    train_end_date_tick = binance_settings["end_date_tick_data"]
    symbol = binance_settings["default_symbol"]
    interval = binance_settings["historical_interval"]
    kline_features = env_config["kline_price_features"]
    cache_dir = binance_settings["historical_cache_dir"]
    tick_resample_interval_ms = env_config.get("tick_resample_interval_ms")

    # This print will be suppressed because current_log_level is 'none' in the test
    if current_log_level != "none": print(f"\n--- Preparing K-line training data ({symbol}, {interval}, {train_start_date_kline} to {train_end_date_kline}) ---")
    kline_df_train = pd.DataFrame() 
    try:
        kline_df_train = load_kline_data_for_range( # This call will be mocked
            symbol=symbol,
            start_date_str=train_start_date_kline,
            end_date_str=train_end_date_kline,
            interval=interval,
            price_features=kline_features,
            cache_dir=cache_dir,
            binance_settings=binance_settings 
        )
        # +++ START DEBUG BLOCK for kline_df_train +++
        print(f"\nDEBUG TRAIN_AGENT: kline_df_train - Actual shape: {kline_df_train.shape}")
        if not kline_df_train.empty:
            print(f"DEBUG TRAIN_AGENT: kline_df_train - Actual head:\n{kline_df_train.head()}\n")
        else:
            print("DEBUG TRAIN_AGENT: kline_df_train is EMPTY.\n")
        # +++ END DEBUG BLOCK +++
        if kline_df_train.empty:
            raise ValueError("K-line training data is empty.")
        # This print will be suppressed
        if current_log_level != "none": print(f"K-line training data loaded: {kline_df_train.shape}")
    except Exception as e:
        print(f"ERROR: K-line training data not loaded. Details: {e}")
        traceback.print_exc()
        return -np.inf

    # This print will be suppressed
    if current_log_level != "none": print(f"\n--- Preparing Tick training data ({symbol}, {train_start_date_tick} to {train_end_date_tick}) ---")
    tick_df_train = pd.DataFrame() 
    try:
        tick_df_train = load_tick_data_for_range( # This call will be mocked
            symbol=symbol,
            start_date_str=train_start_date_tick,
            end_date_str=train_end_date_tick,
            cache_dir=cache_dir,
            binance_settings=binance_settings, 
            tick_resample_interval_ms=tick_resample_interval_ms 
        )
        # +++ START DEBUG BLOCK for tick_df_train +++
        print(f"\nDEBUG TRAIN_AGENT: tick_df_train - Actual shape: {tick_df_train.shape}")
        if not tick_df_train.empty:
            print(f"DEBUG TRAIN_AGENT: tick_df_train - Actual head:\n{tick_df_train.head()}\n")
        else:
            print("DEBUG TRAIN_AGENT: tick_df_train is EMPTY.\n")
        # +++ END DEBUG BLOCK +++
        if tick_df_train.empty:
            raise ValueError("Tick training data is empty.")
        # This print will be suppressed
        if current_log_level != "none": print(f"Tick training data loaded: {tick_df_train.shape}")
    except Exception as e:
        print(f"ERROR: Tick training data not loaded. Details: {e}")
        traceback.print_exc()
        return -np.inf
    
    # ... (kline data time range check - might print if current_log_level != "none") ...

    env = None 
    eval_env_for_callback = None 
    try:
        def create_env_fn(tick_data: pd.DataFrame, kline_data: pd.DataFrame, env_config_dict: dict, monitor_filepath: str = None):
            def _init_env():
                # Env's internal prints will be off due to env_config_dict['log_level'] = "none"
                base_env = SimpleTradingEnv(tick_df=tick_data.copy(), kline_df_with_ta=kline_data.copy(), config=env_config_dict)
                wrapped_env = FlattenAction(base_env) # This prints "Original action space..."
                monitored_env = Monitor(wrapped_env, filename=monitor_filepath) 
                return monitored_env
            return _init_env

        current_env_config = env_config.copy() 
        current_env_config["log_level"] = current_log_level # For test, this is "none"
        current_env_config["custom_print_render"] = "none"

        train_monitor_path = os.path.join(log_dir, "monitor.csv") if log_to_file else None
        vec_env = make_vec_env(
            create_env_fn(tick_df_train, kline_df_train, current_env_config, train_monitor_path),
            n_envs=1, 
            seed=0 
        )
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.) 
        # This print will be suppressed
        if current_log_level != "none": print(f"\nTraining Environment created: Obs Space {env.observation_space.shape}, Act Space {env.action_space.shape}")
    except Exception as e:
        print(f"ERROR: Failed to create training environment: {e}")
        traceback.print_exc()
        return -np.inf

    # Data Fetching for Evaluation
    eval_start_date_kline = evaluation_data_config["start_date_kline_eval"]
    eval_end_date_kline = evaluation_data_config["end_date_kline_eval"]
    eval_start_date_tick = evaluation_data_config.get("start_date_tick_eval", evaluation_data_config["start_date_eval"])
    eval_end_date_tick = evaluation_data_config.get("end_date_tick_eval", evaluation_data_config["end_date_eval"])

    kline_df_eval = pd.DataFrame()
    tick_df_eval = pd.DataFrame()

    # This print will be suppressed
    if current_log_level != "none": print(f"\n--- Preparing K-line evaluation data ---")
    try:
        kline_df_eval = load_kline_data_for_range( # Mocked call
            symbol=symbol, start_date_str=eval_start_date_kline, end_date_str=eval_end_date_kline,
            interval=interval, price_features=kline_features, cache_dir=cache_dir, binance_settings=binance_settings
        )
        print(f"\nDEBUG TRAIN_AGENT: kline_df_eval - Actual shape: {kline_df_eval.shape}") # Debug print
        if not kline_df_eval.empty: print(f"DEBUG TRAIN_AGENT: kline_df_eval - Actual head:\n{kline_df_eval.head()}\n")
        else: print("DEBUG TRAIN_AGENT: kline_df_eval is EMPTY.\n")
        if kline_df_eval.empty and current_log_level != "none": print("WARNING: Eval K-line data empty.")
    except Exception as e:
        print(f"WARNING: Eval K-line data not loaded: {e}.")

    # This print will be suppressed
    if current_log_level != "none": print(f"\n--- Preparing Tick evaluation data ---")
    try:
        tick_df_eval = load_tick_data_for_range( # Mocked call
            symbol=symbol, start_date_str=eval_start_date_tick, end_date_str=eval_end_date_tick,
            cache_dir=cache_dir, binance_settings=binance_settings, tick_resample_interval_ms=tick_resample_interval_ms
        )
        print(f"\nDEBUG TRAIN_AGENT: tick_df_eval - Actual shape: {tick_df_eval.shape}") # Debug print
        if not tick_df_eval.empty: print(f"DEBUG TRAIN_AGENT: tick_df_eval - Actual head:\n{tick_df_eval.head()}\n")
        else: print("DEBUG TRAIN_AGENT: tick_df_eval is EMPTY.\n")
        if tick_df_eval.empty and current_log_level != "none": print("WARNING: Eval Tick data empty.")
    except Exception as e:
        print(f"WARNING: Eval Tick data not loaded: {e}.")

    eval_callback = None
    if not kline_df_eval.empty and not tick_df_eval.empty: # Will be true with current mocks
        try:
            eval_env_config_for_callback = env_config.copy() 
            eval_env_config_for_callback['log_level'] = "none" # For Eval Env
            eval_env_config_for_callback['custom_print_render'] = "none"
            
            eval_monitor_path = os.path.join(log_dir, "eval_monitor.csv") if log_to_file else None
            eval_vec_env = make_vec_env(
                create_env_fn(tick_df_eval, kline_df_eval, eval_env_config_for_callback, eval_monitor_path),
                n_envs=1, seed=0
            )
            eval_env_for_callback = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)
            
            stop_train_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=run_settings.get("eval_freq_episodes", 1) * 2, # Use 1 for faster test
                min_evals=run_settings.get("eval_freq_episodes", 1) * 2,
                verbose=0 # Suppressed for test
            )
            approx_steps_per_episode = max(1, tick_df_train.shape[0] - env_config["tick_feature_window_size"])
            eval_freq_steps = int(run_settings.get("eval_freq_episodes",1) * approx_steps_per_episode)
            
            eval_callback = EvalCallback(
                eval_env_for_callback, 
                best_model_save_path=os.path.join(log_dir, "best_model") if log_to_file else None,
                log_path=log_dir if log_to_file else None,
                eval_freq=max(1, eval_freq_steps), 
                n_eval_episodes=run_settings.get("n_evaluation_episodes",1),
                deterministic=True, render=False, callback_after_eval=stop_train_callback,
                verbose=0 # Suppressed for test
            )
            # This print will be suppressed
            if current_log_level != "none": print(f"\nEvalCallback set up (eval_freq_steps: {eval_freq_steps}).")
        except Exception as e:
            print(f"WARNING: Failed to set up EvalCallback: {e}.")
            eval_callback = None
            if 'eval_env_for_callback' in locals() and eval_env_for_callback: eval_env_for_callback.close() 
            eval_env_for_callback = None 
    else:
        # This print will be suppressed
        if current_log_level != "none": print("EvalCallback skipped: Not enough eval data.")

    model = None
    algo_params_for_init = algo_params.copy()
    total_timesteps_for_learn = algo_params_for_init.pop("total_timesteps", 100000) 
    
    if "policy_kwargs" in algo_params_for_init and isinstance(algo_params_for_init["policy_kwargs"], str):
        try:
            algo_params_for_init["policy_kwargs"] = eval(algo_params_for_init["policy_kwargs"])
        except Exception as e:
            # This print will be suppressed
            if current_log_level != "none": print(f"WARNING: Could not parse policy_kwargs: {e}.")
            if "policy_kwargs" in algo_params_for_init: del algo_params_for_init["policy_kwargs"]

    try:
        model_class_map = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "A2C": A2C}
        model_class = None # Define model_class before assignment
        if agent_type == "RecurrentPPO":
            if SB3_CONTRIB_AVAILABLE: model_class = RecurrentPPO
            else: raise ImportError("RecurrentPPO requested but sb3_contrib not found.")
        else:
            model_class = model_class_map.get(agent_type)
        if not model_class: raise ValueError(f"Unknown agent type: {agent_type}.")

        model = model_class( # This is where PPO() etc. is called, will be mocked in test
            "MlpPolicy" if agent_type != "RecurrentPPO" else "MlpLstmPolicy", 
            env, verbose=0, # Suppressed for test
            tensorboard_log=tensorboard_log_dir, **algo_params_for_init
        )
        # This print will be suppressed
        if current_log_level != "none": print(f"\n{agent_type} Agent created. Total timesteps: {total_timesteps_for_learn}")
    except Exception as e:
        print(f"ERROR: Failed to create {agent_type} agent: {e}")
        traceback.print_exc()
        return -np.inf

    # This print will be suppressed
    if current_log_level != "none": print(f"\n--- Starting {agent_type} Training ---")
    final_return_metric = -np.inf # Default for failed runs or no eval
    callbacks_list = []
    if log_to_file and run_settings.get("save_checkpoints", True):
        callbacks_list.append(CheckpointCallback(save_freq=max(1, total_timesteps_for_learn // 5), save_path=log_dir, name_prefix="rl_model"))
    if eval_callback:
        callbacks_list.append(eval_callback)

    try:
        print(f"\nDEBUG TRAIN_AGENT: About to call model.learn() for {total_timesteps_for_learn} timesteps.")
        print(f"DEBUG TRAIN_AGENT: Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
        learn_start_time = time.time()

        model.learn( # This model is a MagicMock in the test
            total_timesteps=total_timesteps_for_learn,
            callback=callbacks_list if callbacks_list else None, 
            progress_bar=True, # Will show for the test, but it's on mock model.learn
            tb_log_name=tb_log_name 
        )
        
        learn_duration = time.time() - learn_start_time
        print(f"\nDEBUG TRAIN_AGENT: model.learn() COMPLETED.")
        print(f"DEBUG TRAIN_AGENT: Duration: {learn_duration:.2f} seconds.")
        print(f"DEBUG TRAIN_AGENT: Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC\n")

        # This print will be suppressed
        if current_log_level != "none": print("\nTraining completed.")
        
        # This block won't run if log_to_file is False (as in test)
        if log_to_file:
            # ... (saving logic) ...
            pass

        # Logic for final_return_metric
        if eval_callback and hasattr(eval_callback, 'best_mean_reward') and eval_callback.best_mean_reward is not None:
            final_return_metric = float(eval_callback.best_mean_reward)
        elif not eval_callback and current_log_level != "none": # Will be suppressed
             print("WARNING: EvalCallback was not used, final_return_metric will remain as initialized.")
        # If eval_callback is None OR best_mean_reward is None, final_return_metric will be what it was.
        # If it was initialized to -np.inf, it will stay -np.inf unless EvalCallback sets it.
        # In the test, an actual EvalCallback runs on mock data. It will produce a small reward.
        # So, final_return_metric will likely be a small float e.g. -0.0001 from penalties.

    except KeyboardInterrupt:
        # ...
        pass
    except Exception as e:
        print(f"ERROR: An error occurred during training: {e}")
        traceback.print_exc()
        final_return_metric = -np.inf 
    finally:
        if 'env' in locals() and env is not None: env.close()
        if 'eval_env_for_callback' in locals() and eval_env_for_callback is not None: eval_env_for_callback.close()
    
    # This print will show the small float value from EvalCallback
    print(f"\n--- Training Script Execution Completed (Metric: {final_return_metric:.4f}) ---") 
    return final_return_metric

if __name__ == "__main__":
    final_reward = train_agent(log_to_file=True)
    print(f"\n--- Training Script Finished (Final Metric: {final_reward:.4f}) ---")