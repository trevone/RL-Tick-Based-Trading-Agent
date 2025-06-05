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
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    print("WARNING: sb3_contrib (for RecurrentPPO) not found. RecurrentPPO will not be available.")


from src.environments.base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG
from src.environments.custom_wrappers import FlattenAction
from src.data.utils import load_config, merge_configs, get_relevant_config_for_hash, generate_config_hash
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
            "run_settings": {"log_level": "normal", "log_dir_base": "./logs/training/", "model_name": "fallback_agent", "eval_freq_episodes": 10, "n_evaluation_episodes": 3},
            "environment": DEFAULT_ENV_CONFIG.copy(),
            "ppo_params": {"total_timesteps": 100000, "policy_kwargs": "{'net_arch': [64, 64]}"},
            "binance_settings": {"default_symbol": "BTCUSDT", "historical_interval": "1h", "historical_cache_dir": DATA_CACHE_DIR,
                                 "start_date_kline_data": "2024-01-01 00:00:00", "end_date_kline_data": "2024-01-01 23:59:59",
                                 "start_date_tick_data": "2024-01-01 00:00:00", "end_date_tick_data": "2024-01-01 23:59:59",
                                 "testnet": True, "api_request_delay_seconds": 0.2},
            "evaluation_data": {"start_date_kline_eval": "2024-01-02 00:00:00", "end_date_kline_eval": "2024-01-02 23:59:59",
                                "start_date_tick_eval": "2024-01-02 00:00:00", "end_date_tick_eval": "2024-01-02 23:59:59"},
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
        current_log_level = "none"

    env_config["log_level"] = current_log_level
    env_config["custom_print_render"] = "none" if current_log_level == "none" else env_config.get("custom_print_render", "none")

    run_id = "optuna_trial"
    log_dir = "./optuna_temp_logs"
    model_save_dir = log_dir

    if log_to_file:
        relevant_config_for_hash = get_relevant_config_for_hash(effective_config)
        config_hash = generate_config_hash(relevant_config_for_hash)
        run_id = f"{config_hash}_{run_settings['model_name']}"
        log_dir_base = run_settings.get("log_dir_base", "logs/")
        
        if "training" not in log_dir_base.lower().replace("\\", "/").split("/"):
            log_dir_base_for_training_runs = os.path.join(log_dir_base, "training")
        else:
            log_dir_base_for_training_runs = log_dir_base
            
        log_dir = os.path.join(log_dir_base_for_training_runs, run_id)
        model_save_dir = os.path.join(log_dir, "best_model")

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

        try:
            with open(os.path.join(log_dir, "effective_train_config.json"), "w") as f:
                json.dump(convert_to_native_types(effective_config), f, indent=4)
        except Exception as e_json:
            print(f"Warning: Could not save effective_train_config.json: {e_json}")

    tensorboard_log_dir = os.path.join("logs", "tensorboard_logs") if log_to_file else None
    tb_log_name = run_id if log_to_file else None

    print(f"Training run ID: {run_id} (Log Level: {current_log_level})")
    print(f"Agent Type: {agent_type}")
    if log_to_file: print(f"Training logs will be saved to: {log_dir}")

    train_start_date_kline = binance_settings["start_date_kline_data"]
    train_end_date_kline = binance_settings["end_date_kline_data"]
    train_start_date_tick = binance_settings["start_date_tick_data"]
    train_end_date_tick = binance_settings["end_date_tick_data"]
    symbol = binance_settings["default_symbol"]
    interval = binance_settings["historical_interval"]
    kline_features = env_config["kline_price_features"]
    cache_dir = binance_settings["historical_cache_dir"]
    tick_resample_interval_ms = env_config.get("tick_resample_interval_ms")

    if current_log_level != "none": print(f"\n--- Preparing K-line training data ({symbol}, {interval}, {train_start_date_kline} to {train_end_date_kline}) ---")
    kline_df_train = pd.DataFrame()
    try:
        kline_df_train = load_kline_data_for_range(
            symbol=symbol,
            start_date_str=train_start_date_kline,
            end_date_str=train_end_date_kline,
            interval=interval,
            price_features=kline_features,
            cache_dir=cache_dir,
            binance_settings=binance_settings,
            log_level=current_log_level
        )
        if current_log_level == "detailed":
            print(f"\nDEBUG TRAIN_AGENT: kline_df_train - Actual shape: {kline_df_train.shape}")
            if not kline_df_train.empty: print(f"DEBUG TRAIN_AGENT: kline_df_train - Actual head:\n{kline_df_train.head()}\n")
            else: print("DEBUG TRAIN_AGENT: kline_df_train is EMPTY.\n")

        if kline_df_train.empty:
            raise ValueError("K-line training data is empty. Cannot proceed.")
        if current_log_level != "none": print(f"K-line training data loaded: {kline_df_train.shape}")
    except Exception as e:
        print(f"ERROR: K-line training data not loaded. Details: {e}")
        traceback.print_exc()
        return -np.inf

    if current_log_level != "none": print(f"\n--- Preparing Tick training data ({symbol}, {train_start_date_tick} to {train_end_date_tick}) ---")
    tick_df_train = pd.DataFrame()
    try:
        tick_df_train = load_tick_data_for_range(
            symbol=symbol,
            start_date_str=train_start_date_tick,
            end_date_str=train_end_date_tick,
            cache_dir=cache_dir,
            binance_settings=binance_settings,
            tick_resample_interval_ms=tick_resample_interval_ms,
            log_level=current_log_level
        )
        if current_log_level == "detailed":
            print(f"\nDEBUG TRAIN_AGENT: tick_df_train - Actual shape: {tick_df_train.shape}")
            if not tick_df_train.empty: print(f"DEBUG TRAIN_AGENT: tick_df_train - Actual head:\n{tick_df_train.head()}\n")
            else: print("DEBUG TRAIN_AGENT: tick_df_train is EMPTY.\n")

        if tick_df_train.empty:
            raise ValueError("Tick training data is empty. Cannot proceed.")
        if current_log_level != "none": print(f"Tick training data loaded: {tick_df_train.shape}")
    except Exception as e:
        print(f"ERROR: Tick training data not loaded. Details: {e}")
        traceback.print_exc()
        return -np.inf

    env = None
    eval_env_for_callback = None
    try:
        def create_env_fn(tick_data: pd.DataFrame, kline_data: pd.DataFrame, env_config_dict: dict, monitor_filepath: str = None):
            def _init_env():
                base_env = SimpleTradingEnv(tick_df=tick_data.copy(), kline_df_with_ta=kline_data.copy(), config=env_config_dict)
                wrapped_env = FlattenAction(base_env)
                monitored_env = Monitor(wrapped_env, filename=monitor_filepath, allow_early_resets=True)
                return monitored_env
            return _init_env

        current_env_config_for_train = env_config.copy()

        train_monitor_path = os.path.join(log_dir, "train_monitor.csv") if log_to_file else None
        vec_env = make_vec_env(
            create_env_fn(tick_df_train, kline_df_train, current_env_config_for_train, train_monitor_path),
            n_envs=1,
            seed=np.random.randint(0, 10000)
        )
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)
        if current_log_level != "none": print(f"\nTraining Environment created: Obs Space {env.observation_space.shape}, Act Space {env.action_space.shape}")
    except Exception as e:
        print(f"ERROR: Failed to create training environment: {e}")
        traceback.print_exc()
        if env: env.close()
        return -np.inf

    eval_callback = None
    if current_log_level != "none":
        eval_start_date_kline = evaluation_data_config["start_date_kline_eval"]
        eval_end_date_kline = evaluation_data_config["end_date_kline_eval"]
        eval_start_date_tick = evaluation_data_config.get("start_date_tick_eval", evaluation_data_config["start_date_eval"])
        eval_end_date_tick = evaluation_data_config.get("end_date_tick_eval", evaluation_data_config["end_date_eval"])

        kline_df_eval = pd.DataFrame()
        tick_df_eval = pd.DataFrame()

        if current_log_level != "none": print(f"\n--- Preparing K-line evaluation data ({symbol}, {interval}, {eval_start_date_kline} to {eval_end_date_kline}) ---")
        try:
            kline_df_eval = load_kline_data_for_range(
                symbol=symbol, start_date_str=eval_start_date_kline, end_date_str=eval_end_date_kline,
                interval=interval, price_features=kline_features, cache_dir=cache_dir,
                binance_settings=binance_settings, log_level=current_log_level
            )
            if current_log_level == "detailed": print(f"DEBUG TRAIN_AGENT: kline_df_eval - Shape: {kline_df_eval.shape}")
            if kline_df_eval.empty and current_log_level != "none": print("WARNING: Eval K-line data empty.")
        except Exception as e:
            if current_log_level != "none": print(f"WARNING: Eval K-line data not loaded: {e}.")

        if current_log_level != "none": print(f"\n--- Preparing Tick evaluation data ({symbol}, {eval_start_date_tick} to {eval_end_date_tick}) ---")
        try:
            tick_df_eval = load_tick_data_for_range(
                symbol=symbol, start_date_str=eval_start_date_tick, end_date_str=eval_end_date_tick,
                cache_dir=cache_dir, binance_settings=binance_settings,
                tick_resample_interval_ms=tick_resample_interval_ms, log_level=current_log_level
            )
            if current_log_level == "detailed": print(f"DEBUG TRAIN_AGENT: tick_df_eval - Shape: {tick_df_eval.shape}")
            if tick_df_eval.empty and current_log_level != "none": print("WARNING: Eval Tick data empty.")
        except Exception as e:
            if current_log_level != "none": print(f"WARNING: Eval Tick data not loaded: {e}.")

        if not kline_df_eval.empty and not tick_df_eval.empty:
            try:
                current_env_config_for_eval = env_config.copy()
                current_env_config_for_eval['log_level'] = "none"
                current_env_config_for_eval['custom_print_render'] = "none"

                eval_monitor_path = os.path.join(log_dir, "eval_monitor.csv") if log_to_file else None
                eval_vec_env = make_vec_env(
                    create_env_fn(tick_df_eval, kline_df_eval, current_env_config_for_eval, eval_monitor_path),
                    n_envs=1, seed=np.random.randint(0,10000)
                )
                eval_env_for_callback = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)
                eval_env_for_callback.training = False
                eval_env_for_callback.norm_reward = False

                stop_train_callback = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=run_settings.get("stop_training_patience_evals", 5),
                    min_evals=run_settings.get("stop_training_min_evals", 10),
                    verbose=1 if current_log_level != "none" else 0
                )
                
                approx_steps_per_train_episode = max(1, len(tick_df_train) - env_config.get("tick_feature_window_size", 50))
                eval_freq_steps = int(run_settings.get("eval_freq_episodes", 10) * approx_steps_per_train_episode)
                eval_freq_steps = max(eval_freq_steps, 1)

                eval_callback = EvalCallback(
                    eval_env_for_callback,
                    best_model_save_path=model_save_dir if log_to_file else None,
                    log_path=log_dir if log_to_file else None,
                    eval_freq=eval_freq_steps,
                    n_eval_episodes=run_settings.get("n_evaluation_episodes", 3),
                    deterministic=run_settings.get("deterministic_eval", True),
                    render=False,
                    callback_on_new_best=None,
                    callback_after_eval=stop_train_callback if run_settings.get("use_stop_training_callback", True) else None,
                    verbose=1 if current_log_level != "none" else 0
                )
                if current_log_level != "none": print(f"\nEvalCallback set up (eval_freq_steps: {eval_freq_steps}, patience: {run_settings.get('stop_training_patience_evals', 5)} evals).")
            except Exception as e_eval_setup:
                if current_log_level != "none": print(f"WARNING: Failed to set up EvalCallback: {e_eval_setup}.")
                traceback.print_exc()
                eval_callback = None
                if eval_env_for_callback: eval_env_for_callback.close()
                eval_env_for_callback = None
        else:
            if current_log_level != "none": print("EvalCallback skipped: Not enough evaluation data (K-line or Tick data is empty).")
    else:
        if run_settings.get("log_level", "normal") != "none":
             print("EvalCallback skipped due to log_level='none' (likely Optuna trial or test).")

    model = None
    algo_params_for_init = algo_params.copy()
    total_timesteps_for_learn = algo_params_for_init.pop("total_timesteps", 100000)

    if "policy_kwargs" in algo_params_for_init and isinstance(algo_params_for_init["policy_kwargs"], str):
        try:
            algo_params_for_init["policy_kwargs"] = eval(algo_params_for_init["policy_kwargs"])
        except Exception as e_eval_pk:
            if current_log_level != "none": print(f"WARNING: Could not parse policy_kwargs string: {e_eval_pk}. Using defaults or none.")
            if "policy_kwargs" in algo_params_for_init: del algo_params_for_init["policy_kwargs"]

    try:
        model_class_map = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "A2C": A2C}
        model_class = None
        if agent_type == "RecurrentPPO":
            if SB3_CONTRIB_AVAILABLE: model_class = RecurrentPPO
            else: raise ImportError("RecurrentPPO requested but sb3_contrib not found.")
        else:
            model_class = model_class_map.get(agent_type)

        if not model_class: raise ValueError(f"Unknown or unsupported agent type: {agent_type}")

        model = model_class(
            "MlpPolicy" if agent_type != "RecurrentPPO" else "MlpLstmPolicy",
            env,
            verbose=1 if current_log_level == "detailed" else 0,
            tensorboard_log=tensorboard_log_dir,
            device=run_settings.get("device", "auto"),
            **algo_params_for_init
        )
        if current_log_level != "none": print(f"\n{agent_type} Agent created. Total timesteps for training: {total_timesteps_for_learn}")
    except Exception as e_model_create:
        print(f"ERROR: Failed to create {agent_type} agent: {e_model_create}")
        traceback.print_exc()
        if env: env.close()
        if eval_env_for_callback: eval_env_for_callback.close()
        return -np.inf

    if current_log_level != "none": print(f"\n--- Starting {agent_type} Training ---")
    final_return_metric = -np.inf
    
    callbacks_list = []
    if eval_callback:
        callbacks_list.append(eval_callback)
    
    if log_to_file and run_settings.get("save_checkpoints", True):
        checkpoint_save_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_save_dir, exist_ok=True)
        checkpoint_save_freq = max(1, total_timesteps_for_learn // run_settings.get("num_checkpoints_to_save", 5))
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=checkpoint_save_dir,
            name_prefix=f"{run_id}_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=1 if current_log_level == "detailed" else 0
        )
        callbacks_list.append(checkpoint_callback)
        if current_log_level != "none": print(f"CheckpointCallback enabled. Saving every ~{checkpoint_save_freq} steps to {checkpoint_save_dir}")

    try:
        if current_log_level == "detailed":
            print(f"\nDEBUG TRAIN_AGENT: About to call model.learn() for {total_timesteps_for_learn} timesteps.")
            print(f"DEBUG TRAIN_AGENT: Callbacks: {callbacks_list}")
            print(f"DEBUG TRAIN_AGENT: Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
        
        learn_start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps_for_learn,
            callback=callbacks_list, # CORRECTED: Always pass the list (empty or not)
            progress_bar=True if current_log_level != "none" else False,
            tb_log_name=tb_log_name,
            reset_num_timesteps=False
        )

        learn_duration = time.time() - learn_start_time
        if current_log_level == "detailed":
            print(f"\nDEBUG TRAIN_AGENT: model.learn() COMPLETED.")
            print(f"DEBUG TRAIN_AGENT: Duration: {learn_duration:.2f} seconds.")
            print(f"DEBUG TRAIN_AGENT: Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC\n")
        if current_log_level != "none": print("\nTraining completed.")

        if log_to_file:
            final_model_save_path = os.path.join(log_dir, "trained_model_final.zip")
            model.save(final_model_save_path)
            env.save(os.path.join(log_dir, "vec_normalize_final.pkl"))
            if current_log_level != "none": print(f"Final model and VecNormalize stats saved to {log_dir}")
        
        if eval_callback and hasattr(eval_callback, 'best_mean_reward') and eval_callback.best_mean_reward is not None:
            final_return_metric = float(eval_callback.best_mean_reward)
            if current_log_level != "none": print(f"Best mean evaluation reward: {final_return_metric:.4f}")
        elif not eval_callback and current_log_level != "none":
             print("WARNING: EvalCallback was not used or did not produce a best_mean_reward. Final metric is -inf.")
        
    except KeyboardInterrupt:
        if current_log_level != "none": print("\nTraining interrupted by user.")
        if log_to_file and model is not None and run_settings.get("save_on_interrupt", True):
            interrupted_model_path = os.path.join(log_dir, "trained_model_interrupted.zip")
            model.save(interrupted_model_path)
            env.save(os.path.join(log_dir, "vec_normalize_interrupted.pkl"))
            if current_log_level != "none": print(f"Interrupted model saved to {interrupted_model_path}")
        final_return_metric = -np.inf
    except Exception as e_learn:
        print(f"ERROR: An error occurred during training: {e_learn}")
        traceback.print_exc()
        final_return_metric = -np.inf
    finally:
        if env: env.close()
        if eval_env_for_callback: eval_env_for_callback.close()

    if current_log_level != "none": print(f"\n--- Training Script Execution Completed (Metric for Optuna/Caller: {final_return_metric:.4f}) ---")
    return final_return_metric


if __name__ == "__main__":
    final_reward_metric = train_agent(log_to_file=True)
    print(f"\n--- Main Training Script Finished (Final Metric from EvalCallback: {final_reward_metric:.4f}) ---")