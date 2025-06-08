# src/agents/train_agent.py
import os
import json
import traceback
import pandas as pd
import numpy as np
import warnings

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# CORRECTED: Added all required agent imports
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False

from ..environments.env_loader import load_environments
from ..environments.custom_wrappers import FlattenAction
from ..data.config_loader import merge_configs, get_relevant_config_for_hash, generate_config_hash, convert_to_native_types
from ..data.data_loader import load_tick_data_for_range, load_kline_data_for_range
from .agent_utils import load_default_configs_for_training

def train_agent(config_override: dict = None, log_to_file: bool = True):
    """
    Main function to configure and run the RL agent training process.
    """
    print("--- Starting Trading Agent Training Process ---")
    
    # 1. Load Configuration
    config = load_default_configs_for_training()
    if config_override:
        config = merge_configs(config, config_override)

    run_settings = config["run_settings"]
    env_config = config["environment"]
    agent_type = config.get("agent_type", "PPO")
    algo_params = config.get(f"{agent_type.lower()}_params", {})

    # 2. Setup Logging and Paths
    run_id = f"{generate_config_hash(get_relevant_config_for_hash(config))}_{run_settings.get('model_name', 'agent')}"
    log_dir = os.path.join(run_settings.get("log_dir_base", "logs/"), "training", run_id)
    model_save_dir = os.path.join(log_dir, "best_model")
    os.makedirs(model_save_dir, exist_ok=True)
    with open(os.path.join(log_dir, "effective_train_config.json"), "w") as f:
        json.dump(convert_to_native_types(config), f, indent=4)
    print(f"Training run ID: {run_id}")
    print(f"Logs will be saved to: {log_dir}")

    # 3. Load Data
    print("\n--- Preparing Data ---")
    kline_df_train = load_kline_data_for_range(run_settings["default_symbol"], run_settings["start_date_train"], run_settings["end_date_train"], run_settings["historical_interval"], env_config["kline_price_features"], run_settings["historical_cache_dir"], config["binance_settings"])
    tick_df_train = load_tick_data_for_range(run_settings["default_symbol"], run_settings["start_date_train"], run_settings["end_date_train"], run_settings["historical_cache_dir"], config["binance_settings"], env_config.get("tick_resample_interval_ms"))
    if tick_df_train.empty: raise ValueError("Training tick data could not be loaded. Aborting.")
    print("Training data loaded successfully.")

    # 4. Create Training Environment
    print("\n--- Creating Environments ---")
    available_envs = load_environments()
    def create_env(tick_data, kline_data, env_config_dict, monitor_filepath=None):
        env_type = env_config_dict.get("env_type", "simple")
        env_class = available_envs[env_type]
        base_env = env_class(tick_df=tick_data.copy(), kline_df_with_ta=kline_data.copy(), config=env_config_dict)
        return Monitor(FlattenAction(base_env), filename=monitor_filepath, allow_early_resets=True)

    train_vec_env = make_vec_env(lambda: create_env(tick_df_train, kline_df_train, env_config, os.path.join(log_dir, "train_monitor.csv")), n_envs=1)
    
    vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
    if run_settings.get("continue_from_existing_model") and os.path.exists(vec_normalize_path):
        print(f"Loading existing VecNormalize stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, train_vec_env)
        env.training = True
    else:
        print("Initializing new VecNormalize stats.")
        env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)

    # 5. Setup Evaluation Environment & Callback
    eval_callback = None
    kline_df_eval = load_kline_data_for_range(run_settings["default_symbol"], run_settings["start_date_eval"], run_settings["end_date_eval"], run_settings["historical_interval"], env_config["kline_price_features"], run_settings["historical_cache_dir"], config["binance_settings"])
    tick_df_eval = load_tick_data_for_range(run_settings["default_symbol"], run_settings["start_date_eval"], run_settings["end_date_eval"], run_settings["historical_cache_dir"], config["binance_settings"], env_config.get("tick_resample_interval_ms"))
    
    if not tick_df_eval.empty:
        eval_vec_env = make_vec_env(lambda: create_env(tick_df_eval, kline_df_eval, env_config, os.path.join(log_dir, "eval_monitor.csv")), n_envs=1)
        eval_env = VecNormalize(eval_vec_env, training=False, norm_obs=True, norm_reward=False)
        eval_env.obs_rms = env.obs_rms
        
        eval_freq = int(run_settings.get("eval_freq_episodes", 10) * len(tick_df_train))
        eval_callback = EvalCallback(eval_env, best_model_save_path=model_save_dir, log_path=log_dir, eval_freq=max(1, eval_freq), n_eval_episodes=run_settings.get("n_evaluation_episodes", 5), deterministic=True)

    # 6. Create or Load Agent
    model = None
    model_loaded = False
    device = run_settings.get("device", "cpu")
    model_class = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "A2C": A2C, "RecurrentPPO": RecurrentPPO if SB3_CONTRIB_AVAILABLE else None}.get(agent_type)
    if not model_class: raise ValueError(f"Agent type {agent_type} not supported.")

    if run_settings.get("continue_from_existing_model"):
        potential_path = os.path.join(model_save_dir, "best_model.zip")
        if os.path.exists(potential_path):
            print(f"\n--- Loading existing best model from: {potential_path} ---")
            model = model_class.load(potential_path, env=env, device=device)
            model_loaded = True

    if not model:
        print("\n--- Initializing new model ---")
        policy = "MlpLstmPolicy" if agent_type == "RecurrentPPO" else "MlpPolicy"
        
        params_for_init = algo_params.copy()
        params_for_init.pop("total_timesteps", None)
        
        if isinstance(params_for_init.get("policy_kwargs"), str):
            params_for_init["policy_kwargs"] = eval(params_for_init["policy_kwargs"])
            
        model = model_class(policy, env, verbose=0, tensorboard_log=os.path.join("logs", "tensorboard_logs"), device=device, **params_for_init)

    # 7. Train the Agent
    print(f"\n--- Starting Training ---")
    try:
        model.learn(
            total_timesteps=int(algo_params.get("total_timesteps", 1000000)),
            callback=eval_callback,
            reset_num_timesteps=not model_loaded,
            progress_bar=True,
            tb_log_name=run_id
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Saving final model and normalization stats...")
        model.save(os.path.join(log_dir, "final_model.zip"))
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        env.close()

if __name__ == "__main__":
    try:
        train_agent()
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")
        traceback.print_exc()