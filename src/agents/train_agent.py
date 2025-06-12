# src/agents/train_agent.py
import os
import json
import traceback
import pandas as pd
import numpy as np
import warnings
import argparse # Added: Import the argparse module

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
    print("WARNING: sb3_contrib not found. RecurrentPPO agent will not be available. "
          "Please install with 'pip install sb3-contrib' for full functionality.")


from ..environments.env_loader import load_environments
from ..environments.custom_wrappers import FlattenAction
from ..data.config_loader import merge_configs, get_relevant_config_for_hash, generate_config_hash, convert_to_native_types
from ..data.data_loader import load_tick_data_for_range, load_kline_data_for_range
from .agent_utils import load_default_configs_for_training

def train_agent(config_override: dict = None, log_to_file: bool = True):
    """
    Main function to configure and run the RL agent training process.
    It loads configurations, prepares data, sets up environments,
    creates/loads the agent, and initiates training.
    """
    print("--- Starting Trading Agent Training Process ---")
    
    # 1. Load Configuration
    # Loads default configurations and merges with any provided overrides.
    config = load_default_configs_for_training()
    if config_override:
        config = merge_configs(config, config_override)

    run_settings = config["run_settings"]
    env_config = config["environment"]
    agent_type = config.get("agent_type", "PPO")
    algo_params = config.get(f"{agent_type.lower()}_params", {})
    
    # Extract the technical indicators configuration
    # This is now pulled directly from the loaded configuration
    technical_indicators_config = config.get("technical_indicators", {})
    if not technical_indicators_config:
        print("WARNING: 'technical_indicators' section not found in config. "
              "K-line data will be loaded without additional TA features.")


    # 2. Setup Logging and Paths
    # Generates a unique run ID based on hashed configuration for consistent logging.
    run_id = f"{generate_config_hash(get_relevant_config_for_hash(config))}_{run_settings.get('model_name', 'agent')}"
    log_dir = os.path.join(run_settings.get("log_dir_base", "logs/"), "training", run_id)
    model_save_dir = os.path.join(log_dir, "best_model")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save the effective configuration for this run for reproducibility
    with open(os.path.join(log_dir, "effective_train_config.json"), "w") as f:
        json.dump(convert_to_native_types(config), f, indent=4)
    print(f"Training run ID: {run_id}")
    print(f"Logs and models will be saved to: {log_dir}")

    # 3. Load Data
    print("\n--- Preparing Data ---")
    # Load K-line data for training, passing the new technical_indicators_config
    kline_df_train = load_kline_data_for_range(
        symbol=config["training_data"]["symbol"], # Changed: Get symbol from training_data
        start_date_str=config["training_data"]["start_date"], # Changed: Get start_date from training_data
        end_date_str=config["training_data"]["end_date"],     # Changed: Get end_date from training_data
        interval=config["training_data"]["timeframe"],        # Changed: Get timeframe from training_data
        technical_indicators_config=technical_indicators_config, # Changed: Pass the full TA config
        cache_dir=run_settings.get("historical_cache_dir"),
        binance_settings=config.get("binance_settings")
    )
    
    # Load tick data for training
    tick_df_train = load_tick_data_for_range(
        symbol=config["training_data"]["symbol"], # Changed: Get symbol from training_data
        start_date_str=config["training_data"]["start_date"], # Changed: Get start_date from training_data
        end_date_str=config["training_data"]["end_date"],     # Changed: Get end_date from training_data
        cache_dir=run_settings.get("historical_cache_dir"),
        binance_settings=config.get("binance_settings"),
        tick_resample_interval_ms=env_config.get("tick_resample_interval_ms")
    )
    
    # Basic data validation
    if kline_df_train.empty: raise ValueError("Training K-line data could not be loaded. Aborting.")
    if tick_df_train.empty: raise ValueError("Training tick data could not be loaded. Aborting.")
    print("Training data loaded successfully.")

    # 4. Create Training Environment
    print("\n--- Creating Environments ---")
    available_envs = load_environments()
    
    # Helper function to create an environment instance for stable-baselines3
    def create_env(tick_data, kline_data, env_config_dict, monitor_filepath=None):
        env_type = env_config_dict.get("env_type", "simple")
        env_class = available_envs.get(env_type)
        if not env_class:
            raise ValueError(f"Environment type '{env_type}' not found or supported.")
        
        base_env = env_class(tick_df=tick_data.copy(), kline_df_with_ta=kline_data.copy(), config=env_config_dict)
        return Monitor(FlattenAction(base_env), filename=monitor_filepath, allow_early_resets=True)

    # Create vectorized environment for training
    train_vec_env = make_vec_env(lambda: create_env(
        tick_df_train, kline_df_train, env_config, os.path.join(log_dir, "train_monitor.csv")), n_envs=1)
    
    # Setup VecNormalize for observation and reward normalization
    vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
    if run_settings.get("continue_from_existing_model") and os.path.exists(vec_normalize_path):
        print(f"Loading existing VecNormalize stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, train_vec_env)
        env.training = True # Ensure training mode for VecNormalize
    else:
        print("Initializing new VecNormalize stats.")
        env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)

    # 5. Setup Evaluation Environment & Callback
    eval_callback = None
    
    # Load K-line data for evaluation
    kline_df_eval = load_kline_data_for_range(
        symbol=config["evaluation_data"]["symbol"], # Changed: Get symbol from evaluation_data
        start_date_str=config["evaluation_data"]["start_date"], # Changed: Get start_date from evaluation_data
        end_date_str=config["evaluation_data"]["end_date"],     # Changed: Get end_date from evaluation_data
        interval=config["evaluation_data"]["timeframe"],        # Changed: Get timeframe from evaluation_data
        technical_indicators_config=technical_indicators_config, # Changed: Pass the full TA config
        cache_dir=run_settings.get("historical_cache_dir"),
        binance_settings=config.get("binance_settings")
    )

    # Load tick data for evaluation
    tick_df_eval = load_tick_data_for_range(
        symbol=config["evaluation_data"]["symbol"], # Changed: Get symbol from evaluation_data
        start_date_str=config["evaluation_data"]["start_date"], # Changed: Get start_date from evaluation_data
        end_date_str=config["evaluation_data"]["end_date"],     # Changed: Get end_date from evaluation_data
        cache_dir=run_settings.get("historical_cache_dir"),
        binance_settings=config.get("binance_settings"),
        tick_resample_interval_ms=env_config.get("tick_resample_interval_ms")
    )
    
    if not kline_df_eval.empty and not tick_df_eval.empty:
        eval_vec_env = make_vec_env(lambda: create_env(
            tick_df_eval, kline_df_eval, env_config, os.path.join(log_dir, "eval_monitor.csv")), n_envs=1)
        
        # Share observation normalization statistics from the training environment
        eval_env = VecNormalize(eval_vec_env, training=False, norm_obs=True, norm_reward=False)
        eval_env.obs_rms = env.obs_rms
        
        # Configure EvalCallback
        # Calculate eval_freq based on episodes if not explicitly set in run_settings
        eval_freq = int(run_settings.get("eval_freq_timesteps", 10000)) # Default to timesteps if not provided
        if run_settings.get("eval_freq_episodes"):
            # Estimate total timesteps per episode from training data length
            timesteps_per_episode = len(tick_df_train) 
            eval_freq = int(run_settings["eval_freq_episodes"] * timesteps_per_episode)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_save_dir,
            log_path=log_dir,
            eval_freq=max(1, eval_freq), # Ensure eval_freq is at least 1
            n_eval_episodes=run_settings.get("n_evaluation_episodes", 5),
            deterministic=True,
            # Removed 'render_mode=None' as it causes TypeError in older SB3 versions
        )
    else:
        print("WARNING: Evaluation data could not be loaded. Skipping evaluation callback.")

    # 6. Create or Load Agent
    model = None
    model_loaded = False
    device = run_settings.get("device", "cpu")
    
    # Map agent type string to SB3 model class
    model_class_map = {
        "PPO": PPO, "SAC": SAC, "DDPG": DDPG, "A2C": A2C,
        "RecurrentPPO": RecurrentPPO if SB3_CONTRIB_AVAILABLE else None
    }
    model_class = model_class_map.get(agent_type)

    if not model_class:
        raise ValueError(f"Agent type '{agent_type}' not supported. "
                         f"Available: {', '.join(k for k, v in model_class_map.items() if v is not None)}.")

    # Attempt to load an existing model if configured
    if run_settings.get("continue_from_existing_model"):
        potential_path = os.path.join(model_save_dir, "best_model.zip")
        if os.path.exists(potential_path):
            print(f"\n--- Loading existing best model from: {potential_path} ---")
            try:
                # Load the model, ensuring it's compatible with the current environment
                model = model_class.load(potential_path, env=env, device=device)
                model_loaded = True
            except Exception as e:
                print(f"Error loading existing model: {e}. Initializing a new model instead.")
                traceback.print_exc()

    # If no model was loaded, initialize a new one
    if not model:
        print("\n--- Initializing new model ---")
        policy = "MlpLstmPolicy" if agent_type == "RecurrentPPO" else "MlpPolicy"
        
        # Prepare algorithm-specific parameters
        params_for_init = algo_params.copy()
        params_for_init.pop("total_timesteps", None) # total_timesteps is for .learn(), not __init__
        
        # Handle policy_kwargs if it's a string representation of a dict
        if isinstance(params_for_init.get("policy_kwargs"), str):
            try:
                params_for_init["policy_kwargs"] = eval(params_for_init["policy_kwargs"])
            except Exception as e:
                print(f"WARNING: Could not parse policy_kwargs string: {params_for_init['policy_kwargs']}. Error: {e}")
                # Fallback to empty dict or handle as appropriate
                params_for_init["policy_kwargs"] = {}
            
        model = model_class(
            policy,
            env,
            verbose=0, # Set to 1 or 2 for more verbose output from SB3
            tensorboard_log=os.path.join("logs", "tensorboard_logs"),
            device=device,
            **params_for_init
        )

    # 7. Train the Agent
    print(f"\n--- Starting Training ---")
    total_timesteps = int(algo_params.get("total_timesteps", 1000000))
    print(f"Training for {total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            reset_num_timesteps=not model_loaded, # Reset timesteps if not continuing
            progress_bar=True,
            tb_log_name=run_id
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        traceback.print_exc()
    finally:
        print("\nSaving final model and normalization stats...")
        final_model_path = os.path.join(log_dir, "final_model.zip")
        env_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
        
        model.save(final_model_path)
        env.save(env_normalize_path)
        env.close() # Close environment after training
        print(f"Final model saved to: {final_model_path}")
        print(f"VecNormalize stats saved to: {env_normalize_path}")
        print("--- Training Process Completed ---")

if __name__ == "__main__":
    # This block allows running the script directly with command line arguments
    # You can pass a config_override here, e.g.:
    # python src/agents/train_agent.py --config_override '{"run_settings": {"total_timesteps": 50000}}'
    
    parser = argparse.ArgumentParser(description="Train an RL trading agent.")
    parser.add_argument("--config_override", type=str, help="JSON string to override default config settings.")
    
    # If you intend to use command line args, let me know to add argparse import.

    try:
        # Example of how to call with a custom config if argparse was setup
        # For now, we'll just call it without args unless the user needs CLI.
        # If config_override needs to come from CLI, a full argparse setup is needed.
        # train_agent(config_override=json.loads(args.config_override) if args.config_override else None)
        train_agent()
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")
        traceback.print_exc()
