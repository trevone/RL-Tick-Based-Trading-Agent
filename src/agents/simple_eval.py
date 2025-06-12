import os
import pandas as pd
import numpy as np
import argparse
import traceback

# --- Stable Baselines 3 and Environment Imports ---
from stable_baselines3 import PPO, SAC, DDPG, A2C # Added more agent types for compatibility
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    print("WARNING: sb3_contrib not found. RecurrentPPO agent will not be available for evaluation.")


# --- Custom Project Imports ---
from src.agents.agent_utils import load_default_configs_for_evaluation
from src.environments.env_loader import load_environments
from src.environments.custom_wrappers import FlattenAction
from src.data.data_loader import load_kline_data_for_range, load_tick_data_for_range
from src.utils import resolve_model_path


def main():
    """
    Main function to run a simple, single-episode evaluation with detailed step-by-step logging.
    """
    parser = argparse.ArgumentParser(description="Run a simple, verbose evaluation for a trained agent.")
    parser.add_argument("--run_id", type=str, help="Optional: The specific run_id to evaluate. If not provided, it will be derived from the current config.")
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on (e.g., 'cpu', 'cuda').")
    args = parser.parse_args()

    print("--- Starting Simple Evaluation ---")
    
    # 1. Load the complete, merged configuration
    config = load_default_configs_for_evaluation()
    
    # Extract config sections
    run_settings = config["run_settings"]
    env_config = config["environment"]
    binance_settings = config["binance_settings"]
    agent_type = config.get("agent_type", "PPO")

    # Extract the technical indicators configuration (same as used for training)
    technical_indicators_config = config.get("technical_indicators", {})
    if not technical_indicators_config:
        print("WARNING: 'technical_indicators' section not found in config. "
              "K-line data will be loaded without additional TA features for evaluation.")

    # Override model name if run_id is provided via command line
    if args.run_id:
        try:
            # Assuming run_id format is HASH_MODELNAME (e.g., "abcdef123_my_agent")
            # We want to extract the model name part after the first underscore
            parts = args.run_id.split('_', 1)
            if len(parts) > 1:
                config["run_settings"]["model_name"] = parts[1]
                print(f"Overriding model name to evaluate specific run_id: {args.run_id}")
            else:
                # If no underscore, assume the entire run_id is the model_name (less common but possible)
                config["run_settings"]["model_name"] = args.run_id
                print(f"Using provided run_id '{args.run_id}' directly as model_name.")
        except Exception as e:
            print(f"Warning: Error parsing model_name from run_id '{args.run_id}': {e}. Using config default.")

    # 2. Automatically find the model path (which includes the run_id hash)
    # resolve_model_path now needs the full config object to generate the run_id hash
    # and find the correct log directory.
    model_path, full_run_id = resolve_model_path(config) # resolve_model_path should return the full run_id it found
    
    vec_normalize_path = None
    if model_path and os.path.exists(model_path):
        # The vec_normalize.pkl is usually in the parent directory of the 'best_model.zip'
        model_dir = os.path.dirname(os.path.dirname(model_path)) # Go up from best_model.zip to run_id folder
        vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not model_path or not os.path.exists(model_path):
        print(f"\nERROR: Could not find model file at {model_path}. Please ensure a model has been trained and saved.")
        return
    if not os.path.exists(vec_normalize_path):
        print(f"\nERROR: Could not find VecNormalize stats file at {vec_normalize_path}. This is needed for observation normalization.")
        return

    print(f"\nModel Path: {model_path}")
    print(f"VecNormalize Path: {vec_normalize_path}\n")

    # 3. Load Evaluation Data
    # Get parameters from the 'evaluation_data' section, but use training_data's symbol
    eval_symbol = config["training_data"]["symbol"] # Use symbol from training_data config
    eval_start_date = config["evaluation_data"]["start_date"]
    eval_end_date = config["evaluation_data"]["end_date"]
    eval_timeframe = config["evaluation_data"]["timeframe"]

    print(f"Loading evaluation data for {eval_symbol} from {eval_start_date} to {eval_end_date} ({eval_timeframe})...")
    
    kline_df = load_kline_data_for_range(
        symbol=eval_symbol,
        start_date_str=eval_start_date,
        end_date_str=eval_end_date,
        interval=eval_timeframe,
        technical_indicators_config=technical_indicators_config, # Pass the TA config
        cache_dir=run_settings.get("historical_cache_dir"),
        binance_settings=binance_settings,
        log_level='normal' # Ensure logging during evaluation data loading
    )
    
    tick_df = load_tick_data_for_range(
        symbol=eval_symbol,
        start_date_str=eval_start_date,
        end_date_str=eval_end_date,
        cache_dir=run_settings.get("historical_cache_dir"),
        binance_settings=binance_settings,
        tick_resample_interval_ms=env_config.get("tick_resample_interval_ms"),
        log_level='normal' # Ensure logging during evaluation data loading
    )

    if kline_df.empty or tick_df.empty:
        print("ERROR: Could not load data for the evaluation range. Aborting evaluation.")
        return
    print("Data loaded successfully.\n")

    # 4. Create Environment
    available_envs = load_environments()
    env_type = env_config.get("env_type", "simple")
    env_class = available_envs.get(env_type)
    if not env_class:
        raise ValueError(f"Environment type '{env_type}' not found or supported.")

    def make_env():
        # Ensure kline_df_with_ta and tick_df are passed as copies if modified by env
        base_env = env_class(tick_df=tick_df.copy(), kline_df_with_ta=kline_df.copy(), config=env_config)
        return FlattenAction(base_env)

    env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats from the training run
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False # Set to False for evaluation
    env.norm_reward = False # Do not normalize rewards during evaluation

    # 5. Load Model
    print(f"--- Loading Trained {agent_type} Agent ---")
    
    # Map agent type string to SB3 model class
    model_class_map = {
        "PPO": PPO, "SAC": SAC, "DDPG": DDPG, "A2C": A2C,
        "RecurrentPPO": RecurrentPPO if SB3_CONTRIB_AVAILABLE else None
    }
    model_class = model_class_map.get(agent_type)

    if not model_class:
        raise ValueError(f"Agent type '{agent_type}' not supported. "
                         f"Available: {', '.join(k for k, v in model_class_map.items() if v is not None)}.")

    # Load the model using the determined device
    model = model_class.load(model_path, env=env, device=args.device) # Use device from argparse
    
    print("Agent loaded successfully.\n")

    # 6. Run Single Episode and Log Details
    print("--- Running Single Evaluation Episode ---")
    obs = env.reset()
    done = False
    episode_reward = 0.0

    # Print a clear header for the log table
    print(
        f"{'Timestamp':<20} | {'Step':<5} | {'Price':<9} | {'Action':<6} | " # Adjusted Action column width
        f"{'Status':<6} | {'Equity':<12} | {'Reward':<8}" # Adjusted Status column width
    )
    print("-" * 84) # Adjusted separator length

    info_at_decision_time = env.envs[0].unwrapped._get_info()

    while not done:
        action, _states = model.predict(obs, deterministic=True) # _states is often returned, capture it
        obs, reward, done, info_list = env.step(action)

        # episode_reward += reward[0] # Accumulate reward after step, but before next info_at_decision_time.
                                   # This line was duplicated, accumulating twice. Removed one.

        # Correctly interpret the action
        # The action from a continuous action space (like Box) needs to be mapped back
        # to the discrete actions of your environment for logging.
        # Assuming discrete action space is [-1, 0, 1] for short, flat, long,
        # or similar, and action[0][0] is the primary action output.
        # This mapping depends on your FlattenAction wrapper and environment's action space.
        # If FlattenAction maps to a Box(0, 2, (2,), float32) for example:
        # action[0] will be an array like [discrete_action_id, continuous_action_value]
        # discrete_action_id would be a float like 0.0, 1.0, 2.0. Need to round it.
        
        # Ensure action is handled based on its structure from model.predict
        # If action is an array of arrays (e.g., [[0.9]]), get the first element.
        # If action is just an array (e.g., [0.9]), use it directly.
        
        # Check action space type from environment for correct action interpretation
        # This assumes your env.action_space is either Discrete or Box(n)
        if hasattr(env.action_space, 'n'): # Discrete action space
            # Action is a single integer ID
            discrete_action_id = action[0] # For DummyVecEnv, action is usually np.array([action_id])
        elif hasattr(env.action_space, 'shape') and len(env.action_space.shape) == 1 and env.action_space.shape[0] > 1: # Box action space (e.g., flattened)
            # Assuming first element is discrete action, second is continuous magnitude
            discrete_action_id = int(round(action[0][0])) # Access first element of array in action, then first element of that inner array, then round
        else: # Fallback or more complex continuous space
            discrete_action_id = "N/A" # Cannot interpret directly for logging
            print(f"Warning: Cannot interpret action {action} for logging. Action space type: {type(env.action_space)}")

        action_str = env.envs[0].unwrapped.ACTION_MAP.get(discrete_action_id, f"ID:{discrete_action_id}") # Fallback to ID if not in map

        position_status = "OPEN" if info_at_decision_time.get("position_open", False) else "FLAT" # Use .get with default
        timestamp_str = info_at_decision_time.get('timestamp').strftime('%Y-%m-%d %H:%M:%S') if info_at_decision_time.get('timestamp') else "N/A"
        current_tick_price = info_at_decision_time.get('current_tick_price', 0.0)
        current_step = info_at_decision_time.get('current_step', 0)


        print(
            f"{timestamp_str:<20} | {current_step:<5} | "
            f"{current_tick_price:<9.2f} | {action_str:<6} | " # Adjusted action width
            f"{position_status:<6} | {info_at_decision_time.get('equity', 0.0):<12.2f} | " # Adjusted status width
            f"{reward[0]:<8.4f}"
        )

        # Update the info for the next loop iteration
        # info_list[0] contains the info dict for the *next* step, after the action was taken.
        # We need the info *before* the action for the current log line.
        # So, info_at_decision_time should be set from the info_list *after* the current step's logging.
        info_at_decision_time = info_list[0]
        episode_reward += reward[0] # Accumulate episode reward here

    print("\n--- Episode Finished ---")
    final_equity = info_at_decision_time.get('equity', 0)
    # Ensure initial_balance is correctly retrieved from the unwrapped env
    initial_balance = env.envs[0].unwrapped.initial_balance if hasattr(env.envs[0].unwrapped, 'initial_balance') else config['environment'].get('initial_balance', 10000)
    profit = final_equity - initial_balance
    profit_pct = (profit / initial_balance) * 100 if initial_balance != 0 else 0
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Profit: {profit:.2f} ({profit_pct:.2f}%)")
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()

