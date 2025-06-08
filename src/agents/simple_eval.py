import os
import pandas as pd
import numpy as np
import argparse
import traceback

# --- Stable Baselines 3 and Environment Imports ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    args = parser.parse_args()

    print("--- Starting Simple Evaluation ---")
    
    # 1. Load the complete, merged configuration
    config = load_default_configs_for_evaluation()
    
    if args.run_id:
        try:
            config["run_settings"]["model_name"] = args.run_id.split('_', 1)[1]
            print(f"Overriding model name to evaluate specific run_id: {args.run_id}")
        except IndexError:
            print(f"Warning: Could not parse model_name from run_id '{args.run_id}'. Using config default.")

    # 2. Automatically find the model path
    model_path, _ = resolve_model_path(config)
    
    vec_normalize_path = None
    if model_path and os.path.exists(model_path):
        model_dir = os.path.dirname(os.path.dirname(model_path))
        vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not model_path or not os.path.exists(vec_normalize_path):
        print("\nERROR: Could not find model path or vec_normalize.pkl path.")
        return

    print(f"\nModel Path: {model_path}")
    print(f"VecNormalize Path: {vec_normalize_path}\n")

    # 3. Get settings from the loaded config
    env_config = config["environment"]
    run_settings = config["run_settings"]
    binance_settings = config["binance_settings"]
    agent_type = config.get("agent_type", "PPO")
    env_type = run_settings.get("env_type", "simple")
    
    # 4. Load Evaluation Data
    kline_df = load_kline_data_for_range(run_settings["default_symbol"], run_settings["start_date_eval"], run_settings["end_date_eval"], run_settings["historical_interval"], env_config["kline_price_features"], run_settings["historical_cache_dir"], binance_settings, 'none')
    tick_df = load_tick_data_for_range(run_settings["default_symbol"], run_settings["start_date_eval"], run_settings["end_date_eval"], run_settings["historical_cache_dir"], binance_settings, env_config.get("tick_resample_interval_ms"), 'none')

    if kline_df.empty or tick_df.empty:
        print("ERROR: Could not load data for the evaluation range.")
        return
    print("Data loaded successfully.\n")

    # 5. Create Environment
    available_envs = load_environments()
    env_class = available_envs.get(env_type)
        
    def make_env():
        env = env_class(tick_df=tick_df, kline_df_with_ta=kline_df, config=env_config)
        return FlattenAction(env)

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False

    # 6. Load Model
    print(f"--- Loading Trained {agent_type} Agent ---")
    
    # --- THIS IS THE FIX ---
    # Get the device from config and pass it to the load function
    device = run_settings.get("device", "auto")
    model = PPO.load(model_path, env=env, device=device)
    
    print("Agent loaded successfully.\n")

    # 7. Run Single Episode and Log Details
    print("--- Running Single Evaluation Episode ---")
    obs = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        info = info[0]
        action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
        discrete_action = int(np.round(action[0][0]))
        
        print(
            f"Step: {info['current_step']:<5} | "
            f"Price: {info['current_tick_price']:<9.2f} | "
            f"Action: {action_map.get(discrete_action, 'Unknown'):<5} | "
            f"Position: {'OPEN' if info['position_open'] else 'FLAT':<5} | "
            f"Equity: {info['equity']:<10.2f} | "
            f"Reward: {reward[0]:<8.4f}"
        )
        episode_reward += reward[0]

    print("\n--- Episode Finished ---")
    print(f"Final Equity: {info['equity']:.2f}")
    print(f"Total Reward for Episode: {episode_reward:.4f}")
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()