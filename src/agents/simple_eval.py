import os
import pandas as pd
import numpy as np
import argparse
import traceback

# --- Stable Baselines 3 and Environment Imports ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- Custom Project Imports ---
from .agent_utils import load_default_configs_for_evaluation
from ..environments.env_loader import load_environments
from ..environments.custom_wrappers import FlattenAction
from ..data.data_loader import load_kline_data_for_range, load_tick_data_for_range
from ..utils import resolve_model_path


def main():
    """
    Main function to run a simple, single-episode evaluation with detailed step-by-step logging.
    """
    parser = argparse.ArgumentParser(description="Run a simple, verbose evaluation for a trained agent.")
    parser.add_argument("--run_id", type=str, help="Optional: The specific run_id to evaluate. If not provided, it will be derived from the current config.")
    args = parser.parse_args()

    print("--- Starting Simple Evaluation ---")
    
    config = load_default_configs_for_evaluation()
    
    if args.run_id:
        try:
            config["run_settings"]["model_name"] = args.run_id.split('_', 1)[1]
            print(f"Overriding model name to evaluate specific run_id: {args.run_id}")
        except IndexError:
            print(f"Warning: Could not parse model_name from run_id '{args.run_id}'. Using config default.")

    model_path, _ = resolve_model_path(config)
    
    vec_normalize_path = None
    if model_path and os.path.exists(model_path):
        model_dir = os.path.dirname(os.path.dirname(model_path))
        vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not model_path or not os.path.exists(vec_normalize_path):
        print("\nERROR: Could not find model path or vec_normalize.pkl path.")
        print(f"Attempted to find model: {model_path}")
        print(f"Attempted to find VecNormalize: {vec_normalize_path}")
        return

    print(f"\nModel Path: {model_path}")
    print(f"VecNormalize Path: {vec_normalize_path}\n")

    env_config = config["environment"]
    run_settings = config["run_settings"]
    binance_settings = config["binance_settings"]
    agent_type = config.get("agent_type", "PPO")
    env_type = run_settings.get("env_type", "simple")
    
    cache_dir = run_settings.get("historical_cache_dir", "data/")
    start_date = run_settings.get("start_date_eval")
    end_date = run_settings.get("end_date_eval")
    symbol = run_settings.get("default_symbol")
    interval = run_settings.get("historical_interval")
    
    env_config['log_level'] = 'none'

    print(f"Loading evaluation data for {symbol} from {start_date} to {end_date}...")
    kline_df = load_kline_data_for_range(symbol, start_date, end_date, interval, env_config["kline_price_features"], cache_dir, binance_settings, 'none')
    tick_df = load_tick_data_for_range(symbol, start_date, end_date, cache_dir, binance_settings, env_config.get("tick_resample_interval_ms"), 'none')

    if kline_df.empty or tick_df.empty:
        print("ERROR: Could not load data for the evaluation range.")
        return
    print("Data loaded successfully.\n")

    print("--- Creating Evaluation Environment ---")
    available_envs = load_environments()
    env_class = available_envs.get(env_type)
    if not env_class:
        raise ValueError(f"Environment type '{env_type}' not found.")
        
    def make_env():
        env = env_class(tick_df=tick_df, kline_df_with_ta=kline_df, config=env_config)
        return FlattenAction(env)

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False

    print(f"--- Loading Trained {agent_type} Agent ---")
    model = PPO.load(model_path, env=env)
    print("Agent loaded successfully.\n")

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
            # --- THIS IS THE FIX ---
            # Using the 'reward' variable directly, not looking inside 'info'
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