# evaluate_agent.py
import os
import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime, timezone

# --- Plotting Libraries ---
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import Stable Baselines3 PPO (assuming you trained with PPO)
from stable_baselines3 import PPO
# If you trained with RecurrentPPO, uncomment the line below and comment out PPO:
# from sb3_contrib import RecurrentPPO

from stable_baselines3.common.monitor import Monitor # For wrapping the environment to get episode info
from stable_baselines3.common.vec_env import VecNormalize # NEW: Import VecNormalize

# Import custom modules from the project.
try:
    from base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG
    # Import DEFAULT_TRAIN_CONFIG from train_simple_agent to ensure consistent hashing logic
    from train_simple_agent import DEFAULT_TRAIN_CONFIG as TRAIN_SCRIPT_FULL_DEFAULT_CONFIG
    from utils import (
        load_config,
        merge_configs,
        convert_to_native_types,
        fetch_and_cache_kline_data, # Use your existing data fetching functions
        fetch_continuous_aggregate_trades, # Use your existing data fetching functions
        resolve_model_path # For finding the trained model to evaluate.
    )
    from custom_wrappers import FlattenAction # Import your new custom wrapper
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import necessary modules. Ensure simple_trading_env.py, train_simple_agent.py, "
          f"utils.py, and custom_wrappers.py are accessible in the same directory or via PYTHONPATH. Error: {e}")
    import sys
    sys.exit(1)

# Default configuration for the evaluation script, aligned with your project's structure.
# This will be merged with config.yaml's 'evaluation_data' and other relevant sections.
DEFAULT_EVAL_CONFIG = {
    "run_settings": {
        "log_dir_base": "./logs/ppo_trading/", # Base directory where trained models are saved
        "model_name": "tick_trading_agent", # Model name to derive path
        "log_level": "normal", # "none", "normal", "detailed" for eval script's verbosity
        "eval_log_dir": "./logs/evaluation_runs/", # Directory for evaluation specific logs/charts
        "model_path": None, # Explicit path to the model .zip file (optional). If None, uses resolve_model_path.
        "alt_model_path": None, # Alternative explicit model path (e.g., final model if best is primary).
    },
    "evaluation_data": { # Specific dates for evaluation data
        "start_date_kline_eval": "2024-01-04", # Example: period after training/validation.
        "end_date_kline_eval": "2024-01-05",
        "start_date_tick_eval": "2024-01-04 00:00:00",
        "end_date_tick_eval": "2024-01-05 00:00:00",
    },
    "n_evaluation_episodes": 3,         # Default number of episodes to run for evaluation.
    "deterministic_prediction": True,   # Whether the agent should use deterministic actions.
    "print_step_info_freq": 50,         # Frequency to print step information if log_level is "normal".
    "binance_settings": {               # Settings for Binance data.
        "api_key": os.environ.get("BINANCE_API_KEY"), # Get from env var or config
        "api_secret": os.environ.get("BINANCE_API_SECRET"), # Get from env var or config
        "testnet": False,               # Default to Mainnet, change in config.yaml for testnet
        "historical_cache_dir": "./binance_data_cache/", # Shared cache dir
        "default_symbol": "BTCUSDT",
        "historical_interval": "1h",
        "api_request_delay_seconds": 0.2,
        "cache_file_type": "parquet"
    },
    "environment": DEFAULT_ENV_CONFIG # Environment config, will be merged with config.yaml overrides
}

def plot_performance(trade_history: list, price_data: pd.Series, eval_run_id: str, log_dir: str, title: str = "Agent Performance"):
    """
    Plots the price, account balance, and trade actions.
    - price_data should be a pandas Series with a DatetimeIndex and the 'Price' column.
    - trade_history is a list of dicts from env.trade_history.
    """
    if not trade_history:
        print("No trade history to plot.")
        return

    # Extract data for plotting
    equity_history = []
    balance_history = []
    trade_times_buy = []
    buy_prices = []
    trade_times_sell = []
    sell_prices = []

    # Process trade history for plotting
    for t_entry in trade_history:
        # Convert time to datetime object for plotting
        entry_time = pd.to_datetime(t_entry['time']) if 'time' in t_entry else None
        if entry_time is None: continue # Skip if no time available

        # Collect equity and balance over time (for any event type)
        if 'equity' in t_entry and 'balance' in t_entry:
            equity_history.append({'time': entry_time, 'equity': t_entry['equity'], 'balance': t_entry['balance']})

        # Collect buy/sell signals
        if t_entry.get('type') == 'buy':
            trade_times_buy.append(entry_time)
            buy_prices.append(t_entry['price'])
        elif t_entry.get('type') == 'sell':
            trade_times_sell.append(entry_time)
            sell_prices.append(t_entry['price'])
    
    # Create DataFrame for equity/balance history, ensuring time index
    if equity_history:
        equity_balance_df = pd.DataFrame(equity_history).set_index('time').sort_index()
    else:
        equity_balance_df = pd.DataFrame(columns=['equity', 'balance'], index=pd.to_datetime([]))


    # Filter price_data to the relevant evaluation period based on min/max of trade history
    # Or, if trade_history is empty, just use the original price_data range
    if not price_data.empty:
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index, utc=True)
        price_data_plot = price_data.sort_index()

        if not equity_balance_df.empty:
            min_plot_time = min(price_data_plot.index.min(), equity_balance_df.index.min())
            max_plot_time = max(price_data_plot.index.max(), equity_balance_df.index.max())
            # Ensure the price data covers the full range observed by the agent
            price_data_plot = price_data_plot[(price_data_plot.index >= min_plot_time) & (price_data_plot.index <= max_plot_time)]
        # Reindex price_data_plot to have a consistent frequency for plotting if needed,
        # but for scattered points and line, original index should be fine.
    else:
        print("Warning: No price data available for plotting.")
        return

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True) # Increased size for better visibility

    # --- Price Chart (Top) ---
    ax1.plot(price_data_plot.index, price_data_plot.values, label='Price', color='blue', linewidth=0.8)
    ax1.scatter(trade_times_buy, buy_prices, marker='^', color='green', s=100, label='Buy Signal', alpha=0.9, zorder=5)
    ax1.scatter(trade_times_sell, sell_prices, marker='v', color='red', s=100, label='Sell Signal', alpha=0.9, zorder=5)
    ax1.set_title(f'{title} - Price and Trade Signals for {price_data.name}')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Account Balance/Equity Chart (Bottom) ---
    if not equity_balance_df.empty:
        ax2.plot(equity_balance_df.index, equity_balance_df['equity'], label='Equity', color='purple', linewidth=1.5)
        ax2.plot(equity_balance_df.index, equity_balance_df['balance'], label='Balance', color='orange', linewidth=0.8, linestyle='--')
    else:
        print("Warning: No equity/balance history to plot.")

    ax2.set_title('Account Equity and Balance')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amount ($)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Format x-axis as dates
    fig.autofmt_xdate()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(log_dir, f"{eval_run_id}_performance_chart.png")
    plt.savefig(plot_filename)
    print(f"Performance chart saved to {plot_filename}")
    plt.show() # Display the plot

def main():
    """
    Main function to evaluate a trained agent on historical Binance data.
    Handles configuration loading, data preparation (Binance-only), model loading,
    running evaluation episodes, and reporting/saving results.
    """
    # --- 1. Load and Merge Configurations ---
    yaml_config_full = load_config()
    
    # Merge DEFAULT_EVAL_CONFIG with the full YAML config.
    effective_eval_config = merge_configs(DEFAULT_EVAL_CONFIG, yaml_config_full)

    current_log_level = effective_eval_config.get("run_settings", {}).get("log_level", "normal")
    
    # --- 2. Prepare Environment Configuration for Evaluation Instance ---
    eval_env_config_for_instance = DEFAULT_ENV_CONFIG.copy()
    
    if "environment" in yaml_config_full:
        eval_env_config_for_instance = merge_configs(eval_env_config_for_instance, yaml_config_full["environment"], "environment_base_yaml")
    
    evaluation_section_from_yaml = yaml_config_full.get("evaluation_data", {})
    if "environment_overrides" in evaluation_section_from_yaml and evaluation_section_from_yaml["environment_overrides"]:
         eval_env_config_for_instance = merge_configs(eval_env_config_for_instance, evaluation_section_from_yaml["environment_overrides"], "environment_evaluation_override")

    eval_env_config_for_instance["log_level"] = current_log_level
    eval_env_config_for_instance["custom_print_render"] = "none"

    if current_log_level == "detailed":
        print(f"Final environment config for instance: {json.dumps(convert_to_native_types(eval_env_config_for_instance), indent=2)}")

    effective_eval_config["environment"] = eval_env_config_for_instance


    # --- 3. Resolve Model Path ---
    model_load_path, alt_model_path_info = resolve_model_path(
        eval_specific_config=effective_eval_config["run_settings"],
        full_yaml_config=yaml_config_full,
        train_script_fallback_config=TRAIN_SCRIPT_FULL_DEFAULT_CONFIG,
        env_script_fallback_config=DEFAULT_ENV_CONFIG,
        log_level=current_log_level
    )

    if not model_load_path:
        print("Error: Could not determine a valid model path for evaluation. Exiting.")
        return

    # Determine logging directory for this evaluation run
    eval_run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    eval_log_dir = os.path.join(effective_eval_config["run_settings"]["eval_log_dir"], eval_run_id)
    os.makedirs(eval_log_dir, exist_ok=True)

    print(f"--- Evaluating Model: {model_load_path} (Log Level: {current_log_level}) ---")
    print(f"Evaluation run ID: {eval_run_id}")
    print(f"Evaluation logs and charts will be saved to: {eval_log_dir}")

    # Save the effective configuration for this evaluation run
    with open(os.path.join(eval_log_dir, "effective_eval_config.json"), "w") as f:
        json.dump(convert_to_native_types(effective_eval_config), f, indent=4)
    if current_log_level != "none":
        print("Effective evaluation configuration saved to effective_eval_config.json")
        if current_log_level == "detailed":
            print("Effective Evaluation Config (detailed):", json.dumps(convert_to_native_types(effective_eval_config), indent=2))

    # --- 4. Prepare Evaluation Data (using your current utils.py functions) ---
    eval_binance_settings = effective_eval_config["binance_settings"]
    eval_data_settings = effective_eval_config["evaluation_data"]

    print("\n--- Fetching and preparing K-line evaluation data ---")
    eval_kline_df = pd.DataFrame() # Initialize
    try:
        eval_kline_df = fetch_and_cache_kline_data(
            symbol=eval_binance_settings["default_symbol"],
            interval=eval_binance_settings["historical_interval"],
            start_date_str=eval_data_settings["start_date_kline_eval"],
            end_date_str=eval_data_settings["end_date_kline_eval"],
            cache_dir=eval_binance_settings["historical_cache_dir"],
            price_features_to_add=eval_env_config_for_instance["kline_price_features"], # Use env's features for TA calculation
            api_key=eval_binance_settings["api_key"],
            api_secret=eval_binance_settings["api_secret"],
            testnet=eval_binance_settings["testnet"],
            cache_file_type=eval_binance_settings.get("cache_file_type", "parquet"),
            log_level=current_log_level,
            api_request_delay_seconds=eval_binance_settings.get("api_request_delay_seconds", 0.2)
        )
        if eval_kline_df.empty:
            raise ValueError("K-line evaluation data not loaded.")
        print(f"K-line eval data loaded: {eval_kline_df.shape} from {eval_kline_df.index.min()} to {eval_kline_df.index.max()}")
    except Exception as e:
        print(f"ERROR: K-line evaluation data not loaded. Details: {e}")
        traceback.print_exc()
        exit(1)

    print(f"\n--- Fetching and preparing Tick evaluation data from {eval_data_settings['start_date_tick_eval']} to {eval_data_settings['end_date_tick_eval']} ---")
    eval_tick_df = pd.DataFrame() # Initialize
    try:
        eval_tick_df = fetch_continuous_aggregate_trades(
            symbol=eval_binance_settings["default_symbol"],
            start_date_str=eval_data_settings["start_date_tick_eval"],
            end_date_str=eval_data_settings["end_date_tick_eval"],
            cache_dir=eval_binance_settings["historical_cache_dir"],
            api_key=eval_binance_settings["api_key"],
            api_secret=eval_binance_settings["api_secret"],
            testnet=eval_binance_settings["testnet"],
            cache_file_type=eval_binance_settings.get("cache_file_type", "parquet"),
            log_level=current_log_level,
            api_request_delay_seconds=eval_binance_settings.get("api_request_delay_seconds", 0.2)
        )
        if eval_tick_df.empty:
            raise ValueError("Tick evaluation data not loaded.")
        print(f"Tick eval data loaded: {eval_tick_df.shape} from {eval_tick_df.index.min()} to {eval_tick_df.index.max()}")
    except Exception as e:
        print(f"ERROR: Tick evaluation data not loaded. Details: {e}")
        traceback.print_exc()
        exit(1)

    # --- 5. Create Evaluation Environment and Load Model ---
    eval_env = None
    try:
        # Create the base environment
        base_eval_env = SimpleTradingEnv(tick_df=eval_tick_df.copy(), kline_df_with_ta=eval_kline_df.copy(), config=eval_env_config_for_instance)
        # Apply the FlattenAction wrapper
        wrapped_eval_env = FlattenAction(base_eval_env) # Apply the wrapper here
        
        # NEW: Apply VecNormalize for observation standardization
        # During evaluation, it's crucial to load the normalization statistics from training.
        # The `VecNormalize` object is saved as 'vec_normalize.pkl' in the model's directory.
        # We need to determine the directory of the `model_load_path` to find the .pkl file.
        model_dir = os.path.dirname(model_load_path)
        vec_normalize_stats_path = os.path.join(model_dir, "vec_normalize.pkl")
        
        # Instantiate VecNormalize with norm_obs=True, and then load the statistics
        eval_env_normalized = VecNormalize(wrapped_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.) # NEW
        
        if os.path.exists(vec_normalize_stats_path):
            eval_env_normalized = VecNormalize.load(vec_normalize_stats_path, eval_env_normalized) # Load stats into the new instance # NEW
            if current_log_level != "none": print(f"VecNormalize statistics loaded from: {vec_normalize_stats_path}")
        else:
            if current_log_level != "none": print(f"WARNING: VecNormalize stats not found at {vec_normalize_stats_path}. Evaluation observations will NOT be normalized consistently with training. This can lead to poor performance.")
            # If stats are not found, it's better to explicitly normalize to avoid issues.
            # But the agent might perform poorly if it was trained on normalized data.
            # For robust evaluation, it's critical these stats are present.

        eval_env = Monitor(eval_env_normalized, filename=os.path.join(eval_log_dir, "eval_monitor.csv")) # Wrap with Monitor
        
        print("\nEvaluation environment created successfully and configured for normalization.")
    except Exception as e:
        print(f"Error creating evaluation environment: {e}")
        traceback.print_exc()
        if eval_env: eval_env.close()
        exit(1)

    try:
        # Load the trained model (PPO expects the wrapped env)
        # The env passed to model.load() should be a VecEnv, if the model was trained with VecEnv.
        # And if it was trained with VecNormalize, the env here should also be VecNormalize
        model = PPO.load(model_load_path, env=eval_env) # Pass the Monitor-wrapped (VecNormalize) env
        # If using RecurrentPPO, change above to: model = RecurrentPPO.load(model_load_path, env=eval_env)
        print("Model loaded successfully for evaluation.")
    except Exception as e:
        print(f"Error loading agent model from '{model_load_path}': {e}")
        traceback.print_exc()
        eval_env.close()
        return

    # --- 6. Run Evaluation Episodes ---
    num_eval_episodes = effective_eval_config.get('n_evaluation_episodes', 3)
    print(f"Starting evaluation for {num_eval_episodes} episodes...")
    all_episodes_rewards, all_episodes_profits_pct = [], []
    all_combined_trade_history = [] # To store all trade history for plotting and saving

    for episode in range(num_eval_episodes):
        obs, info = eval_env.reset()
        
        # For RecurrentPPO, you would manage LSTM states here:
        # lstm_states = None
        # episode_starts = np.array([True])

        terminated, truncated = False, False
        episode_reward, current_episode_step = 0, 0
        print(f"\n--- Evaluation Episode {episode + 1}/{num_eval_episodes} ---")

        while not (terminated or truncated):
            # action_array from model.predict will be the flattened action (e.g., np.array([action_choice_float, profit_target_float]))
            action_array, _states = model.predict(obs, deterministic=effective_eval_config.get("deterministic_prediction", True))
            
            # Pass the raw action_array from the model to eval_env.step().
            # The FlattenAction wrapper will internally convert it back to (discrete, Box) tuple for SimpleTradingEnv.
            obs, reward, terminated, truncated, info = eval_env.step(action_array)
            
            # For logging, extract the discrete action and profit target from the `action_array`
            discrete_action_for_log = int(np.round(action_array[0])) # Convert back to discrete (0, 1, 2) for logging
            profit_target_param_for_log = action_array[1] # The continuous profit target
            
            episode_reward += reward
            current_episode_step += 1
            
            if current_log_level == "normal" and \
                 current_episode_step % effective_eval_config.get("print_step_info_freq", 50) == 0 :
                # Access the underlying SimpleTradingEnv's ACTION_MAP using .env.env (VecNormalize wraps Monitor which wraps FlattenAction which wraps SimpleTradingEnv)
                print(f"  Step: {info.get('current_step')}, Action: {eval_env.env.env.env.ACTION_MAP.get(discrete_action_for_log)}, "
                      f"Reward: {reward:.3f}, Equity: {info.get('equity',0):.2f}")
            elif current_log_level == "detailed":
                print(f"  Step: {info.get('current_step')}, Action: {eval_env.env.env.env.ACTION_MAP.get(discrete_action_for_log)}, ProfitTgt: {profit_target_param_for_log:.4f}, "
                      f"Reward: {reward:.3f}, Equity: {info.get('equity',0):.2f}, Pos: {info.get('position_open')}")

        final_equity = info.get('equity', eval_env.initial_balance)
        # Access initial_balance from the unwrapped SimpleTradingEnv
        initial_balance_env = eval_env.env.env.env.initial_balance # Access initial_balance from the base env
        episode_profit_pct = ((final_equity - initial_balance_env) / (initial_balance_env + 1e-9)) * 100

        print(f"Episode finished. Steps: {current_episode_step}. Reward: {episode_reward:.2f}. "
              f"Final Equity: {final_equity:.2f} Profit: {episode_profit_pct:.2f}%")
        all_episodes_rewards.append(episode_reward)
        all_episodes_profits_pct.append(episode_profit_pct)

        # Append current episode's trade history to the combined list
        all_combined_trade_history.extend(eval_env.env.env.env.trade_history) # Access trade_history from the base env

        if current_log_level == "detailed" or (eval_env.env.env.env.trade_history and len(eval_env.env.env.env.trade_history) > 1):
             print(f"--- Trade History Ep {episode+1} (Last 10 trades if any) ---")
             temp_episode_trades_native_for_print = convert_to_native_types(eval_env.env.env.env.trade_history)
             relevant_trades = [t for t in temp_episode_trades_native_for_print if t.get('type') != 'initial_balance']
             if relevant_trades:
                [print(f"  {t}") for t in relevant_trades[-10:]]
             else:
                print("  No trades executed in this episode.")
        elif current_log_level == "normal":
            print(f"  Total trades this episode: {info.get('num_trades_in_episode',0)}")

    # --- 7. Save Full Trade History for All Episodes ---
    trade_history_filename = f"evaluation_{eval_run_id}_trade_history.json"
    trade_history_save_path = os.path.join(eval_log_dir, trade_history_filename)

    if all_combined_trade_history:
        try:
            data_to_save = convert_to_native_types(all_combined_trade_history)
            os.makedirs(os.path.dirname(trade_history_save_path), exist_ok=True)
            with open(trade_history_save_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"\nFull evaluation trade history saved to: {trade_history_save_path}")
        except Exception as e:
            print(f"Error saving trade history: {e}")
            traceback.print_exc()

    # --- 8. Print Overall Evaluation Summary ---
    print("\n--- Overall Evaluation Summary ---")
    num_episodes_actually_run = len(all_episodes_rewards)
    if num_episodes_actually_run > 0:
        print(f"Number of episodes run: {num_episodes_actually_run}")
        print(f"Average Reward: {float(np.mean(all_episodes_rewards)):.2f} (Std: {float(np.std(all_episodes_rewards)):.2f})")
        print(f"Median Reward: {float(np.median(all_episodes_rewards)):.2f}")
        print(f"Average Profit: {float(np.mean(all_episodes_profits_pct)):.2f}% (Std: {float(np.std(all_episodes_profits_pct)):.2f}%)")
        print(f"Median Profit: {float(np.median(all_episodes_profits_pct)):.2f}%")
        print(f"Min Profit: {float(np.min(all_episodes_profits_pct)):.2f}%, Max Profit: {float(np.max(all_episodes_profits_pct)):.2f}%")
    else:
        print("No evaluation episodes completed or no data to summarize.")

    # --- 9. Plotting ---
    print("\n--- Generating Performance Chart ---")
    plot_price_data_series = eval_tick_df['Price'].copy()
    plot_price_data_series.name = eval_binance_settings['default_symbol'] # Set series name for plot title

    plot_performance(all_combined_trade_history, plot_price_data_series, eval_run_id, eval_log_dir,
                     title=f"Agent Evaluation: {eval_binance_settings['default_symbol']} ({eval_data_settings['start_date_tick_eval']} to {eval_data_settings['end_date_tick_eval']})")

    # --- Cleanup ---
    if eval_env:
        eval_env.close()
        print("Evaluation environment closed.")
    print("\n--- Evaluation script finished ---")


if __name__ == '__main__':
    main()