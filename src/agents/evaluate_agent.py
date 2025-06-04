# evaluate_agent.py
import pytest # Import pytest for the fixture if not already imported by user
import os
import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime, timezone

# --- Plotting Libraries ---
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import Stable Baselines3 algorithms dynamically
from stable_baselines3 import PPO, SAC, DDPG, A2C
# For RecurrentPPO (if available and chosen)
try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    print("WARNING: sb3_contrib (for RecurrentPPO) not found. RecurrentPPO will not be available.")

from stable_baselines3.common.monitor import Monitor # For wrapping the environment to get episode info
from stable_baselines3.common.vec_env import VecNormalize # Import VecNormalize for observation standardization

# Import custom modules from the project's new structure
from src.environments.base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG
from src.environments.custom_wrappers import FlattenAction
from src.data.utils import (
    load_config,
    merge_configs,
    convert_to_native_types,
    load_tick_data_for_range,
    load_kline_data_for_range,
    resolve_model_path,
    DATA_CACHE_DIR # For passing to load functions
)


# --- NEW: Function to load default configurations from files ---
def load_default_configs_for_evaluation(config_dir="configs/defaults") -> dict:
    """Loads default configurations from the specified directory for evaluation."""
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "environment.yaml"),
        os.path.join(config_dir, "binance_settings.yaml"),
        os.path.join(config_dir, "evaluation_data.yaml"),
        os.path.join(config_dir, "hash_keys.yaml"), # Needed for resolve_model_path
        os.path.join(config_dir, "ppo_params.yaml"), # Include all algo params for hashing to resolve model
        os.path.join(config_dir, "sac_params.yaml"),
        os.path.join(config_dir, "ddpg_params.yaml"),
        os.path.join(config_dir, "a2c_params.yaml"),
        os.path.join(config_dir, "recurrent_ppo_params.yaml"),
    ]
    
    # Use the new load_config from src.data.utils which merges multiple files
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)


# --- Fixture to disable plotting in tests (add to your test_evaluation.py not src/agents/evaluate_agent.py) ---
# This part belongs in tests/agents/test_evaluation.py, but I'm including it here as a reminder.
# @pytest.fixture(autouse=True)
# def disable_plotting_in_tests(monkeypatch):
#     monkeypatch.setattr(plt, 'show', lambda: None)
#     monkeypatch.setattr(plt, 'savefig', lambda *args, **kwargs: None)


def plot_performance(trade_history: list, price_data: pd.Series, eval_run_id: str, log_dir: str, title: str = "Agent Performance"):
    """
    Plots the price, account balance, and trade actions.
    - price_data should be a pandas Series with a DatetimeIndex and the 'Price' column.
    - trade_history is a list of dicts from env.trade_history.
    """
    if not trade_history:
        print("No trade history to plot.")
        return

    # Filter out 'initial_balance' events as they don't represent a trade signal on the chart
    trade_events = [t for t in trade_history if t.get('type') not in ['initial_balance', 'sell_eof_auto', 'sell_ruin_auto']]

    # Extract data for plotting
    equity_history = []
    balance_history = []
    trade_times_buy = []
    buy_prices = []
    trade_times_sell = []
    sell_prices = []

    # Process trade history for plotting
    for t_entry in trade_history: # Use full trade_history for equity/balance tracking
        # Convert time to datetime object for plotting
        entry_time = pd.to_datetime(t_entry['time']) if 'time' in t_entry else None
        if entry_time is None: continue # Skip if no time available

        # Collect equity and balance over time (for any event type)
        if 'equity' in t_entry and 'balance' in t_entry:
            equity_history.append({'time': entry_time, 'equity': t_entry['equity'], 'balance': t_entry['balance']})

    for t_event in trade_events: # Use filtered trade_events for buy/sell markers
        entry_time = pd.to_datetime(t_event['time'])
        if t_event.get('type') == 'buy':
            trade_times_buy.append(entry_time)
            buy_prices.append(t_event['price'])
        elif t_event.get('type') == 'sell':
            trade_times_sell.append(entry_time)
            sell_prices.append(t_event['price'])
    
    # Create DataFrame for equity/balance history, ensuring time index
    if equity_history:
        equity_balance_df = pd.DataFrame(equity_history).set_index('time').sort_index()
        # Drop duplicates in index, keeping the last one for potential multiple events at same time
        equity_balance_df = equity_balance_df[~equity_balance_df.index.duplicated(keep='last')]
    else:
        equity_balance_df = pd.DataFrame(columns=['equity', 'balance'], index=pd.to_datetime([]))


    # Filter price_data to the relevant evaluation period based on min/max of trade history
    if not price_data.empty:
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index, utc=True)
        price_data_plot = price_data.sort_index()

        if not equity_balance_df.empty:
            min_plot_time = equity_balance_df.index.min()
            max_plot_time = equity_balance_df.index.max()
            # Extend plot range slightly beyond trade history for visual context
            min_plot_time -= pd.Timedelta(minutes=5)
            max_plot_time += pd.Timedelta(minutes=5)
            
            # Ensure the price data covers the full range observed by the agent, potentially extending slightly
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
    Handles configuration loading, data preparation, model loading,
    running evaluation episodes, and reporting/saving results.
    """
    # --- 1. Load and Merge Configurations ---
    effective_eval_config = load_default_configs_for_evaluation()
    
    current_log_level = effective_eval_config.get("run_settings", {}).get("log_level", "normal")
    agent_type = effective_eval_config.get("agent_type", "PPO") # Default to PPO

    # DEBUG PRINT
    if current_log_level == "detailed":
        print(f"DEBUG: effective_eval_config at start of main:\n{json.dumps(convert_to_native_types(effective_eval_config), indent=2)}")

    # Apply environment overrides specified in the evaluation_data section of config.yaml
    # This is already handled by load_config if environment_overrides are top-level in config.yaml
    # or if evaluation_data section is merged correctly.
    # Ensure explicit env config for env instance uses the correct source
    eval_env_config_for_instance = effective_eval_config["environment"]

    eval_env_config_for_instance["log_level"] = "none" # Always suppress env logging during eval
    eval_env_config_for_instance["custom_print_render"] = "none"

    if current_log_level == "detailed":
        print(f"Final effective evaluation config:\n{json.dumps(convert_to_native_types(effective_eval_config), indent=2)}")

    # --- 2. Resolve Model Path ---
    # `resolve_model_path` needs the full effective config for hashing consistency.
    model_load_path, alt_model_path_info = resolve_model_path(
        effective_config=effective_eval_config,
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
    print(f"Agent Type: {agent_type}")

    # Save the effective configuration for this evaluation run
    with open(os.path.join(eval_log_dir, "effective_eval_config.json"), "w") as f:
        json.dump(convert_to_native_types(effective_eval_config), f, indent=4)
    if current_log_level != "none":
        print("Effective evaluation configuration saved to effective_eval_config.json")

    # --- 3. Prepare Evaluation Data ---
    eval_binance_settings = effective_eval_config["binance_settings"]
    eval_data_settings = effective_eval_config["evaluation_data"]

    # Use the kline_price_features from the effective environment config
    kline_features_for_data_fetch = effective_eval_config["environment"]["kline_price_features"]

    print("\n--- Fetching and preparing K-line evaluation data ---")
    eval_kline_df = pd.DataFrame() # Initialize
    try:
        eval_kline_df = load_kline_data_for_range(
            symbol=eval_binance_settings["default_symbol"],
            interval=eval_binance_settings["historical_interval"],
            start_date_str=eval_data_settings["start_date_kline_eval"],
            end_date_str=eval_data_settings["end_date_kline_eval"],
            cache_dir=DATA_CACHE_DIR, # Use DATA_CACHE_DIR
            price_features=kline_features_for_data_fetch,
            binance_settings=eval_binance_settings # Pass binance settings for API keys/testnet
        )
        if eval_kline_df.empty:
            raise ValueError("K-line evaluation data is empty. Cannot proceed with evaluation.")
        print(f"K-line eval data loaded: {eval_kline_df.shape} from {eval_kline_df.index.min()} to {eval_kline_df.index.max()}")
    except Exception as e:
        print(f"ERROR: K-line evaluation data not loaded. Details: {e}")
        traceback.print_exc()
        exit(1) # This exit is what caused the FAILED test

    print(f"\n--- Fetching and preparing Tick evaluation data from {eval_data_settings['start_date_tick_eval']} to {eval_data_settings['end_date_tick_eval']} ---")
    eval_tick_df = pd.DataFrame() # Initialize
    try:
        eval_tick_df = load_tick_data_for_range(
            symbol=eval_binance_settings["default_symbol"],
            start_date_str=eval_data_settings["start_date_tick_eval"],
            end_date_str=eval_data_settings["end_date_tick_eval"],
            cache_dir=DATA_CACHE_DIR, # Use DATA_CACHE_DIR
            binance_settings=eval_binance_settings # Pass binance settings for API keys/testnet
        )
        if eval_tick_df.empty:
            raise ValueError("Tick evaluation data is empty. Cannot proceed with evaluation.")
        print(f"Tick eval data loaded: {eval_tick_df.shape} from {eval_tick_df.index.min()} to {eval_tick_df.index.max()}")
    except Exception as e:
        print(f"ERROR: Tick evaluation data not loaded. Details: {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        exit(1) # This exit is also what caused the FAILED test

    # --- 4. Create Evaluation Environment and Load Model ---
    eval_env = None
    try:
        # Create the base environment instance using the effective environment config
        base_eval_env = SimpleTradingEnv(tick_df=eval_tick_df.copy(), kline_df_with_ta=eval_kline_df.copy(), config=eval_env_config_for_instance)
        # Apply the FlattenAction wrapper
        wrapped_eval_env = FlattenAction(base_eval_env)
        
        # Load VecNormalize statistics from the training run's log directory
        # The model's path is something like logs/training/<HASH_MODELNAME>/best_model/best_model.zip
        # So the VecNormalize stats are in logs/training/<HASH_MODELNAME>/vec_normalize.pkl
        model_training_run_dir = os.path.dirname(os.path.dirname(model_load_path)) # Go up two levels
        vec_normalize_stats_path = os.path.join(model_training_run_dir, "vec_normalize.pkl")
        
        # Instantiate VecNormalize. norm_reward=False for this env.
        # clip_obs must be consistent with training (typically 10.0 or from SB3 defaults)
        eval_env_normalized = VecNormalize(wrapped_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        if os.path.exists(vec_normalize_stats_path):
            eval_env_normalized = VecNormalize.load(vec_normalize_stats_path, eval_env_normalized) # Load stats into the new instance
            # Ensure evaluation environment is in evaluation mode (important for VecNormalize)
            eval_env_normalized.training = False # This disables updating mean/std
            eval_env_normalized.norm_reward = False # Ensure reward normalization is off for evaluation if it was during training
            if current_log_level != "none": print(f"VecNormalize statistics loaded and applied from: {vec_normalize_stats_path}")
        else:
            print(f"WARNING: VecNormalize stats not found at {vec_normalize_stats_path}. Evaluation observations will NOT be normalized consistently with training. This can lead to poor performance.")
            print("Consider ensuring `vec_normalize.pkl` is saved by `train_agent.py` in the model's training log directory.")
            # If stats are not found, the agent will receive unnormalized observations, likely leading to poor performance.

        # Monitor should wrap the final environment passed to the model (VecNormalize in this case)
        eval_env = Monitor(eval_env_normalized, filename=os.path.join(eval_log_dir, "eval_monitor.csv"))
        
        print("\nEvaluation environment created successfully and configured for normalization.")
    except Exception as e:
        print(f"Error creating evaluation environment: {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        exit(1)

    try:
        # Dynamically load the correct model class
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
            else:
                raise ImportError("RecurrentPPO model type requested but sb3_contrib is not installed.")
        else:
            raise ValueError(f"Unknown agent type '{agent_type}'. Supported: PPO, SAC, DDPG, A2C, RecurrentPPO.")

        # Load the trained model (pass the VecNormalize env).
        model = model_class.load(model_load_path, env=eval_env)
        print("Model loaded successfully for evaluation.")
    except Exception as e:
        print(f"Error loading agent model from '{model_load_path}': {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        eval_env.close()
        return

    # --- 5. Run Evaluation Episodes ---
    num_eval_episodes = effective_eval_config.get('n_evaluation_episodes', 3)
    print(f"Starting evaluation for {num_eval_episodes} episodes...")
    all_episodes_rewards, all_episodes_profits_pct = [], []
    all_combined_trade_history = [] # To store all trade history for plotting and saving

    for episode in range(num_eval_episodes):
        obs, info = eval_env.reset() # Reset the VecNormalize env
        
        terminated, truncated = False, False
        episode_reward, current_episode_step = 0, 0
        
        # Access initial_balance from the unwrapped SimpleTradingEnv instance
        # This is complex due to multiple wrappers: VecNormalize -> Monitor -> FlattenAction -> SimpleTradingEnv
        base_env_instance = eval_env.venv.envs[0].env.env # For Monitor(VecNormalize(FlattenAction(SimpleTradingEnv)))
        initial_balance_this_episode = base_env_instance.initial_balance

        print(f"\n--- Evaluation Episode {episode + 1}/{num_eval_episodes} (Initial Balance: {initial_balance_this_episode:.2f}) ---")

        while not (terminated or truncated):
            # action_array from model.predict will be the flattened action (e.g., np.array([action_choice_float, profit_target_float]))
            action_array, _states = model.predict(obs, deterministic=effective_eval_config.get("deterministic_prediction", True))
            
            # Pass the raw action_array from the model to eval_env.step().
            # The FlattenAction wrapper will internally convert it back to (discrete, Box) tuple for SimpleTradingEnv.
            obs, reward, terminated, truncated, info = eval_env.step(action_array)
            
            # info from monitor is a list of dicts, so we take the first one (assuming n_envs=1)
            info = info[0] 

            # For logging, extract the discrete action and profit target from the `action_array`
            discrete_action_for_log = int(np.round(action_array[0])) # Convert back to discrete (0, 1, 2) for logging
            profit_target_param_for_log = action_array[1] # The continuous profit target
            
            episode_reward += reward[0] # Reward from VecEnv is an array, take first element for single env
            current_episode_step += 1
            
            if current_log_level == "normal" and \
                 current_episode_step % effective_eval_config.get("print_step_info_freq", 50) == 0 :
                # Access the underlying SimpleTradingEnv's ACTION_MAP: VecNormalize -> Monitor -> FlattenAction -> SimpleTradingEnv
                print(f"  Step: {info.get('current_step')}, Action: {base_env_instance.ACTION_MAP.get(discrete_action_for_log)}, "
                      f"Reward: {reward[0]:.3f}, Equity: {info.get('equity',0):.2f}")
            elif current_log_level == "detailed":
                print(f"  Step: {info.get('current_step')}, Action: {base_env_instance.ACTION_MAP.get(discrete_action_for_log)}, ProfitTgt: {profit_target_param_for_log:.4f}, "
                      f"Reward: {reward[0]:.3f}, Equity: {info.get('equity',0):.2f}, Pos: {info.get('position_open')}")

        final_equity = info.get('equity', initial_balance_this_episode)
        episode_profit_pct = ((final_equity - initial_balance_this_episode) / (initial_balance_this_episode + 1e-9)) * 100

        print(f"Episode finished. Steps: {current_episode_step}. Reward: {episode_reward:.2f}. "
              f"Final Equity: {final_equity:.2f} Profit: {episode_profit_pct:.2f}%")
        all_episodes_rewards.append(episode_reward)
        all_episodes_profits_pct.append(episode_profit_pct)

        # Append current episode's trade history to the combined list
        # Access trade_history from the base SimpleTradingEnv instance
        all_combined_trade_history.extend(base_env_instance.trade_history)

        if current_log_level == "detailed" or (base_env_instance.trade_history and len(base_env_instance.trade_history) > 1):
             print(f"--- Trade History Ep {episode+1} (Last 10 trades if any) ---")
             temp_episode_trades_native_for_print = convert_to_native_types(base_env_instance.trade_history)
             relevant_trades = [t for t in temp_episode_trades_native_for_print if t.get('type') != 'initial_balance']
             if relevant_trades:
                [print(f"  {t}") for t in relevant_trades[-10:]]
             else:
                print("  No trades executed in this episode.")
        elif current_log_level == "normal":
            print(f"  Total trades this episode: {info.get('num_trades_in_episode',0)}")

    # --- 6. Save Full Trade History for All Episodes ---
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
            traceback_str = traceback.format_exc()
            print(traceback_str)

    # --- 7. Print Overall Evaluation Summary ---
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

    # --- 8. Plotting ---
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