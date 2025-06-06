# src/agents/evaluate_agent.py
import os
import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime, timezone

# --- Plotting Libraries ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend BEFORE pyplot is imported
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import Stable Baselines3 algorithms dynamically
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False

# --- UPDATED IMPORTS ---
from src.environments.base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG
from src.environments.custom_wrappers import FlattenAction
from src.data.config_loader import load_config, merge_configs, convert_to_native_types
from src.data.data_loader import load_tick_data_for_range, load_kline_data_for_range
from src.utils import resolve_model_path
# --- END UPDATED IMPORTS ---

def load_default_configs_for_evaluation(config_dir="configs/defaults") -> dict:
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "environment.yaml"),
        os.path.join(config_dir, "binance_settings.yaml"),
        os.path.join(config_dir, "evaluation_data.yaml"),
        os.path.join(config_dir, "hash_keys.yaml"),
        os.path.join(config_dir, "ppo_params.yaml"),
        os.path.join(config_dir, "sac_params.yaml"),
        os.path.join(config_dir, "ddpg_params.yaml"),
        os.path.join(config_dir, "a2c_params.yaml"),
        os.path.join(config_dir, "recurrent_ppo_params.yaml"),
    ]
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)


def plot_performance(trade_history: list, price_data: pd.Series, eval_run_id: str, log_dir: str, log_level: str = "normal", title: str = "Agent Performance"):
    # ... (rest of the function remains unchanged)
    if not trade_history:
        if log_level != "none":
            print("No trade history to plot.")
        return

    trade_events = [t for t in trade_history if t.get('type') not in ['initial_balance', 'sell_eof_auto', 'sell_ruin_auto']]
    equity_history = []
    trade_times_buy = []
    buy_prices = []
    trade_times_sell = []
    sell_prices = []

    for t_entry in trade_history:
        entry_time = pd.to_datetime(t_entry['time']) if 'time' in t_entry else None
        if entry_time is None: continue

        if 'equity' in t_entry and 'balance' in t_entry:
            equity_history.append({'time': entry_time, 'equity': t_entry['equity'], 'balance': t_entry['balance']})

    for t_event in trade_events:
        entry_time = pd.to_datetime(t_event['time'])
        if t_event.get('type') == 'buy':
            trade_times_buy.append(entry_time)
            buy_prices.append(t_event['price'])
        elif t_event.get('type') == 'sell':
            trade_times_sell.append(entry_time)
            sell_prices.append(t_event['price'])

    if equity_history:
        equity_balance_df = pd.DataFrame(equity_history).set_index('time').sort_index()
        equity_balance_df = equity_balance_df[~equity_balance_df.index.duplicated(keep='last')]
    else:
        equity_balance_df = pd.DataFrame(columns=['equity', 'balance'], index=pd.to_datetime([]))

    if not price_data.empty:
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index, utc=True)
        price_data_plot = price_data.sort_index()

        if not equity_balance_df.empty: # Adjust price data range if equity data exists
            min_plot_time = equity_balance_df.index.min()
            max_plot_time = equity_balance_df.index.max()
            min_plot_time -= pd.Timedelta(minutes=5)
            max_plot_time += pd.Timedelta(minutes=5)
            price_data_plot = price_data_plot[(price_data_plot.index >= min_plot_time) & (price_data_plot.index <= max_plot_time)]
    else:
        if log_level != "none":
            print("Warning: No price data available for plotting.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    ax1.plot(price_data_plot.index, price_data_plot.values, label='Price', color='blue', linewidth=0.8)
    ax1.scatter(trade_times_buy, buy_prices, marker='^', color='green', s=100, label='Buy Signal', alpha=0.9, zorder=5)
    ax1.scatter(trade_times_sell, sell_prices, marker='v', color='red', s=100, label='Sell Signal', alpha=0.9, zorder=5)
    ax1.set_title(f'{title} - Price and Trade Signals for {price_data.name}')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    if not equity_balance_df.empty:
        ax2.plot(equity_balance_df.index, equity_balance_df['equity'], label='Equity', color='purple', linewidth=1.5)
        ax2.plot(equity_balance_df.index, equity_balance_df['balance'], label='Balance', color='orange', linewidth=0.8, linestyle='--')
    else:
        if log_level != "none":
            print("Note: Equity/balance chart is empty as no relevant trade history was found.")

    ax2.set_title('Account Equity and Balance')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amount ($)')
    
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend()

    ax2.grid(True, linestyle='--', alpha=0.6)

    fig.autofmt_xdate()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_filename = os.path.join(log_dir, f"{eval_run_id}_performance_chart.png")
    try:
        plt.savefig(plot_filename)
        if log_level != "none":
            print(f"Performance chart saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)


def main():
    effective_eval_config = load_default_configs_for_evaluation()

    run_settings = effective_eval_config.get("run_settings", {})
    current_log_level = run_settings.get("log_level", "normal")
    agent_type = effective_eval_config.get("agent_type", "PPO")

    eval_env_config_for_instance = effective_eval_config["environment"].copy()
    eval_env_config_for_instance["log_level"] = "none"
    eval_env_config_for_instance["custom_print_render"] = "none"

    model_load_path, _ = resolve_model_path(
        effective_config=effective_eval_config,
        log_level=current_log_level
    )

    if not model_load_path:
        print("Error: Could not determine a valid model path for evaluation. Exiting.")
        return

    eval_run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    eval_log_dir_base = run_settings.get("eval_log_dir", "logs/evaluation/")
    eval_log_dir = os.path.join(eval_log_dir_base, eval_run_id)
    os.makedirs(eval_log_dir, exist_ok=True)

    if current_log_level != "none":
        print(f"--- Evaluating Model: {model_load_path} (Log Level: {current_log_level}) ---")
        print(f"Evaluation run ID: {eval_run_id}")
        print(f"Evaluation logs and charts will be saved to: {eval_log_dir}")
        print(f"Agent Type: {agent_type}")

    with open(os.path.join(eval_log_dir, "effective_eval_config.json"), "w") as f:
        json.dump(convert_to_native_types(effective_eval_config), f, indent=4)
    if current_log_level != "none":
        print("Effective evaluation configuration saved to effective_eval_config.json")

    binance_settings = effective_eval_config["binance_settings"]
    start_date_eval = run_settings["start_date_eval"]
    end_date_eval = run_settings["end_date_eval"]
    symbol = run_settings["default_symbol"]
    interval = run_settings["historical_interval"]
    cache_dir = run_settings["historical_cache_dir"]
    kline_features_for_data_fetch = eval_env_config_for_instance["kline_price_features"]
    tick_resample_interval_ms = eval_env_config_for_instance.get("tick_resample_interval_ms")

    if current_log_level != "none": print(f"\n--- Fetching and preparing K-line evaluation data from {start_date_eval} to {end_date_eval} ---")
    eval_kline_df = pd.DataFrame()
    try:
        eval_kline_df = load_kline_data_for_range(
            symbol=symbol,
            interval=interval,
            start_date_str=start_date_eval,
            end_date_str=end_date_eval,
            cache_dir=cache_dir,
            price_features=kline_features_for_data_fetch,
            binance_settings=binance_settings,
            log_level=current_log_level
        )
        if eval_kline_df.empty:
            raise ValueError("K-line evaluation data is empty. Cannot proceed with evaluation.")
        if current_log_level != "none": print(f"K-line eval data loaded: {eval_kline_df.shape} from {eval_kline_df.index.min()} to {eval_kline_df.index.max()}")
    except Exception as e:
        print(f"ERROR: K-line evaluation data not loaded. Details: {e}")
        traceback.print_exc()
        raise

    if current_log_level != "none": print(f"\n--- Fetching and preparing Tick evaluation data from {start_date_eval} to {end_date_eval} ---")
    eval_tick_df = pd.DataFrame()
    try:
        eval_tick_df = load_tick_data_for_range(
            symbol=symbol,
            start_date_str=start_date_eval,
            end_date_str=end_date_eval,
            cache_dir=cache_dir,
            binance_settings=binance_settings,
            tick_resample_interval_ms=tick_resample_interval_ms,
            log_level=current_log_level
        )
        if eval_tick_df.empty:
            raise ValueError("Tick evaluation data is empty. Cannot proceed with evaluation.")
        if current_log_level != "none": print(f"Tick eval data loaded: {eval_tick_df.shape} from {eval_tick_df.index.min()} to {eval_tick_df.index.max()}")
    except Exception as e:
        print(f"ERROR: Tick evaluation data not loaded. Details: {e}")
        traceback.print_exc()
        raise

    env_for_model = None
    try:
        base_eval_env = SimpleTradingEnv(tick_df=eval_tick_df.copy(), kline_df_with_ta=eval_kline_df.copy(), config=eval_env_config_for_instance)
        wrapped_eval_env = FlattenAction(base_eval_env)
        monitored_single_eval_env = Monitor(wrapped_eval_env, filename=os.path.join(eval_log_dir, "eval_monitor.csv"), allow_early_resets=True)
        vec_env = DummyVecEnv([lambda: monitored_single_eval_env])
        env_for_model = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

        model_training_run_dir = os.path.dirname(os.path.dirname(model_load_path))
        vec_normalize_stats_path = os.path.join(model_training_run_dir, "vec_normalize.pkl")

        if os.path.exists(vec_normalize_stats_path):
            env_for_model = VecNormalize.load(vec_normalize_stats_path, env_for_model)
            env_for_model.training = False
            env_for_model.norm_reward = False
            if current_log_level != "none": print(f"VecNormalize statistics loaded and applied from: {vec_normalize_stats_path}")
        else:
            if current_log_level != "none": print(f"WARNING: VecNormalize stats not found at {vec_normalize_stats_path}. Evaluation observations will NOT be normalized consistently with training.")

        if current_log_level != "none": print("\nEvaluation environment created successfully and configured for normalization.")
    except Exception as e:
        print(f"Error creating evaluation environment: {e}")
        traceback.print_exc()
        raise

    try:
        model_class_map = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "A2C": A2C}
        model_class = None
        if agent_type == "RecurrentPPO":
            if SB3_CONTRIB_AVAILABLE: model_class = RecurrentPPO
            else: raise ImportError("RecurrentPPO model type requested but sb3_contrib not found.")
        else:
            model_class = model_class_map.get(agent_type)
        if not model_class: raise ValueError(f"Unknown agent type: {agent_type}.")

        model = model_class.load(model_load_path, env=env_for_model)
        if current_log_level != "none": print("Model loaded successfully for evaluation.")
    except Exception as e:
        print(f"Error loading agent model from '{model_load_path}': {e}")
        traceback.print_exc()
        if env_for_model: env_for_model.close()
        return
    
    # ... (rest of the main function remains unchanged)
    num_eval_episodes = run_settings.get('n_evaluation_episodes', 3)
    if current_log_level != "none": print(f"Starting evaluation for {num_eval_episodes} episodes...")
    all_episodes_rewards, all_episodes_profits_pct = [], []
    all_combined_trade_history = []
    current_info = {}

    for episode in range(num_eval_episodes):
        obs = env_for_model.reset()

        terminated, truncated = False, False
        episode_reward, current_episode_step = 0, 0

        actual_base_env = None
        try:
            if isinstance(env_for_model.venv, DummyVecEnv):
                monitor_env = env_for_model.venv.envs[0]
                if hasattr(monitor_env, 'env') and isinstance(monitor_env.env, FlattenAction):
                    flatten_action_env = monitor_env.env
                    if hasattr(flatten_action_env, 'env') and isinstance(flatten_action_env.env, SimpleTradingEnv):
                         actual_base_env = flatten_action_env.env
        except AttributeError:
            if current_log_level != "none":
                print("Warning: Could not reliably access underlying SimpleTradingEnv instance.")

        initial_balance_this_episode = actual_base_env.initial_balance if actual_base_env else eval_env_config_for_instance.get("initial_balance", DEFAULT_ENV_CONFIG["initial_balance"])

        if current_log_level != "none": print(f"\n--- Evaluation Episode {episode + 1}/{num_eval_episodes} (Initial Balance: {initial_balance_this_episode:.2f}) ---")

        while not (terminated or truncated):
            action_array, _states = model.predict(obs, deterministic=run_settings.get("deterministic_prediction", True))
            obs, reward, done_array, info_list = env_for_model.step(action_array)

            terminated = done_array[0]
            current_info = info_list[0]
            if "TimeLimit.truncated" in current_info:
                truncated = current_info["TimeLimit.truncated"]
            elif current_info.get("is_success") is False :
                 truncated = True
            elif terminated and not current_info.get('TimeLimit.truncated', False):
                 truncated = True
            else:
                 truncated = False


            discrete_action_for_log = int(np.round(action_array[0][0]))
            profit_target_param_for_log = action_array[0][1]

            episode_reward += reward[0]
            current_episode_step += 1

            print_freq = run_settings.get("print_step_info_freq", 50)
            action_map_for_log = actual_base_env.ACTION_MAP if actual_base_env else {0:"Hold",1:"Buy",2:"Sell"}

            if current_log_level == "normal" and current_episode_step % print_freq == 0 :
                print(f"  Step: {current_info.get('current_step')}, Action: {action_map_for_log.get(discrete_action_for_log, 'Unknown')}, "
                      f"Reward: {reward[0]:.3f}, Equity: {current_info.get('equity',0):.2f}")
            elif current_log_level == "detailed":
                print(f"  Step: {current_info.get('current_step')}, Action: {action_map_for_log.get(discrete_action_for_log, 'Unknown')}, ProfitTgt: {profit_target_param_for_log:.4f}, "
                      f"Reward: {reward[0]:.3f}, Equity: {current_info.get('equity',0):.2f}, Pos: {current_info.get('position_open')}")

            if terminated or truncated:
                break

        final_equity = current_info.get('equity', initial_balance_this_episode)
        episode_profit_pct = ((final_equity - initial_balance_this_episode) / (initial_balance_this_episode + 1e-9)) * 100

        if current_log_level != "none":
            print(f"Episode finished. Steps: {current_episode_step}. Reward: {episode_reward:.2f}. "
                  f"Final Equity: {final_equity:.2f} Profit: {episode_profit_pct:.2f}%")
        all_episodes_profits_pct.append(episode_profit_pct)
        all_episodes_rewards.append(episode_reward)

        if actual_base_env and hasattr(actual_base_env, 'trade_history'):
            all_combined_trade_history.extend(actual_base_env.trade_history)
            if current_log_level == "detailed" or (actual_base_env.trade_history and len(actual_base_env.trade_history) > 1):
                 print(f"--- Trade History Ep {episode+1} (Last 10 trades if any) ---")
                 temp_episode_trades_native_for_print = convert_to_native_types(actual_base_env.trade_history)
                 relevant_trades = [t for t in temp_episode_trades_native_for_print if t.get('type') != 'initial_balance']
                 if relevant_trades:
                    [print(f"  {json.dumps(t)}") for t in relevant_trades[-10:]]
                 else:
                    print("  No trades executed in this episode.")
            elif current_log_level == "normal":
                episode_info_from_monitor = current_info.get('episode', {})
                num_trades = current_info.get('num_trades_in_episode', 0)
                if not num_trades and episode_info_from_monitor:
                     pass
                print(f"  Total trades this episode: {num_trades}")


    trade_history_filename = f"evaluation_{eval_run_id}_trade_history.json"
    trade_history_save_path = os.path.join(eval_log_dir, trade_history_filename)

    if all_combined_trade_history:
        try:
            data_to_save = convert_to_native_types(all_combined_trade_history)
            os.makedirs(os.path.dirname(trade_history_save_path), exist_ok=True)
            with open(trade_history_save_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            if current_log_level != "none": print(f"\nFull evaluation trade history saved to: {trade_history_save_path}")
        except Exception as e:
            print(f"Error saving trade history: {e}")
            traceback.print_exc()

    if current_log_level != "none": print("\n--- Overall Evaluation Summary ---")
    num_episodes_actually_run = len(all_episodes_rewards)
    if num_episodes_actually_run > 0:
        if current_log_level != "none":
            print(f"Number of episodes run: {num_episodes_actually_run}")
            print(f"Average Reward: {float(np.mean(all_episodes_rewards)):.2f} (Std: {float(np.std(all_episodes_rewards)):.2f})")
            print(f"Median Reward: {float(np.median(all_episodes_rewards)):.2f}")
            print(f"Average Profit: {float(np.mean(all_episodes_profits_pct)):.2f}% (Std: {float(np.std(all_episodes_profits_pct)):.2f}%)")
            print(f"Median Profit: {float(np.median(all_episodes_profits_pct)):.2f}%")
            print(f"Min Profit: {float(np.min(all_episodes_profits_pct)):.2f}%, Max Profit: {float(np.max(all_episodes_profits_pct)):.2f}%")
    elif current_log_level != "none":
        print("No evaluation episodes completed or no data to summarize.")

    if current_log_level != "none": print("\n--- Generating Performance Chart ---")
    if not eval_tick_df.empty:
        plot_price_data_series = eval_tick_df['Price'].copy()
        plot_price_data_series.name = symbol

        plot_performance(all_combined_trade_history, plot_price_data_series, eval_run_id, eval_log_dir,
                        title=f"Agent Evaluation: {symbol} ({start_date_eval} to {end_date_eval})",
                        log_level=current_log_level
                        )
    elif current_log_level != "none":
        print("Skipping plot generation as eval_tick_df is empty.")


    if env_for_model:
        env_for_model.close()
        if current_log_level != "none": print("Evaluation environment closed.")
    if current_log_level != "none": print("\n--- Evaluation script finished ---")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in evaluate_agent main: {e}")
        traceback.print_exc()