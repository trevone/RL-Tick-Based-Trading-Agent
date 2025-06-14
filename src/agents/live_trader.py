# src/agents/live_trader.py

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import traceback
import queue

# --- Binance API and WebSocket ---
try:
    from binance.client import Client
    from binance.websockets import BinanceSocketManager
    BINANCE_CLIENT_AVAILABLE = True
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False
    print("CRITICAL ERROR: python-binance library not found. Live trading will not function. "
          "Please install with 'pip install python-binance websocket-client'.")

# Import Stable Baselines3 algorithms dynamically
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    print("WARNING: sb3_contrib (for RecurrentPPO) not found. RecurrentPPO will not be available for live trading.")

# --- UPDATED IMPORTS ---
from src.environments.custom_wrappers import FlattenAction
from src.data.config_loader import load_config, merge_configs, convert_to_native_types
from src.data.data_loader import load_tick_data_for_range
from src.data.binance_client import fetch_and_cache_kline_data
from src.utils import resolve_model_path
from src.environments.env_loader import load_environments
# --- END UPDATED IMPORTS ---

def load_default_configs_for_live_trading(config_dir="configs/defaults") -> dict:
    """Loads default configurations from the specified directory for live trading."""
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "environment.yaml"),
        os.path.join(config_dir, "binance_settings.yaml"),
        os.path.join(config_dir, "hash_keys.yaml"),
        os.path.join(config_dir, "ppo_params.yaml"),
        os.path.join(config_dir, "sac_params.yaml"),
        os.path.join(config_dir, "ddpg_params.yaml"),
        os.path.join(config_dir, "a2c_params.yaml"),
        os.path.join(config_dir, "recurrent_ppo_params.yaml"),
    ]
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)

class LiveTrader:
    def __init__(self, config_override: dict = None):
        print("--- Initializing Live Trading Agent ---")

        if not BINANCE_CLIENT_AVAILABLE:
            raise ImportError("Binance API Client not available. Cannot run live trader.")

        # 1. Load and Merge Configurations
        self.effective_config = load_default_configs_for_live_trading()
        if config_override:
            self.effective_config = merge_configs(self.effective_config, config_override)

        self.run_settings = self.effective_config["run_settings"]
        self.env_config = self.effective_config["environment"]
        self.binance_settings = self.effective_config["binance_settings"]
        self.live_trader_settings = self.effective_config.get("live_trader_settings", {})
        self.agent_type = self.effective_config.get("agent_type", "PPO")
        
        # --- NEW: Get env_type and load available environments ---
        self.env_type = self.run_settings.get("env_type", "simple")
        available_envs = load_environments()
        self.env_class = available_envs.get(self.env_type)
        if self.env_class is None:
            raise ValueError(f"Environment type '{self.env_type}' not found. Available: {list(available_envs.keys())}")
        print(f">>> Live Trader using Environment: {self.env_class.__name__} (type: '{self.env_type}') <<<")
        # --- END NEW ---

        self.log_level = self.run_settings.get("log_level", "normal")
        self.env_config["log_level"] = "none"
        self.env_config["custom_print_render"] = "none"

        self.live_run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.live_log_dir = os.path.join(self.run_settings.get("live_log_dir", "logs/live_trading/"), self.live_run_id)
        os.makedirs(self.live_log_dir, exist_ok=True)

        with open(os.path.join(self.live_log_dir, "effective_live_config.json"), "w") as f:
            json.dump(convert_to_native_types(self.effective_config), f, indent=4)
        print(f"Live trading logs will be saved to: {self.live_log_dir}")
        print(f"Agent Type: {self.agent_type}")

        # 2. Resolve and Load Trained Model
        self.model_load_path, _ = resolve_model_path(self.effective_config, log_level=self.log_level)
        if not self.model_load_path:
            raise ValueError("Could not determine a valid model path for live trading. Exiting.")

        model_training_run_dir = os.path.dirname(os.path.dirname(self.model_load_path))
        self.vec_normalize_stats_path = os.path.join(model_training_run_dir, "vec_normalize.pkl")
        if not os.path.exists(self.vec_normalize_stats_path):
            print(f"WARNING: VecNormalize stats not found at {self.vec_normalize_stats_path}. Live observations will NOT be normalized consistently with training.")
            self.vec_normalize = None
        else:
            # Use the dynamically loaded env_class for the dummy env
            dummy_base_env = self.env_class(pd.DataFrame(), pd.DataFrame(), self.env_config)
            dummy_wrapped_env = FlattenAction(dummy_base_env)
            dummy_vec_env = DummyVecEnv([lambda: dummy_wrapped_env])
            self.vec_normalize = VecNormalize.load(self.vec_normalize_stats_path, dummy_vec_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            print(f"VecNormalize statistics loaded from: {self.vec_normalize_stats_path}")

        # --- Binance Client and WebSocket ---
        self.client = Client(
            self.binance_settings.get("api_key", os.environ.get('BINANCE_API_KEY')),
            self.binance_settings.get("api_secret", os.environ.get('BINANCE_API_SECRET')),
            tld=self.binance_settings.get("tld", "us"),
            testnet=self.binance_settings.get("testnet", True)
        )
        if self.binance_settings.get("testnet", True):
            self.client.API_URL = 'https://testnet.binance.vision/api'
            print("WARNING: Using Binance Testnet for live trading.")
        else:
            print("CAUTION: Using Binance Live/Production environment.")

        self.bm = BinanceSocketManager(self.client)
        self.tick_data_queue = queue.Queue()
        self.conn_key = self.bm.start_aggtrade_socket(self.run_settings["default_symbol"].lower(), self._process_tick_message)
        self.bm.start()
        print("Binance WebSocket started for aggTrades.")

        self.env = None
        self.current_kline_data = pd.DataFrame()
        self.historical_ticks_for_obs = pd.DataFrame()
        self._initialize_env_data()

        # 4. Load Model and Bind to Environment (VecNormalize handles obs normalization)
        try:
            # Dynamically load the correct model class
            model_class = None
            if self.agent_type == "PPO": model_class = PPO
            elif self.agent_type == "SAC": model_class = SAC
            elif self.agent_type == "DDPG": model_class = DDPG
            elif self.agent_type == "A2C": model_class = A2C
            elif self.agent_type == "RecurrentPPO":
                if SB3_CONTRIB_AVAILABLE: model_class = RecurrentPPO
                else: raise ImportError("RecurrentPPO model type requested but sb3_contrib is not installed.")
            else: raise ValueError(f"Unknown agent type '{self.agent_type}'. Supported: PPO, SAC, DDPG, A2C, RecurrentPPO.")

            self.model = model_class.load(self.model_load_path, env=None) # Load model first, then set env

            # Set the VecNormalize environment to the model after loading its stats if available
            if self.vec_normalize:
                base_env = self.env_class(pd.DataFrame(), pd.DataFrame(), self.env_config)
                wrapped_env = FlattenAction(base_env)
                vec_wrapped_env = DummyVecEnv([lambda: wrapped_env])
                
                self.env = VecNormalize(
                    vec_wrapped_env,
                    norm_obs=True, norm_reward=False, clip_obs=10.0
                )
                self.env = VecNormalize.load(self.vec_normalize_stats_path, self.env)
                self.env.training = False
                self.env.norm_reward = False
                self.model.set_env(self.env)
                print("Model env set with loaded VecNormalize.")
            else:
                base_env = self.env_class(pd.DataFrame(), pd.DataFrame(), self.env_config)
                self.env = FlattenAction(base_env)
                self.model.set_env(self.env)
                print("Model env set (no VecNormalize applied).")

            print(f"Model {self.agent_type} loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load model or set environment: {e}")
            traceback.print_exc()
            self._close_connections()
            raise

        # Initial Observation
        self.current_obs = None
        self.env_reset_called = False
        self.last_action_timestamp = datetime.now(timezone.utc)
        self.last_decision_price = 0.0
        self.episode_counter = 0


    def _initialize_env_data(self):
        """
        Fetches initial historical k-line data and initializes tick history for the environment.
        This data forms the initial observation space.
        """
        symbol = self.run_settings["default_symbol"]
        interval = self.run_settings["historical_interval"]
        cache_dir = self.run_settings["historical_cache_dir"]
        kline_window_size = self.env_config["kline_window_size"]
        tick_feature_window_size = self.env_config["tick_feature_window_size"]
        kline_features = self.env_config["kline_price_features"]
        tick_resample_interval_ms = self.env_config.get("tick_resample_interval_ms")
        
        # Fetch initial K-line data for context
        end_time_kline = datetime.now(timezone.utc)
        start_time_kline = end_time_kline - timedelta(hours=kline_window_size * 2)
        
        print(f"\nFetching initial K-line data ({symbol}, {interval}) for observation context...")
        self.current_kline_data = fetch_and_cache_kline_data(
            symbol=symbol,
            interval=interval,
            start_date_str=start_time_kline.strftime("%Y-%m-%d %H:%M:%S"),
            end_date_str=end_time_kline.strftime("%Y-%m-%d %H:%M:%S"),
            cache_dir=cache_dir,
            price_features_to_add=kline_features,
            api_key=self.binance_settings.get("api_key"),
            api_secret=self.binance_settings.get("api_secret"),
            testnet=self.binance_settings.get("testnet", True),
            log_level=self.log_level,
            api_request_delay_seconds=self.binance_settings.get("api_request_delay_seconds", 0.2)
        )
        if self.current_kline_data.empty:
            print("WARNING: Initial K-line data is empty. Environment might not initialize correctly.")

        # Initialize historical ticks for observation
        end_time_tick = datetime.now(timezone.utc)
        start_time_tick = end_time_tick - timedelta(minutes=10)
        
        print(f"\nFetching initial Tick data for observation context ({symbol}, {start_time_tick} to {end_time_tick})...")
        try:
            self.historical_ticks_for_obs = load_tick_data_for_range(
                symbol=symbol,
                start_date_str=start_time_tick.strftime("%Y-%m-%d %H:%M:%S"),
                end_date_str=end_time_tick.strftime("%Y-%m-%d %H:%M:%S"),
                cache_dir=cache_dir,
                binance_settings=self.binance_settings,
                tick_resample_interval_ms=tick_resample_interval_ms
            )
            if len(self.historical_ticks_for_obs) < tick_feature_window_size:
                print(f"WARNING: Fetched tick data ({len(self.historical_ticks_for_obs)} ticks) is less than tick_feature_window_size ({tick_feature_window_size}). Padding with dummy data.")
                if not self.historical_ticks_for_obs.empty:
                    last_price = self.historical_ticks_for_obs['Price'].iloc[-1]
                    last_qty = self.historical_ticks_for_obs['Quantity'].iloc[-1]
                    last_ism = self.historical_ticks_for_obs['IsBuyerMaker'].iloc[-1]
                else:
                    last_price = self.current_kline_data['Close'].iloc[-1] if not self.current_kline_data.empty else 0.0
                    last_qty = 1.0
                    last_ism = False

                dummy_rows_needed = tick_feature_window_size - len(self.historical_ticks_for_obs)
                dummy_timestamps = [end_time_tick - timedelta(milliseconds=i) for i in range(dummy_rows_needed-1, -1, -1)]
                dummy_padding_df = pd.DataFrame(
                    [{'Price': last_price, 'Quantity': last_qty, 'IsBuyerMaker': last_ism}] * dummy_rows_needed,
                    index=pd.to_datetime(dummy_timestamps, utc=True)
                )
                self.historical_ticks_for_obs = pd.concat([dummy_padding_df, self.historical_ticks_for_obs]).sort_index().tail(tick_feature_window_size)
            else:
                self.historical_ticks_for_obs = self.historical_ticks_for_obs.tail(tick_feature_window_size)
            
            for col in self.env_config["tick_features_to_use"]:
                if col not in self.historical_ticks_for_obs.columns:
                    self.historical_ticks_for_obs[col] = 0.0

        except Exception as e:
            print(f"ERROR: Failed to load initial historical tick data: {e}. Initializing with full dummy data.")
            traceback.print_exc()
            dummy_price = self.current_kline_data['Close'].iloc[-1] if not self.current_kline_data.empty else 0.0
            dummy_qty = 1.0
            dummy_ism = False
            dummy_ticks_fallback = pd.DataFrame(
                [{'Price': dummy_price, 'Quantity': dummy_qty, 'IsBuyerMaker': dummy_ism}] * tick_feature_window_size,
                index=pd.to_datetime([datetime.now(timezone.utc) - timedelta(milliseconds=i) for i in range(tick_feature_window_size-1, -1, -1)], utc=True)
            )
            self.historical_ticks_for_obs = dummy_ticks_fallback


        print(f"Initial environment data prepared: K-line data shape {self.current_kline_data.shape}, Tick history shape {self.historical_ticks_for_obs.shape}")


    def _process_tick_message(self, msg):
        """Callback function for Binance WebSocket aggTrade stream."""
        if msg['e'] == 'error':
            print(f"WebSocket Error: {msg['m']}")
            return
        
        if msg['e'] == 'aggTrade':
            timestamp = pd.to_datetime(msg['T'], unit='ms', utc=True)
            price = float(msg['p'])
            quantity = float(msg['q'])
            is_buyer_maker = msg['m']
            
            self.tick_data_queue.put({'Timestamp': timestamp, 'Price': price, 'Quantity': quantity, 'IsBuyerMaker': is_buyer_maker})
            if self.log_level == "detailed":
                print(f"Tick received: Time={timestamp}, Price={price:.4f}, Qty={quantity:.4f}, Maker={is_buyer_maker}")

    def _update_kline_data_live(self):
        """
        Fetches the latest k-line data to keep the environment's k-line context current.
        """
        symbol = self.run_settings["default_symbol"]
        interval = self.run_settings["historical_interval"]
        cache_dir = self.run_settings["historical_cache_dir"]
        kline_window_size = self.env_config["kline_window_size"]

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=kline_window_size * 2)
        
        try:
            latest_klines = fetch_and_cache_kline_data(
                symbol=symbol,
                interval=interval,
                start_date_str=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_date_str=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                cache_dir=cache_dir,
                price_features_to_add=self.env_config["kline_price_features"],
                api_key=self.binance_settings.get("api_key"),
                api_secret=self.binance_settings.get("api_secret"),
                testnet=self.binance_settings.get("testnet", True),
                log_level="none"
            )
            if not latest_klines.empty:
                combined_df = pd.concat([self.current_kline_data, latest_klines]).drop_duplicates(subset=latest_klines.columns, keep='last')
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index().tail(kline_window_size + 10)
                self.current_kline_data = combined_df
                if self.log_level == "detailed":
                    print(f"K-line data updated. New shape: {self.current_kline_data.shape}. Last candle: {self.current_kline_data.index[-1]}")
            else:
                if self.log_level != "none": print("WARNING: No new k-lines fetched. K-line data remains unchanged.")
        except Exception as e:
            if self.log_level != "none": print(f"ERROR: Failed to update K-line data: {e}")
            traceback.print_exc()

    def _execute_trade(self, action: int, profit_target_param: float, current_price: float):
        """
        Executes a trade (BUY/SELL) via Binance API.
        """
        trade_success = False
        order_response = None
        
        if isinstance(self.env, VecNormalize):
            base_env_instance = self.env.venv.envs[0].env.env
        else:
            base_env_instance = self.env.env

        if not base_env_instance.position_open and action == 1: # BUY
            cash_for_trade = base_env_instance.current_balance * base_env_instance.base_trade_amount_ratio
            units_to_buy_precise = cash_for_trade / (current_price + 1e-9) 
            units_to_buy = max(base_env_instance.min_tradeable_unit, np.floor(units_to_buy_precise / base_env_instance.min_tradeable_unit) * base_env_instance.min_tradeable_unit)
            
            if units_to_buy > 0:
                print(f"LIVE: Attempting BUY order for {units_to_buy:.6f} {self.run_settings['default_symbol'][:-4]} at {current_price:.4f}")
                try:
                    order_response = self.client.order_limit_buy(
                        symbol=self.run_settings['default_symbol'],
                        quantity=f"{units_to_buy:.6f}",
                        price=f"{current_price:.4f}"
                    )
                    filled_qty = float(order_response.get('executedQty', units_to_buy))
                    filled_price = float(order_response.get('fills', [{}])[0].get('price', current_price)) if order_response.get('fills') else current_price
                    commission_cost = sum(float(f['commission']) for f in order_response.get('fills', []) if f.get('commissionAsset') == 'BNB')
                    
                    cost = filled_qty * filled_price + commission_cost
                    base_env_instance.current_balance -= cost
                    base_env_instance.position_open = True
                    base_env_instance.entry_price = filled_price
                    base_env_instance.position_volume = filled_qty
                    base_env_instance.current_desired_profit_target = profit_target_param
                    trade_success = True
                    print(f"LIVE: BUY order successful! Filled {filled_qty:.6f} at {filled_price:.4f}. Balance: {base_env_instance.current_balance:.2f}")
                    self._log_trade("BUY", filled_price, filled_qty, commission_cost, base_env_instance.current_balance, base_env_instance.current_balance + (filled_qty * filled_price), profit_target_param)
                except Exception as e:
                    print(f"LIVE: ERROR executing BUY order: {e}")
                    traceback.print_exc()
            else:
                print("LIVE: Not enough units to buy or balance for trade.")
        
        elif base_env_instance.position_open and action == 2: # SELL
            print(f"LIVE: Attempting SELL order for {base_env_instance.position_volume:.6f} {self.run_settings['default_symbol'][:-4]} at {current_price:.4f}")
            try:
                order_response = self.client.order_limit_sell(
                    symbol=self.run_settings['default_symbol'],
                    quantity=f"{base_env_instance.position_volume:.6f}",
                    price=f"{current_price:.4f}"
                )
                filled_qty = float(order_response.get('executedQty', base_env_instance.position_volume))
                filled_price = float(order_response.get('fills', [{}])[0].get('price', current_price)) if order_response.get('fills') else current_price
                commission_cost = sum(float(f['commission']) for f in order_response.get('fills', []) if f.get('commissionAsset') == 'BNB')
                
                revenue = filled_qty * filled_price - commission_cost
                pnl = revenue - (base_env_instance.entry_price * base_env_instance.position_volume)
                base_env_instance.current_balance += revenue
                
                print(f"LIVE: SELL order successful! Filled {filled_qty:.6f} at {filled_price:.4f}. PnL: {pnl:.2f}. Balance: {base_env_instance.current_balance:.2f}")
                self._log_trade("SELL", filled_price, filled_qty, commission_cost, base_env_instance.current_balance, base_env_instance.current_balance, base_env_instance.current_desired_profit_target, pnl)
                
                base_env_instance.position_open = False
                base_env_instance.entry_price = 0.0
                base_env_instance.position_volume = 0.0
                base_env_instance.current_desired_profit_target = 0.0
                trade_success = True
            except Exception as e:
                print(f"LIVE: ERROR executing SELL order: {e}")
                traceback.print_exc()
        elif action == 0:
            pass
        
        return trade_success

    def _log_trade(self, trade_type: str, price: float, volume: float, commission: float, balance: float, equity: float, profit_target: float = 0.0, pnl: float = 0.0):
        """Logs trade details to a file."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": trade_type,
            "price": price,
            "volume": volume,
            "commission": commission,
            "balance": balance,
            "equity": equity,
            "profit_target_set": profit_target,
            "pnl": pnl,
            "model_path": self.model_load_path
        }
        log_file_path = os.path.join(self.live_log_dir, "live_trades.jsonl")
        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"LIVE_LOG: Trade recorded: {log_entry}")

    def run_live_trading(self):
        """Main loop for live trading."""
        print("\n--- Starting Live Trading Loop ---")
        self.running = True
        
        tick_process_freq_seconds = self.live_trader_settings.get("tick_process_freq_seconds", 1.0)
        kline_update_freq_minutes = self.live_trader_settings.get("kline_update_freq_minutes", 10)
        action_cooldown_seconds = self.live_trader_settings.get("action_cooldown_seconds", 5)

        last_kline_update_time = datetime.now(timezone.utc)
        
        try:
            if isinstance(self.env, VecNormalize):
                base_env_instance = self.env.venv.envs[0].env.env
            else:
                base_env_instance = self.env.env

            base_env_instance.tick_df = self.historical_ticks_for_obs.copy()
            base_env_instance.kline_df_with_ta = self.current_kline_data.copy()
            
            initial_obs, initial_info = self.env.reset()
            self.current_obs = initial_obs
            self.env_reset_called = True
            
            initial_balance_from_info = initial_info[0].get('current_balance')
            initial_equity_from_info = initial_info[0].get('equity')
            
            print(f"Live environment reset: Initial Balance: {initial_balance_from_info:.2f}, Initial Equity: {initial_equity_from_info:.2f}")

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to reset live environment: {e}")
            traceback.print_exc()
            self._close_connections()
            return
            
        while self.running:
            try:
                processed_ticks_count = 0
                while not self.tick_data_queue.empty():
                    tick = self.tick_data_queue.get(timeout=0.1)
                    self.historical_ticks_for_obs = pd.concat([self.historical_ticks_for_obs, pd.DataFrame([tick]).set_index('Timestamp')]).tail(self.env_config["tick_feature_window_size"])
                    processed_ticks_count += 1
                
                if processed_ticks_count > 0:
                    self.last_decision_price = self.historical_ticks_for_obs['Price'].iloc[-1]
                    if self.log_level == "detailed":
                        print(f"Processed {processed_ticks_count} new ticks. Last tick price: {self.last_decision_price:.4f}")
                else:
                    if self.log_level == "detailed":
                        print("No new ticks to process. Waiting...")
                    if not self.historical_ticks_for_obs.empty:
                        self.last_decision_price = self.historical_ticks_for_obs['Price'].iloc[-1]
                    else:
                        time.sleep(1)
                        continue
                
                if (datetime.now(timezone.utc) - last_kline_update_time).total_seconds() >= kline_update_freq_minutes * 60:
                    self._update_kline_data_live()
                    last_kline_update_time = datetime.now(timezone.utc)
                
                if isinstance(self.env, VecNormalize):
                    base_env_instance = self.env.venv.envs[0].env.env
                else:
                    base_env_instance = self.env.env
                
                base_env_instance.tick_df = self.historical_ticks_for_obs.copy()
                base_env_instance.kline_df_with_ta = self.current_kline_data.copy()

                if not base_env_instance.tick_df.empty:
                    base_env_instance.current_step = len(base_env_instance.tick_df) - 1
                else:
                    if self.log_level != "none": print("WARNING: Tick data for base env is empty. Cannot update current_step.")
                    time.sleep(tick_process_freq_seconds)
                    continue

                raw_obs_from_env = base_env_instance._get_observation()
                
                if self.vec_normalize:
                    if raw_obs_from_env.ndim == 1:
                        raw_obs_from_env = raw_obs_from_env[np.newaxis, :]
                    self.current_obs = self.vec_normalize.normalize_obs(raw_obs_from_env).squeeze(0)
                else:
                    self.current_obs = raw_obs_from_env

                if self.current_obs is None or not len(self.current_obs) == self.env.observation_space.shape[0]:
                    print("WARNING: Observation is not ready or has incorrect shape. Skipping action decision.")
                    time.sleep(tick_process_freq_seconds)
                    continue

                if (datetime.now(timezone.utc) - self.last_action_timestamp).total_seconds() < action_cooldown_seconds:
                    if self.log_level == "detailed":
                        print(f"Action cooldown. Next decision in {action_cooldown_seconds - (datetime.now(timezone.utc) - self.last_action_timestamp).total_seconds():.2f}s")
                    time.sleep(tick_process_freq_seconds)
                    continue
                
                action_array, _states = self.model.predict(self.current_obs, deterministic=self.live_trader_settings.get("deterministic_prediction", True))
                
                discrete_action_for_trade = int(np.round(action_array[0]))
                profit_target_for_trade = float(np.clip(action_array[1], base_env_instance.config['min_profit_target_low'], base_env_instance.config['min_profit_target_high']))

                print(f"LIVE: Agent decision: Action: {base_env_instance.ACTION_MAP.get(discrete_action_for_trade)}, Profit Target: {profit_target_for_trade:.4f}, Price: {self.last_decision_price:.4f}")
                
                if discrete_action_for_trade != 0:
                    self._execute_trade(discrete_action_for_trade, profit_target_for_trade, self.last_decision_price)
                    self.last_action_timestamp = datetime.now(timezone.utc)
                else:
                    if self.log_level == "detailed":
                        print("LIVE: Agent decided to HOLD.")


            except queue.Empty:
                if self.log_level == "detailed":
                    print("Tick queue empty, waiting for new data...")
            except KeyboardInterrupt:
                print("\nLive trading interrupted by user.")
                self.running = False
            except Exception as e:
                print(f"CRITICAL ERROR in live trading loop: {e}")
                traceback.print_exc()
                self.running = False
            finally:
                time.sleep(tick_process_freq_seconds)

    def _close_connections(self):
        """Closes WebSocket and cleans up."""
        if hasattr(self, 'bm') and self.bm:
            self.bm.stop_socket(self.conn_key)
            self.bm.close()
            print("Binance WebSocket closed.")
        if hasattr(self, 'env') and self.env:
            self.env.close()
            print("Live environment closed.")
        if hasattr(self, 'client') and self.client:
            pass 
        print("Live trader connections closed.")

    def stop(self):
        self.running = False
        print("Live trading stopping...")


if __name__ == "__main__":
    live_trader = None
    try:
        live_trader = LiveTrader()
        live_trader.run_live_trading()
    except Exception as e:
        print(f"Failed to start live trader: {e}")
        traceback.print_exc()
    finally:
        if live_trader:
            live_trader.stop()
            live_trader._close_connections()