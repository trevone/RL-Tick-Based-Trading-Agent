# src/environments/base_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import re

# We need the feature calculation function to run it dynamically
from src.data.feature_engineer import calculate_technical_indicators, TALIB_AVAILABLE

# --- A complete dictionary of all environment parameters ---
DEFAULT_ENV_CONFIG = {
    "kline_window_size": 20, "tick_feature_window_size": 50,
    "kline_price_features": ["Open", "High", "Low", "Close", "Volume"],
    "tick_features_to_use": ["Price", "Quantity"],
    "initial_balance": 10000.0, "commission_pct": 0.001,
    "base_trade_amount_ratio": 0.02, "catastrophic_loss_threshold_pct": 0.3,
    "penalty_catastrophic_loss": -100.0,
    # --- Default reward/penalty values for the base environment ---
    "reward_correct_buy": 0.5,
    "reward_correct_sell_profit": 1.0,
    "penalty_incorrect_action": -1.0,
    "penalty_hold_flat": -0.01,
    "penalty_hold_losing": -0.05,
    "reward_hold_profit": 0.02,
}

# --- HELPER FUNCTION TO PARSE TA PERIODS ---
def _get_max_ta_period(feature_names: list) -> int:
    max_period = 0
    for name in feature_names:
        parts = re.split('_|-', name)
        for part in parts:
            if part.isdigit():
                max_period = max(max_period, int(part))
    # Provide a reasonable minimum for TAs without explicit periods (e.g., MACD, ADX)
    if any(f in ['MACD', 'ADX', 'STOCH_K', 'ATR'] for f in feature_names):
        max_period = max(max_period, 26) # 26 is from MACD's slow period
    return max(1, max_period) # Return at least 1

class SimpleTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    ACTION_MAP = {0: "Hold", 1: "Buy", 2: "Sell"}

    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        super().__init__()
        self.config = {**DEFAULT_ENV_CONFIG, **(config if config else {})}

        if tick_df.empty: raise ValueError("tick_df must be non-empty.")
        self.tick_df = tick_df.copy()
        
        # We now need both the raw kline data and the one with pre-calculated TAs
        self.raw_kline_df = kline_df_with_ta[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        self.kline_df_with_ta = kline_df_with_ta.copy()
        
        if not self.kline_df_with_ta.empty:
            if not isinstance(self.kline_df_with_ta.index, pd.DatetimeIndex): self.kline_df_with_ta.index = pd.to_datetime(self.kline_df_with_ta.index)
            if not self.kline_df_with_ta.index.is_monotonic_increasing: self.kline_df_with_ta.sort_index(inplace=True)

        self.initial_balance = float(self.config["initial_balance"])
        self.commission_pct = float(self.config["commission_pct"])
        self.base_trade_amount_ratio = float(self.config["base_trade_amount_ratio"])
        self.catastrophic_loss_limit = self.initial_balance * (1.0 - float(self.config["catastrophic_loss_threshold_pct"]))
        
        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)))
        
        self.tick_feature_window_size = int(self.config["tick_feature_window_size"])
        self.kline_window_size = int(self.config["kline_window_size"])
        obs_shape_dim = (self.tick_feature_window_size * len(self.config["tick_features_to_use"])) + \
                        (self.kline_window_size * len(self.config["kline_price_features"])) + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape_dim,), dtype=np.float32)

        self._prepare_data()
        self.reset()

    def _prepare_data(self):
        if not isinstance(self.tick_df.index, pd.DatetimeIndex): self.tick_df.index = pd.to_datetime(self.tick_df.index)
        self.tick_df.sort_index(inplace=True)
        self.tick_price_data = {name: self.tick_df[name].values.astype(np.float32) for name in self.config["tick_features_to_use"]}
        self.decision_prices = self.tick_price_data['Price']
        
        # Get the max lookback period needed for the live TA calculation
        self.max_ta_period = _get_max_ta_period(self.config["kline_price_features"])

        self.kline_feature_arrays = {name: self.kline_df_with_ta[name].values.astype(np.float32) for name in self.config["kline_price_features"] if name in self.kline_df_with_ta}
        
        self.end_step = len(self.tick_df) - 1
        min_start_step = self.tick_feature_window_size - 1

        if not self.kline_df_with_ta.empty and len(self.kline_df_with_ta) >= self.kline_window_size:
            first_valid_kline_timestamp = self.kline_df_with_ta.index[self.kline_window_size - 1]
            tick_indexer = self.tick_df.index.get_indexer([first_valid_kline_timestamp], method='bfill')
            
            if tick_indexer[0] != -1:
                kline_start_buffer = tick_indexer[0]
                self.start_step = max(min_start_step, kline_start_buffer)
            else:
                self.start_step = self.end_step
        else:
            self.start_step = self.end_step
        
        if self.start_step >= self.end_step:
            raise ValueError("Not enough data to create a full observation window. Check data alignment and length.")

    def _get_observation(self) -> np.ndarray:
        safe_step = min(self.current_step, self.end_step)
        current_tick_price = self.decision_prices[safe_step]

        tick_start_idx = max(0, safe_step - self.tick_feature_window_size + 1)
        tick_features_list = []
        for name in self.config["tick_features_to_use"]:
            arr = self.tick_price_data[name][tick_start_idx:safe_step + 1]
            padding = self.tick_feature_window_size - len(arr)
            if padding > 0: arr = np.concatenate((np.full(padding, arr[0] if len(arr) > 0 else 0), arr))
            tick_features_list.append(arr)
        tick_features = np.concatenate(tick_features_list)

        kline_features = np.zeros(self.kline_window_size * len(self.config["kline_price_features"]), dtype=np.float32)
        if not self.kline_df_with_ta.empty:
            kline_indexer = self.kline_df_with_ta.index.get_indexer([self.tick_df.index[safe_step]], method='ffill')
            if kline_indexer[0] != -1:
                kline_idx = kline_indexer[0]
                kline_start_idx = kline_idx - self.kline_window_size + 1
                
                window_features = {name: arr[kline_start_idx:kline_idx + 1] for name, arr in self.kline_feature_arrays.items()}
                
                if TALIB_AVAILABLE and self.max_ta_period > 1:
                    recalc_start_idx = max(0, kline_idx - self.max_ta_period + 1)
                    live_recalc_window_df = self.raw_kline_df.iloc[recalc_start_idx:kline_idx + 1].copy()
                    live_recalc_window_df.iloc[-1, live_recalc_window_df.columns.get_loc('Close')] = current_tick_price

                    live_ta_values_df = calculate_technical_indicators(
                        live_recalc_window_df, self.config["kline_price_features"]
                    )
                    
                    latest_live_ta_series = live_ta_values_df.iloc[-1]
                    
                    for name in self.config["kline_price_features"]:
                        if name in latest_live_ta_series and name in window_features:
                            window_features[name][-1] = latest_live_ta_series[name]

                kline_features_list = []
                for name in self.config["kline_price_features"]:
                    arr = window_features.get(name, np.zeros(self.kline_window_size))
                    padding = self.kline_window_size - len(arr)
                    if padding > 0: arr = np.concatenate((np.full(padding, arr[0] if len(arr) > 0 else 0), arr))
                    kline_features_list.append(arr)
                kline_features = np.concatenate(kline_features_list)

        portfolio_state = np.array([1.0 if self.position_open else 0.0, (self.entry_price / current_tick_price - 1) if self.position_open and current_tick_price > 1e-9 else 0.0, ((current_tick_price - self.entry_price) * self.position_volume) / self.initial_balance if self.position_open else 0.0], dtype=np.float32)
        
        return np.concatenate([tick_features, kline_features, portfolio_state])

    def _get_info(self) -> dict:
        safe_step = min(self.current_step, self.end_step)
        price = self.decision_prices[safe_step]
        equity = self.current_balance + (self.position_volume * price if self.position_open else 0)
        timestamp = self.tick_df.index[safe_step]
        return {"current_step": self.current_step, "timestamp": timestamp, "equity": equity, "position_open": self.position_open, "current_tick_price": price}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_balance, self.position_open, self.entry_price, self.position_volume = self.initial_balance, False, 0.0, 0.0
        self.trade_start_step = 0
        self.last_price = 0.0
        self.current_step = self.start_step
        return self._get_observation(), self._get_info()
    
    def _calculate_reward(self, discrete_action, price_for_action):
        if discrete_action == 1:
            return self.config["reward_correct_buy"] if not self.position_open else self.config["penalty_incorrect_action"]
        elif discrete_action == 2:
            if not self.position_open:
                return self.config["penalty_incorrect_action"]
            else:
                pnl = (price_for_action - self.entry_price) * self.position_volume
                return self.config["reward_correct_sell_profit"] + pnl
        else:
            if not self.position_open:
                return self.config["penalty_hold_flat"]
            else:
                is_losing = price_for_action < self.entry_price
                return self.config["penalty_hold_losing"] if is_losing else self.config["reward_hold_profit"]

    def step(self, action_tuple):
        discrete_action, _ = action_tuple
        price = self.decision_prices[self.current_step]
        terminated, truncated = False, False

        reward = self._calculate_reward(discrete_action, price)

        if discrete_action == 1 and not self.position_open:
            cost = (self.initial_balance * self.base_trade_amount_ratio) * (1 + self.commission_pct)
            if self.current_balance >= cost:
                self.position_volume = (self.initial_balance * self.base_trade_amount_ratio) / price
                self.current_balance -= cost
                self.position_open, self.entry_price = True, price
                self.trade_start_step = self.current_step
        
        elif discrete_action == 2 and self.position_open:
            revenue = self.position_volume * price * (1 - self.commission_pct)
            self.current_balance += revenue
            self.position_open, self.position_volume, self.entry_price = False, 0.0, 0.0
            self.trade_start_step = 0
       
        self.current_step += 1
        current_equity = self.current_balance + (self.position_volume * price if self.position_open else 0)

        if current_equity < self.catastrophic_loss_limit:
            terminated = True
            reward += self.config["penalty_catastrophic_loss"]

        if self.current_step > self.end_step:
            truncated = True
        
        if (terminated or truncated) and self.position_open:
            self.current_balance += self.position_volume * price * (1 - self.commission_pct)
            self.position_open = False

        self.last_price = price

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def close(self): pass