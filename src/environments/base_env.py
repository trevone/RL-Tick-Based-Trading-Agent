# src/environments/base_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import os
import traceback

# This is now the default configuration, incorporating the profit target logic.
DEFAULT_ENV_CONFIG = {
    # Data Window Sizes
    "kline_window_size": 20,
    "tick_feature_window_size": 50,
    
    # Feature Selection
    "kline_price_features": ["Open", "High", "Low", "Close", "Volume"],
    "tick_features_to_use": ["Price", "Quantity"],

    # Trading Parameters
    "initial_balance": 10000.0,
    "commission_pct": 0.001,
    "base_trade_amount_ratio": 0.02, # Using the safer 2% as a default
    "min_tradeable_unit": 1e-6,
    "catastrophic_loss_threshold_pct": 0.3,

    # Observation Clipping
    "obs_clip_low": -5.0,
    "obs_clip_high": 5.0,

    # Action Space Parameters
    "min_profit_target_low": 0.001,
    "min_profit_target_high": 0.01,

    # --- Profit Target Reward Settings ---
    "profit_target_pct": 0.002,          # Desired profit target (0.2%) to guide rewards
    "reward_factor_above_target": 50.0,  # Multiplier for reward when PnL is ABOVE the target
    "penalty_factor_below_target": 100.0, # Multiplier for penalty when PnL is IN PROFIT but BELOW the target
    "penalty_loss_trade_factor": 1.0,     # Multiplier for the PnL of a simple losing trade

    # --- Holding Penalty Settings ---
    "penalty_hold_losing_position": -0.0005,
    "penalty_hold_flat_position": -0.0001,
    
    # System Settings
    "custom_print_render": "none",
    "log_level": "normal",
}

class SimpleTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    ACTION_MAP = {0: "Hold", 1: "Buy", 2: "Sell"}

    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        super().__init__()
        self.config = {**DEFAULT_ENV_CONFIG, **(config if config else {})}

        if not isinstance(tick_df, pd.DataFrame) or tick_df.empty or not isinstance(kline_df_with_ta, pd.DataFrame):
            # Allow empty kline_df for simpler tests, but tick_df is essential
            if kline_df_with_ta.empty: print("Warning: kline_df_with_ta is empty.")
            else: raise ValueError("tick_df must be a non-empty pandas DataFrame.")

        self.tick_df = tick_df.copy()
        self.kline_df_with_ta = kline_df_with_ta.copy()

        # Basic setup from original file...
        if not isinstance(self.kline_df_with_ta.index, pd.DatetimeIndex) and not self.kline_df_with_ta.empty:
            self.kline_df_with_ta.index = pd.to_datetime(self.kline_df_with_ta.index)
        if not self.kline_df_with_ta.empty and not self.kline_df_with_ta.index.is_monotonic_increasing:
            self.kline_df_with_ta.sort_index(inplace=True)

        self.kline_window_size = int(self.config["kline_window_size"])
        self.tick_feature_window_size = int(self.config["tick_feature_window_size"])
        self.tick_features_to_use = self.config.get("tick_features_to_use", ["Price"])
        self.initial_balance = float(self.config["initial_balance"])
        self.commission_pct = float(self.config["commission_pct"])
        self.base_trade_amount_ratio = float(self.config["base_trade_amount_ratio"])
        self.catastrophic_loss_limit = self.initial_balance * (1.0 - float(self.config["catastrophic_loss_threshold_pct"]))
        self.log_level = self.config.get("log_level", "normal")
        
        # Action Space Definition
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(
                low=float(self.config["min_profit_target_low"]),
                high=float(self.config["min_profit_target_high"]),
                shape=(1,),
                dtype=np.float32
            )
        ))

        # Observation Space Definition
        num_tick_features = len(self.tick_features_to_use)
        num_kline_features = len(self.config.get("kline_price_features", []))
        obs_shape_dim = (self.tick_feature_window_size * num_tick_features) + (self.kline_window_size * num_kline_features) + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape_dim,), dtype=np.float32)

        self._prepare_data()
        self.reset()
        self.env_id = f"STE-{os.getpid()}"


    def _prepare_data(self):
        # Data preparation logic remains the same...
        if self.tick_df.empty:
            self.decision_prices = np.array([])
            self.start_step = 0
            self.end_step = -1
            return
        if not isinstance(self.tick_df.index, pd.DatetimeIndex):
            self.tick_df.index = pd.to_datetime(self.tick_df.index)
        if not self.tick_df.index.is_monotonic_increasing:
            self.tick_df.sort_index(inplace=True)
        self.tick_price_data = {name: self.tick_df[name].values.astype(np.float32) for name in self.tick_features_to_use}
        self.decision_prices = self.tick_price_data.get('Price', np.array([]))
        self.kline_feature_arrays = {name: self.kline_df_with_ta[name].values.astype(np.float32) for name in self.config["kline_price_features"] if name in self.kline_df_with_ta}
        self.start_step = self.tick_feature_window_size - 1
        self.end_step = len(self.tick_df) - 1

    def _get_observation(self) -> np.ndarray:
        # Observation logic remains the same...
        if self.tick_df.empty: return np.zeros(self.observation_space.shape, dtype=np.float32)
        safe_step = min(self.current_step, self.end_step)
        
        # Tick Features
        tick_start_idx = max(0, safe_step - self.tick_feature_window_size + 1)
        tick_features_list = [self.tick_price_data[name][tick_start_idx : safe_step + 1] for name in self.tick_features_to_use]
        # Pad if necessary
        for i, arr in enumerate(tick_features_list):
            if len(arr) < self.tick_feature_window_size:
                padding = np.full(self.tick_feature_window_size - len(arr), arr[0] if len(arr) > 0 else 0)
                tick_features_list[i] = np.concatenate((padding, arr))
        tick_features = np.concatenate(tick_features_list)

        # K-line Features
        kline_features = np.zeros(self.kline_window_size * len(self.config["kline_price_features"]), dtype=np.float32)
        if not self.kline_df_with_ta.empty:
            current_ts = self.tick_df.index[safe_step]
            kline_idx = self.kline_df_with_ta.index.get_indexer([current_ts], method='ffill')[0]
            if kline_idx != -1:
                kline_start_idx = max(0, kline_idx - self.kline_window_size + 1)
                kline_features_list = []
                for name in self.config["kline_price_features"]:
                    series = self.kline_feature_arrays.get(name, np.zeros(len(self.kline_df_with_ta)))
                    arr = series[kline_start_idx : kline_idx + 1]
                    if len(arr) < self.kline_window_size:
                        padding = np.full(self.kline_window_size - len(arr), arr[0] if len(arr) > 0 else 0)
                        arr = np.concatenate((padding, arr))
                    kline_features_list.append(arr)
                kline_features = np.concatenate(kline_features_list)

        # Portfolio State
        price = self.decision_prices[safe_step]
        pos_open = 1.0 if self.position_open else 0.0
        entry_price_norm = (self.entry_price / price - 1) if self.position_open else 0.0
        pnl_norm = ((price - self.entry_price) * self.position_volume) / self.initial_balance if self.position_open else 0.0
        portfolio_state = np.array([pos_open, entry_price_norm, pnl_norm], dtype=np.float32)
        
        return np.concatenate([tick_features, kline_features, portfolio_state])

    def _get_info(self) -> dict:
        # Info dictionary logic remains the same...
        price = self.decision_prices[min(self.current_step, self.end_step)]
        equity = self.current_balance + (self.position_volume * price if self.position_open else 0)
        return {"current_step": self.current_step, "current_tick_price": price, "equity": equity, "position_open": self.position_open}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_balance = self.initial_balance
        self.position_open = False
        self.entry_price = 0.0
        self.position_volume = 0.0
        self.current_step = self.start_step
        self.trade_history = []
        return self._get_observation(), self._get_info()

    def step(self, action_tuple):
        # This is the core logic that has been updated
        discrete_action, _ = action_tuple
        if isinstance(discrete_action, np.integer): discrete_action = int(discrete_action)
        
        reward = 0.0
        terminated = False
        truncated = False
        price = self.decision_prices[self.current_step]

        if discrete_action == 1 and not self.position_open: # BUY
            trade_size = self.initial_balance * self.base_trade_amount_ratio
            self.position_volume = trade_size / price
            cost = self.position_volume * price * (1 + self.commission_pct)
            self.current_balance -= cost
            self.position_open = True
            self.entry_price = price

        elif discrete_action == 2 and self.position_open: # SELL
            revenue = self.position_volume * price * (1 - self.commission_pct)
            pnl = revenue - (self.position_volume * self.entry_price)
            self.current_balance += revenue
            
            actual_pnl_ratio = (price / self.entry_price - 1) if self.entry_price > 0 else 0
            profit_target = self.config['profit_target_pct']

            if pnl <= 0:
                reward = pnl * self.config.get("penalty_loss_trade_factor", 1.0)
            else:
                performance_vs_target = actual_pnl_ratio - profit_target
                if performance_vs_target >= 0:
                    reward = performance_vs_target * self.config['reward_factor_above_target']
                else:
                    reward = performance_vs_target * self.config['penalty_factor_below_target']
            
            self.position_open = False
            self.trade_history.append({'type': 'sell', 'step': self.current_step, 'price': price, 'pnl': pnl})

        elif discrete_action == 0: # HOLD
            if self.position_open:
                if price < self.entry_price:
                    reward += self.config["penalty_hold_losing_position"]
            else:
                reward += self.config["penalty_hold_flat_position"]

        self.current_step += 1
        
        # Termination conditions
        equity = self.current_balance + (self.position_volume * price if self.position_open else 0)
        if equity < self.catastrophic_loss_limit:
            terminated = True
            reward -= 100 # Large penalty for ruin
        if self.current_step > self.end_step:
            truncated = True
            if self.position_open:
                # Liquidate at end of episode
                self.current_balance += self.position_volume * price * (1 - self.commission_pct)
                self.position_open = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def close(self):
        pass