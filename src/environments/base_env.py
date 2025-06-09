# src/environments/base_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# --- A complete dictionary of all environment parameters ---
DEFAULT_ENV_CONFIG = {
    "kline_window_size": 20, "tick_feature_window_size": 50,
    "kline_price_features": ["Open", "High", "Low", "Close", "Volume"],
    "tick_features_to_use": ["Price", "Quantity"],
    "initial_balance": 10000.0, "commission_pct": 0.001,
    "base_trade_amount_ratio": 0.02, "catastrophic_loss_threshold_pct": 0.3,
    "penalty_catastrophic_loss": -100.0,
    "penalty_hold_losing_position": -0.0005,
    "penalty_hold_flat_position": -0.0001,
    "reward_hold_profitable_position": 0.0001,
    "penalty_sell_no_position": -0.1,
    "penalty_buy_position_already_open": -0.1,
}

class SimpleTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    ACTION_MAP = {0: "Hold", 1: "Buy", 2: "Sell"}

    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        super().__init__()
        # The provided config now cleanly overrides the complete default config
        self.config = {**DEFAULT_ENV_CONFIG, **(config if config else {})}

        if tick_df.empty: raise ValueError("tick_df must be non-empty.")
        self.tick_df = tick_df.copy()
        self.kline_df_with_ta = kline_df_with_ta.copy()

        if not self.kline_df_with_ta.empty:
            if not isinstance(self.kline_df_with_ta.index, pd.DatetimeIndex): self.kline_df_with_ta.index = pd.to_datetime(self.kline_df_with_ta.index)
            if not self.kline_df_with_ta.index.is_monotonic_increasing: self.kline_df_with_ta.sort_index(inplace=True)

        # All parameters are now consistently accessed from self.config
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
        self.kline_feature_arrays = {name: self.kline_df_with_ta[name].values.astype(np.float32) for name in self.config["kline_price_features"] if name in self.kline_df_with_ta}
        self.start_step, self.end_step = self.tick_feature_window_size - 1, len(self.tick_df) - 1

    def _get_observation(self) -> np.ndarray:
        safe_step = min(self.current_step, self.end_step)
        tick_start_idx = max(0, safe_step - self.tick_feature_window_size + 1)
        tick_features_list = []
        for name in self.config["tick_features_to_use"]:
            arr = self.tick_price_data[name][tick_start_idx:safe_step + 1]; padding = self.tick_feature_window_size - len(arr)
            if padding > 0: arr = np.concatenate((np.full(padding, arr[0] if len(arr) > 0 else 0), arr))
            tick_features_list.append(arr)
        tick_features = np.concatenate(tick_features_list)

        kline_features = np.zeros(self.kline_window_size * len(self.config["kline_price_features"]), dtype=np.float32)
        if not self.kline_df_with_ta.empty and self.kline_df_with_ta.index.get_indexer([self.tick_df.index[safe_step]], method='ffill')[0] != -1:
            kline_idx = self.kline_df_with_ta.index.get_indexer([self.tick_df.index[safe_step]], method='ffill')[0]
            kline_start_idx = max(0, kline_idx - self.kline_window_size + 1)
            kline_features_list = []
            for name in self.config["kline_price_features"]:
                series = self.kline_feature_arrays.get(name, np.zeros(len(self.kline_df_with_ta)))
                arr = series[kline_start_idx:kline_idx + 1]; padding = self.kline_window_size - len(arr)
                if padding > 0: arr = np.concatenate((np.full(padding, arr[0] if len(arr) > 0 else 0), arr))
                kline_features_list.append(arr)
            if kline_features_list: kline_features = np.concatenate(kline_features_list)

        price = self.decision_prices[safe_step]
        portfolio_state = np.array([1.0 if self.position_open else 0.0, (self.entry_price / price - 1) if self.position_open and price > 1e-9 else 0.0, ((price - self.entry_price) * self.position_volume) / self.initial_balance if self.position_open else 0.0], dtype=np.float32)
        return np.concatenate([tick_features, kline_features, portfolio_state])

    def _get_info(self) -> dict:
        price = self.decision_prices[min(self.current_step, self.end_step)]
        equity = self.current_balance + (self.position_volume * price if self.position_open else 0)
        return {"current_step": self.current_step, "equity": equity, "position_open": self.position_open, "current_tick_price": price}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_balance, self.position_open, self.entry_price, self.position_volume = self.initial_balance, False, 0.0, 0.0
        self.current_step = self.start_step
        return self._get_observation(), self._get_info()

    def step(self, action_tuple):
        discrete_action, _ = action_tuple
        reward, terminated, truncated = 0.0, False, False
        price = self.decision_prices[self.current_step]

        # --- Define simple reward constants for rule adherence ---
        REWARD_CORRECT_TRADE_ACTION = 1.0    # Reward for BUY when flat, SELL when in position
        PENALTY_INCORRECT_TRADE_ACTION = -1.0 # Penalty for BUY when in position, SELL when flat
        PENALTY_HOLD_ACTION = -0.05          # Small penalty for HOLD to encourage trading actions

        # --- Action: BUY (discrete_action == 1) ---
        if discrete_action == 1:
            if not self.position_open:
                # Agent correctly attempts to BUY when FLAT
                cost = (self.initial_balance * self.base_trade_amount_ratio) * (1 + self.commission_pct)
                if self.current_balance >= cost: # Check if balance is sufficient
                    self.position_volume = (self.initial_balance * self.base_trade_amount_ratio) / price
                    self.current_balance -= cost
                    self.position_open, self.entry_price = True, price
                    reward += REWARD_CORRECT_TRADE_ACTION
                    self.entry_price_before_action = price
                else:
                    # Penalize: Correct intent (BUY when flat) but insufficient funds
                    # This teaches the agent about balance constraints indirectly
                    reward += PENALTY_INCORRECT_TRADE_ACTION / 2 # Smaller penalty than a completely wrong action type
            else:
                # Agent incorrectly attempts to BUY when already IN POSITION
                reward += PENALTY_INCORRECT_TRADE_ACTION

        # --- Action: SELL (discrete_action == 2) ---
        elif discrete_action == 2:
            if self.position_open:
                # Agent correctly attempts to SELL when IN POSITION
                revenue = self.position_volume * price * (1 - self.commission_pct)
                self.current_balance += revenue
                self.position_open, self.position_volume, self.entry_price = False, 0.0, 0.0
                reward += REWARD_CORRECT_TRADE_ACTION


                actual_pnl_ratio = (price / self.entry_price_before_action - 1) if self.entry_price_before_action > 1e-9 else 0.0

                # Calculate proportional component based on your desired mapping:
                # Maps actual_pnl_ratio=0 to -1 reward component
                # Maps actual_pnl_ratio=profit_target_pct to +1 reward component
                
                target_pct = self.config['profit_target_pct']
                
                if target_pct <= 1e-9: # Handle cases where target_pct is zero or extremely small to avoid division by zero
                    proportional_component = -1.0 # If target is 0, and PnL is 0, reward is -1. If PnL is negative, it's very negative.
                else:
                    proportional_component = (2 / target_pct) * actual_pnl_ratio - 1
                
                # --- CHANGE STARTS HERE ---
                # Removed all clipping to allow proportional_component to go arbitrarily high or low.
                # --- CHANGE ENDS HERE ---

                # Final reward for successful sell: Base reward (+1) + Proportional component
                reward += proportional_component


            else:
                # Agent incorrectly attempts to SELL when FLAT
                reward += PENALTY_INCORRECT_TRADE_ACTION

        # --- Action: HOLD (discrete_action == 0) ---
        elif discrete_action == 0:
            # Penalize HOLD to encourage the agent to take BUY/SELL actions
            # regardless of position status initially, then learn the rules.
            reward += PENALTY_HOLD_ACTION

        # --- Common logic for advancing step and checking for termination ---
        self.current_step += 1
        current_equity = self.current_balance + (self.position_volume * price if self.position_open else 0)

        # Catastrophic loss: Still a hard termination and significant penalty
        if current_equity < self.catastrophic_loss_limit:
            terminated = True
            reward += self.config["penalty_catastrophic_loss"]

        # End of data: Episode truncated
        if self.current_step > self.end_step:
            truncated = True
        
        # Force close any open position at the end of the episode to clean up
        if (terminated or truncated) and self.position_open:
            self.current_balance += self.position_volume * price * (1 - self.commission_pct)
            self.position_open = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def close(self): pass