# base_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json # For pretty printing info if needed
import os # For env_id
import traceback

# Default configuration for the environment (Tick-based actions)
DEFAULT_ENV_CONFIG = {
    # K-line data configuration (e.g., for 1H candles)
    "kline_window_size": 20,                # Number of historical k-lines (e.g., 1H candles) to include in obs
    "kline_price_features": ["Open", "High", "Low", "Close", "Volume"], # Base features from k-lines
    # price_features_to_add (TAs like SMA, RSI) will be dynamically added based on kline_df_with_ta columns

    # Tick data configuration
    "tick_feature_window_size": 50,         # Number of historical ticks to include in obs
    "tick_features_to_use": ["Price", "Quantity"], # Features to extract from tick data for observation

    # General trading parameters
    "initial_balance": 10000.0,
    "commission_pct": 0.001,                # Commission per trade
    "base_trade_amount_ratio": 0.1,         # Proportion of initial_balance to trade
    "min_tradeable_unit": 1e-6,             # Smallest unit of asset that can be traded
    "catastrophic_loss_threshold_pct": 0.3, # Episode terminates if equity drops below this % of initial
    "obs_clip_low": -5.0,
    "obs_clip_high": 5.0,

    # Continuous Profit Target action parameter ranges
    "min_profit_target_low": 0.001,        # e.g., 0.1% (Adjust for tick-level decisions)
    "min_profit_target_high": 0.05,        # e.g., 5% (Adjust for tick-level decisions)

    # Reward and Penalty settings (May need significant tuning for tick-level actions)
    "reward_open_buy_position": 0.001,      # Smaller for tick-level
    "penalty_buy_insufficient_balance": -0.1,
    "penalty_buy_position_already_open": -0.05,
    "reward_sell_profit_base": 0.1,         # Scaled down
    "reward_sell_profit_factor": 10.0,      # Scaled down
    "penalty_sell_loss_factor": 10.0,       # Scaled down
    "penalty_sell_loss_base": -0.01,        # Scaled down
    "penalty_sell_no_position": -0.1,
    "reward_hold_profitable_position": 0.0001, # Very small for tick-level
    "penalty_hold_losing_position": -0.0005, # Very small for tick-level
    "penalty_hold_flat_position": -0.0001,   # Very small for tick-level
    "penalty_catastrophic_loss": -100.0,
    "reward_eof_sell_factor": 5.0,          # Scaled down

    "reward_sell_meets_target_bonus": 0.5,  # Scaled down
    "penalty_sell_profit_below_target": -0.05, # Scaled down

    "custom_print_render": "none",
    "log_level": "normal",
    # "price_features_to_add" is no longer directly used here; TAs come from kline_df_with_ta
}

class SimpleTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array'], 'render_fps': 100} # Higher FPS for ticks
    ACTION_MAP = {0: "Hold", 1: "Buy", 2: "Sell"}

    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        super().__init__()
        self.config = {**DEFAULT_ENV_CONFIG, **(config if config else {})}

        if not isinstance(tick_df, pd.DataFrame) or tick_df.empty:
            raise ValueError("tick_df must be a non-empty pandas DataFrame.")
        if not isinstance(kline_df_with_ta, pd.DataFrame) or kline_df_with_ta.empty:
            raise ValueError("kline_df_with_ta must be a non-empty pandas DataFrame.")
        
        self.tick_df = tick_df.copy() # This is the primary DataFrame for stepping
        self.kline_df_with_ta = kline_df_with_ta.copy()

        # Ensure kline_df_with_ta index is DatetimeIndex and sorted
        if not isinstance(self.kline_df_with_ta.index, pd.DatetimeIndex):
            self.kline_df_with_ta.index = pd.to_datetime(self.kline_df_with_ta.index)
        if not self.kline_df_with_ta.index.is_monotonic_increasing:
            self.kline_df_with_ta.sort_index(inplace=True)


        # Load parameters from config
        self.kline_window_size = int(self.config["kline_window_size"])
        self.tick_feature_window_size = int(self.config["tick_feature_window_size"])
        self.tick_features_to_use = self.config.get("tick_features_to_use", ["Price"])
        
        self.initial_balance = float(self.config["initial_balance"])
        self.commission_pct = float(self.config["commission_pct"])
        self.base_trade_amount_ratio = float(self.config["base_trade_amount_ratio"])
        self.min_tradeable_unit = float(self.config["min_tradeable_unit"])
        cat_loss_pct = float(self.config["catastrophic_loss_threshold_pct"])
        self.catastrophic_loss_limit = self.initial_balance * (1.0 - cat_loss_pct)
        self.obs_clip_low = float(self.config["obs_clip_low"])
        self.obs_clip_high = float(self.config["obs_clip_high"])
        self.custom_print_render = self.config.get("custom_print_render", "none")
        self.log_level = self.config.get("log_level", "normal") # Default log level from config
        
        # Override log_level for debugging during development if needed
        # self.log_level = "detailed" # Temporary override for debugging

        self.min_profit_target_low = float(self.config.get("min_profit_target_low"))
        self.min_profit_target_high = float(self.config.get("min_profit_target_high"))
        self.reward_sell_meets_target_bonus = float(self.config.get("reward_sell_meets_target_bonus"))
        self.penalty_sell_profit_below_target = float(self.config.get("penalty_sell_profit_below_target"))

        # Define Action Space
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(
                low=self.min_profit_target_low,
                high=self.min_profit_target_high,
                shape=(1,),
                dtype=np.float32
            )
        ))

        # Define Observation Space
        # 1. Tick features part
        self.num_tick_features_per_step = len(self.tick_features_to_use)
        obs_shape_ticks = self.tick_feature_window_size * self.num_tick_features_per_step
        
        # 2. K-line features part (OHLCV + TAs)
        # kline_price_features from config now includes base OHLCV AND selected TAs
        self.kline_base_price_features = [f for f in self.config.get("kline_price_features", ["Open", "High", "Low", "Close", "Volume"]) if f in ['Open', 'High', 'Low', 'Close', 'Volume']]
        self.kline_ta_features = [f for f in self.config.get("kline_price_features", []) if f not in self.kline_base_price_features]

        self.num_kline_features_per_step = len(self.kline_base_price_features) + len(self.kline_ta_features)
        obs_shape_klines = self.kline_window_size * self.num_kline_features_per_step
        
        # 3. Additional portfolio state features
        additional_state_features = 3 # pos_open, norm_entry_price, norm_pnl
        
        obs_shape_dim = obs_shape_ticks + obs_shape_klines + additional_state_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape_dim,), dtype=np.float32)

        self._prepare_data()

        self.current_balance = 0.0
        self.position_open = False
        self.entry_price = 0.0
        self.position_volume = 0.0
        self.current_desired_profit_target = 0.0
        self.current_step = 0 # Iterates over ticks
        self.trade_history = []
        self.env_id = f"STE_Tick-{os.getpid()}-{id(self)%1000}"

        if self.log_level == "detailed":
            print(f"ENV_INIT (ID: {self.env_id}): Tick-based environment initialized.")
            print(f"  Tick Feature Window: {self.tick_feature_window_size} ticks, Features per tick: {self.num_tick_features_per_step} ({self.tick_features_to_use})")
            print(f"  K-line Window: {self.kline_window_size} k-lines, Features per k-line: {self.num_kline_features_per_step} ({self.kline_base_price_features + self.kline_ta_features})")
            print(f"  Action Space: {self.action_space}")
            print(f"  Observation space shape: {self.observation_space.shape} -> {obs_shape_ticks}(ticks) + {obs_shape_klines}(klines) + {additional_state_features}(portfolio)")


    def _prepare_data(self):
        if not isinstance(self.tick_df.index, pd.DatetimeIndex):
            try: self.tick_df.index = pd.to_datetime(self.tick_df.index)
            except Exception as e: raise ValueError(f"Could not convert tick_df index to DatetimeIndex: {e}")
        if not self.tick_df.index.is_monotonic_increasing:
            self.tick_df.sort_index(inplace=True)

        # Extract tick data arrays
        self.tick_price_data = {}
        for feature_name in self.tick_features_to_use:
            if feature_name not in self.tick_df.columns:
                raise ValueError(f"Tick feature '{feature_name}' not found in tick_df columns: {self.tick_df.columns.tolist()}")
            self.tick_price_data[feature_name] = self.tick_df[feature_name].values.astype(np.float32)
        
        # Decision prices are the 'Price' feature from tick data
        if 'Price' not in self.tick_price_data:
            raise ValueError("'Price' must be one of the tick_features_to_use for decision making.")
        self.decision_prices = self.tick_price_data['Price']

        # Extract k-line data arrays
        self.kline_feature_arrays = {}
        # Iterate over ALL kline_price_features (base + TAs) as they should all be in kline_df_with_ta
        for feature_name in self.config.get("kline_price_features", []):
            if feature_name not in self.kline_df_with_ta.columns:
                # Fallback if a requested feature is missing from the provided kline_df_with_ta
                print(f"Warning: K-line feature '{feature_name}' not found in kline_df_with_ta columns: {self.kline_df_with_ta.columns.tolist()}. Will use zeros.")
                self.kline_feature_arrays[feature_name] = np.zeros(len(self.kline_df_with_ta), dtype=np.float32)
            else:
                self.kline_feature_arrays[feature_name] = self.kline_df_with_ta[feature_name].values.astype(np.float32)

        if len(self.decision_prices) < self.tick_feature_window_size:
            raise ValueError(f"Tick data length ({len(self.decision_prices)}) is less than tick_feature_window_size ({self.tick_feature_window_size}).")

        self.start_step = self.tick_feature_window_size - 1
        self.end_step = len(self.tick_df) - 1

    def _get_observation(self) -> np.ndarray:
        safe_current_step = min(self.current_step, self.end_step)

        # 1. Tick Features
        tick_obs_parts = []
        tick_start_idx = max(0, safe_current_step - self.tick_feature_window_size + 1)
        for feature_name in self.tick_features_to_use:
            series = self.tick_price_data[feature_name]
            actual_slice = series[tick_start_idx : safe_current_step + 1]
            if len(actual_slice) < self.tick_feature_window_size:
                padding = np.full(self.tick_feature_window_size - len(actual_slice), series[0] if len(series)>0 else 0.0, dtype=np.float32)
                window_slice = np.concatenate((padding, actual_slice))
            else:
                window_slice = actual_slice
            tick_obs_parts.append(window_slice)
        
        tick_features_stacked = np.concatenate(tick_obs_parts) if tick_obs_parts else np.array([], dtype=np.float32)

        # 2. K-line Features
        kline_obs_parts = []
        current_tick_timestamp = self.tick_df.index[safe_current_step]
        
        kline_current_idx_pos = self.kline_df_with_ta.index.get_indexer([current_tick_timestamp], method='ffill')
        
        if kline_current_idx_pos[0] == -1:
            if self.log_level != "none": print(f"Warning (ENV_ID: {self.env_id}): No k-line data found for current tick timestamp {current_tick_timestamp}. Using zeros for k-line features.")
            kline_features_stacked = np.zeros(self.kline_window_size * self.num_kline_features_per_step, dtype=np.float32)
        else:
            kline_current_idx = kline_current_idx_pos[0]
            kline_start_idx = max(0, kline_current_idx - self.kline_window_size + 1)

            # Iterate over all features that define the K-line part of the observation
            all_kline_obs_features = self.config.get("kline_price_features", ["Open", "High", "Low", "Close", "Volume"])
            
            for feature_name in all_kline_obs_features:
                series = self.kline_feature_arrays.get(feature_name) # Use .get() in case feature was missing earlier
                if series is None: # Fallback if feature was not in kline_feature_arrays (e.g. not in kline_df_with_ta)
                    series = np.zeros(len(self.kline_df_with_ta), dtype=np.float32) # Provide zeros
                
                actual_slice = series[kline_start_idx : kline_current_idx + 1]
                if len(actual_slice) < self.kline_window_size:
                    padding_val = actual_slice[0] if len(actual_slice) > 0 else 0.0 # Pad with first known value or 0
                    padding = np.full(self.kline_window_size - len(actual_slice), padding_val, dtype=np.float32)
                    window_slice = np.concatenate((padding, actual_slice))
                else:
                    window_slice = actual_slice
                kline_obs_parts.append(window_slice)
            kline_features_stacked = np.concatenate(kline_obs_parts) if kline_obs_parts else np.array([], dtype=np.float32)


        # 3. Portfolio State Features
        current_decision_price = self.decision_prices[safe_current_step]
        is_pos_open_feat = 1.0 if self.position_open else 0.0
        norm_entry_price_feat = 0.0
        unreal_pnl_ratio_feat = 0.0
        if self.position_open and self.entry_price > 1e-9:
            norm_entry_price_feat = self.entry_price / (current_decision_price + 1e-9)
            unrealized_pnl = (current_decision_price - self.entry_price) * self.position_volume
            unreal_pnl_ratio_feat = unrealized_pnl / (self.initial_balance + 1e-9)
        
        additional_features = np.array([
            is_pos_open_feat,
            np.clip(norm_entry_price_feat, self.obs_clip_low, self.obs_clip_high),
            np.clip(unreal_pnl_ratio_feat, self.obs_clip_low, self.obs_clip_high)
        ], dtype=np.float32)

        # Concatenate all parts
        observation = np.concatenate([
            tick_features_stacked, 
            kline_features_stacked, 
            additional_features
        ])
        
        # Apply nan_to_num and clipping
        final_observation = np.nan_to_num(observation, nan=0.0, posinf=self.obs_clip_high, neginf=self.obs_clip_low).astype(np.float32)

        # Add these lines to print or log observation details
        if self.log_level == "detailed":
            # Set print options to display full arrays
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            print(f"\n--- Observation at Tick {safe_current_step} ---")
            print(f"  Observation shape: {final_observation.shape}")
            print(f"  Tick Features:\n{tick_features_stacked}") # Print whole array
            print(f"  K-line Features:\n{kline_features_stacked}") # Print whole array
            print(f"  Additional State Features: {additional_features}") # Already small, print as is
            print(f"  Min/Max Observation values: {np.min(final_observation):.4f} / {np.max(final_observation):.4f}")
            print("---------------------------------------")
            # Reset print options to default to avoid affecting other prints
            np.set_printoptions(threshold=1000, linewidth=75) # Default values, adjust if needed

        return final_observation

    def _get_info(self) -> dict:
        safe_step = min(self.current_step, self.end_step)
        current_tick_price = self.decision_prices[safe_step] if safe_step < len(self.decision_prices) else 0.0
        
        current_position_value = self.position_volume * current_tick_price if self.position_open else 0.0
        equity = self.current_balance + current_position_value
        return {
            "current_step": self.current_step,
            "current_tick_price": current_tick_price,
            "current_tick_timestamp": self.tick_df.index[safe_step].isoformat() if safe_step < len(self.tick_df.index) else None,
            "current_balance": self.current_balance,
            "equity": equity,
            "position_open": self.position_open,
            "entry_price": self.entry_price,
            "position_volume": self.position_volume,
            "current_desired_profit_target": self.current_desired_profit_target if self.position_open else 0.0,
            "num_trades_in_episode": len([t for t in self.trade_history if t['type'] in ['buy', 'sell']])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_balance = self.initial_balance
        self.position_open = False
        self.entry_price = 0.0
        self.position_volume = 0.0
        self.current_desired_profit_target = 0.0
        self.current_step = self.start_step # Start from where we have enough tick history

        initial_equity = self.current_balance
        self.trade_history = [{'step': self.current_step -1, 'type': 'initial_balance', 'price': 0, 'volume': 0, 
                               'balance': self.current_balance, 'pnl': 0, 'equity': initial_equity, 'commission': 0, 
                               'profit_target':0}]
        if self.log_level == "detailed":
            print(f"ENV_RESET (ID: {self.env_id}): current_step set to {self.current_step}, balance={self.current_balance:.2f}")
            print(f"ENV_RESET (ID: {self.env_id}): Catastrophic Loss Limit set to {self.catastrophic_loss_limit:.2f}") # Debug print
        return self._get_observation(), self._get_info()

    def step(self, action_tuple):
        # Unpack action
        discrete_action = action_tuple[0]
        profit_target_param = action_tuple[1][0]
        if isinstance(discrete_action, np.integer): discrete_action = int(discrete_action)

        current_price_for_action = self.decision_prices[self.current_step] 
        reward = 0.0
        terminated = False
        truncated = False
        trade_executed_this_step = False
        trade_details_for_log = {}

        if self.log_level == "detailed":
            print(f"\n--- ENV_STEP (ID: {self.env_id}, TickStep: {self.current_step}, Time: {self.tick_df.index[self.current_step]}) ---")
            print(f"  DEBUG: Action: {self.ACTION_MAP.get(discrete_action, 'Unk')}, ProfitTgtParam={profit_target_param:.4f}")
            print(f"  DEBUG: Initial Reward for step: {reward:.4f}")
            print(f"  DEBUG: State B4 Action: Bal={self.current_balance:.4f}, PosOp={self.position_open}, EntryP={self.entry_price:.4f}, Vol={self.position_volume:.4f}, TgtP={self.current_desired_profit_target:.4f}")
            print(f"  DEBUG: Current Tick Price: {current_price_for_action:.4f}")


        # --- Process Agent's Action ---
        if discrete_action == 1: # BUY
            if not self.position_open:
                cash_for_trade = self.initial_balance * self.base_trade_amount_ratio
                units_to_buy_precise = cash_for_trade / (current_price_for_action + 1e-9) 
                units_to_buy = max(self.min_tradeable_unit, np.floor(units_to_buy_precise / self.min_tradeable_unit) * self.min_tradeable_unit)
                cost_of_units = units_to_buy * current_price_for_action
                commission_cost = cost_of_units * self.commission_pct
                total_cost = cost_of_units + commission_cost

                if self.current_balance >= total_cost and units_to_buy > 0:
                    self.current_balance -= total_cost
                    self.position_open = True
                    self.entry_price = current_price_for_action 
                    self.position_volume = units_to_buy
                    self.current_desired_profit_target = profit_target_param
                    reward += self.config["reward_open_buy_position"]
                    equity_after_buy = self.current_balance + (self.position_volume * current_price_for_action)
                    trade_details_for_log = {'type': 'buy', 'price': current_price_for_action, 'volume': units_to_buy, 
                                             'balance': self.current_balance, 'pnl': 0, 'commission': commission_cost, 
                                             'equity': equity_after_buy, 'profit_target_set': self.current_desired_profit_target}
                    trade_executed_this_step = True
                    if self.log_level == "detailed": print(f"  DEBUG: BUY executed. New balance={self.current_balance:.4f}, Position Vol={self.position_volume:.4f}, Reward={reward:.4f}")
                else: 
                    reward += self.config["penalty_buy_insufficient_balance"]
                    if self.log_level == "detailed": print(f"  DEBUG: BUY failed due to insufficient balance. Reward={reward:.4f}")
            else: 
                reward += self.config["penalty_buy_position_already_open"]
                if self.log_level == "detailed": print(f"  DEBUG: BUY failed as position already open. Reward={reward:.4f}")


        elif discrete_action == 2: # SELL
            if self.position_open:
                revenue_from_sale = self.position_volume * current_price_for_action
                commission_cost = revenue_from_sale * self.commission_pct
                net_revenue = revenue_from_sale - commission_cost
                cost_basis_of_position = self.entry_price * self.position_volume
                pnl_this_trade = net_revenue - cost_basis_of_position

                self.current_balance += net_revenue
                actual_pnl_ratio = (current_price_for_action / self.entry_price - 1) if self.entry_price > 1e-9 else 0
                
                if self.log_level == "detailed":
                    print(f"  DEBUG: SELL in progress. PnL This Trade={pnl_this_trade:.4f}, Actual PnL Ratio={actual_pnl_ratio:.4f}")

                if pnl_this_trade > 0:
                    reward += self.config["reward_sell_profit_base"] + (pnl_this_trade / (self.initial_balance + 1e-9)) * self.config["reward_sell_profit_factor"]
                    if self.current_desired_profit_target > 0:
                        if actual_pnl_ratio >= self.current_desired_profit_target:
                            reward += self.reward_sell_meets_target_bonus
                            if self.log_level == "detailed": print(f"  DEBUG: Profit target MET. Bonus added. Current Reward={reward:.4f}")
                        else: 
                            reward += self.penalty_sell_profit_below_target 
                            if self.log_level == "detailed": print(f"  DEBUG: Profit below target. Penalty added. Current Reward={reward:.4f}")
                    if self.log_level == "detailed": print(f"  DEBUG: Profitable SELL. Base reward added. Current Reward={reward:.4f}")
                else: 
                    reward += (pnl_this_trade / (self.initial_balance + 1e-9)) * self.config["penalty_sell_loss_factor"] + self.config["penalty_sell_loss_base"]
                    if self.log_level == "detailed": print(f"  DEBUG: Losing SELL. Penalty added. Current Reward={reward:.4f}")
                
                trade_details_for_log = {'type': 'sell', 'price': current_price_for_action, 'volume': self.position_volume, 
                                           'balance': self.current_balance, 'pnl': pnl_this_trade, 'commission': commission_cost, 
                                           'equity': self.current_balance, 'profit_target_aimed': self.current_desired_profit_target,
                                           'pnl_ratio_achieved': actual_pnl_ratio}
                self.position_open = False; self.entry_price = 0.0; self.position_volume = 0.0; self.current_desired_profit_target = 0.0
                trade_executed_this_step = True
                if self.log_level == "detailed": print(f"  DEBUG: SELL executed. New balance={self.current_balance:.4f}, Position Closed. Reward={reward:.4f}")
            else: 
                reward += self.config["penalty_sell_no_position"]
                if self.log_level == "detailed": print(f"  DEBUG: SELL failed as no position open. Reward={reward:.4f}")

        elif discrete_action == 0: # HOLD
            if self.position_open:
                unrealized_pnl_at_hold = (current_price_for_action - self.entry_price) * self.position_volume
                if unrealized_pnl_at_hold > 0: 
                    reward += self.config["reward_hold_profitable_position"]
                    if self.log_level == "detailed": print(f"  DEBUG: HOLD with profitable position. Reward={reward:.4f}")
                else: 
                    reward += self.config["penalty_hold_losing_position"]
                    if self.log_level == "detailed": print(f"  DEBUG: HOLD with losing position. Reward={reward:.4f}")
            else: 
                reward += self.config["penalty_hold_flat_position"]
                if self.log_level == "detailed": print(f"  DEBUG: HOLD with flat position. Reward={reward:.4f}")
        
        if trade_executed_this_step:
            self.trade_history.append({'step': self.current_step, 'time': self.tick_df.index[self.current_step].isoformat(), **trade_details_for_log})
            if self.log_level == "detailed": print(f"  DEBUG: Trade logged to history. History Length: {len(self.trade_history)}")

        self.current_step += 1

        # Check for catastrophic loss or end of data
        current_equity_for_loss_check = self.current_balance + (self.position_volume * current_price_for_action if self.position_open else 0)
        
        if self.log_level == "detailed":
            print(f"  DEBUG: Before Termination Check: current_equity_for_loss_check={current_equity_for_loss_check:.4f}, Cat Loss Limit={self.catastrophic_loss_limit:.4f}, Terminated={terminated}")
            print(f"  DEBUG: Current Reward before final adjustments: {reward:.4f}")


        if current_equity_for_loss_check < self.catastrophic_loss_limit:
            terminated = True
            reward += self.config["penalty_catastrophic_loss"]
            if self.position_open:
                price_at_ruin = self.decision_prices[min(self.current_step -1, self.end_step)]
                self._liquidate_position(price_at_ruin, "sell_ruin_auto", self.current_step-1)
            if self.log_level == "detailed": print(f"  DEBUG: CATASTROPHIC LOSS TRIGGERED! New Reward={reward:.4f}, Terminated={terminated}")

        if not terminated and self.current_step > self.end_step:
            truncated = True
            if self.position_open:
                price_at_eof = self.decision_prices[self.end_step]
                _, _, pnl_eof = self._liquidate_position(price_at_eof, "sell_eof_auto", self.end_step)
                if pnl_eof > 0:
                     reward += (pnl_eof / (self.initial_balance + 1e-9)) * self.config["reward_eof_sell_factor"]
            if self.log_level == "detailed": print(f"  DEBUG: EPISODE TRUNCATED AT EOF! Reward={reward:.4f}, Truncated={truncated}")
        
        observation = self._get_observation()
        info = self._get_info()

        if self.log_level == "detailed":
            print(f"  DEBUG: Final Return: Reward={np.nan_to_num(reward).item():.4f}, Term={terminated}, Trunc={truncated}")
            print(f"--- END ENV_STEP (ID: {self.env_id}, TickStep: {self.current_step-1}) ---")
        
        if self.custom_print_render == "human": self.render(action_taken=discrete_action, profit_target_param=profit_target_param)
        
        return observation, np.nan_to_num(reward).item(), terminated, truncated, info

    def _liquidate_position(self, price, trade_type_label, step_label):
        if not self.position_open: return 0,0,0
        revenue = self.position_volume * price
        commission = revenue * self.commission_pct
        net_revenue = revenue - commission
        cost_basis = self.entry_price * self.position_volume
        pnl = net_revenue - cost_basis
        self.current_balance += net_revenue
        self.trade_history.append({
            'step': step_label, 
            'time': self.tick_df.index[step_label].isoformat(),
            'type': trade_type_label, 'price': price,
            'volume': self.position_volume, 'balance': self.current_balance,
            'pnl': pnl, 'commission': commission, 'equity': self.current_balance,
            'profit_target_aimed': self.current_desired_profit_target,
            'pnl_ratio_achieved': (price / self.entry_price -1) if self.entry_price > 1e-9 else 0
        })
        if self.log_level in ["normal", "detailed"]:
            print(f"ENV_LIQUIDATE (ID: {self.env_id}, Step: {step_label}): {trade_type_label} @ {price:.4f}, Vol: {self.position_volume:.4f}, PnL: {pnl:.2f}")
        self.position_open = False; self.position_volume = 0.0; self.entry_price = 0.0; self.current_desired_profit_target = 0.0
        return net_revenue, commission, pnl

    def render(self, action_taken=None, profit_target_param=None):
        if self.custom_print_render == 'human' and (self.current_step % 10 == 0 or action_taken !=0 or self.current_step >= self.end_step):
            info = self._get_info()
            action_str = self.ACTION_MAP.get(action_taken, "N/A") if action_taken is not None else ""
            profit_target_str = f", TgtP: {profit_target_param:.3f}" if profit_target_param is not None else ""
            print(f"RENDER Tick (data from tick {info['current_step']-1}), Act: {action_str}{profit_target_str}, TickPx: {info['current_tick_price']:.3f}, "
                  f"Bal: {info['current_balance']:.2f}, Eq: {info['equity']:.2f}, Pos: {info['position_open']}")

    def close(self):
        if self.log_level in ["normal", "detailed"]: 
            print(f"ENV_CLOSE (ID: {self.env_id}): Environment instance closed.")
        pass

if __name__ == '__main__':
    print("--- Testing SimpleTradingEnv (Tick-based actions) ---")
    SCRIPT_LOG_LEVEL = "normal" 

    try:
        # For standalone test, create mock data
        # Mock Tick Data (e.g., 1000 ticks)
        num_ticks = 1000
        base_price = 20000
        tick_dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=num_ticks, freq='ms')
        mock_tick_df = pd.DataFrame(index=tick_dates)
        mock_tick_df['Price'] = base_price + np.cumsum(np.random.randn(num_ticks) * 0.1)
        mock_tick_df['Quantity'] = np.random.rand(num_ticks) * 10 + 1
        mock_tick_df['IsBuyerMaker'] = np.random.choice([True, False], size=num_ticks)


        # Mock K-line Data (e.g., 1H candles covering the tick data period)
        kline_start_time = tick_dates[0] - pd.Timedelta(hours=DEFAULT_ENV_CONFIG["kline_window_size"] + 5)
        num_klines = (tick_dates[-1] - kline_start_time).total_seconds() / 3600 + DEFAULT_ENV_CONFIG["kline_window_size"]
        num_klines = int(max(num_klines, DEFAULT_ENV_CONFIG["kline_window_size"] + 10))

        kline_idx = pd.date_range(start=kline_start_time, periods=num_klines, freq='h')
        mock_kline_df = pd.DataFrame(index=kline_idx)
        mock_kline_df['Open'] = base_price + np.random.randn(num_klines) * 10
        mock_kline_df['High'] = mock_kline_df['Open'] + np.random.rand(num_klines) * 20
        mock_kline_df['Low'] = mock_kline_df['Open'] - np.random.rand(num_klines) * 20
        mock_kline_df['Close'] = mock_kline_df['Open'] + (np.random.rand(num_klines) - 0.5) * 20
        mock_kline_df['Volume'] = np.random.rand(num_klines) * 1000 + 100
        
        # Mock TAs (consistent with how utils.py adds them now)
        mock_kline_df['SMA_10'] = mock_kline_df['Close'].rolling(window=10, min_periods=1).mean()
        mock_kline_df['RSI_7'] = np.random.rand(num_klines) * 100 # Mock RSI_7
        mock_kline_df['RSI_14'] = np.random.rand(num_klines) * 100 # Mock RSI_14
        mock_kline_df['ATR'] = np.random.rand(num_klines) * 50 # Mock ATR
        mock_kline_df['MACD'] = np.random.randn(num_klines) * 10 # Mock MACD
        mock_kline_df['ADX'] = np.random.rand(num_klines) * 100 # Mock ADX
        mock_kline_df['STOCH_K'] = np.random.rand(num_klines) * 100 # Mock STOCH_K
        mock_kline_df['BBANDS_Upper'] = mock_kline_df['Close'] + np.random.rand(num_klines) * 10 # Mock BBANDS_Upper
        mock_kline_df['AD'] = np.random.randn(num_klines) * 10000 # Mock AD
        mock_kline_df['OBV'] = np.random.randn(num_klines) * 100000 # Mock OBV

        # Mock Candlestick Patterns (0, 100, -100)
        mock_kline_df['CDLDOJI'] = np.random.choice([0, 100, -100], size=num_klines)
        mock_kline_df['CDLHAMMER'] = np.random.choice([0, 100, -100], size=num_klines)
        mock_kline_df['CDLENGULFING'] = np.random.choice([0, 100, -100], size=num_klines)
        mock_kline_df['CDLMORNINGSTAR'] = np.random.choice([0, 100], size=num_klines) # Morning Star is bullish

        # Ensure all columns exist before passing to env
        mock_kline_df.bfill(inplace=True)
        mock_kline_df.ffill(inplace=True)
        mock_kline_df.fillna(0, inplace=True)


        test_env_config = DEFAULT_ENV_CONFIG.copy()
        test_env_config["log_level"] = SCRIPT_LOG_LEVEL
        test_env_config["custom_print_render"] = "human" if SCRIPT_LOG_LEVEL == "detailed" else "none"
        # Update kline_price_features for the test env instance to match config.yaml
        test_env_config["kline_price_features"] = [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_10", "RSI_7", "RSI_14", "ATR", "MACD", "ADX", "STOCH_K", "BBANDS_Upper", "AD", "OBV",
            "CDLDOJI", "CDLHAMMER", "CDLENGULFING", "CDLMORNINGSTAR" # Add mock patterns
        ]


        print(f"Mock tick_df shape: {mock_tick_df.shape}, Index: {mock_tick_df.index.name}, Time range: {mock_tick_df.index.min()} to {mock_tick_df.index.max()}")
        print(f"Mock kline_df_with_ta shape: {mock_kline_df.shape}, Index: {mock_kline_df.index.name}, Time range: {mock_kline_df.index.min()} to {mock_kline_df.index.max()}")


        env = SimpleTradingEnv(tick_df=mock_tick_df, kline_df_with_ta=mock_kline_df, config=test_env_config)

        from gymnasium.utils.env_checker import check_env
        try:
            print("Checking environment compatibility with Gymnasium...")
            check_env(env, warn=True, skip_render_check=True)
            print("Gymnasium Env check passed (or warnings issued)!")
        except Exception as e:
            print(f"Gymnasium Env check FAILED: {e}")
            traceback.print_exc()

        obs, info = env.reset()
        print(f"Initial Observation Shape: {obs.shape if obs is not None else 'None'}")
        print(f"Initial Info (after reset): {json.dumps(info, indent=2)}")

        terminated, truncated = False, False
        total_reward_test = 0.0
        max_steps_for_test = min(len(mock_tick_df) - env.start_step -1, 500) 

        print(f"\nRunning test episode for max {max_steps_for_test} tick-steps with random actions...")
        for step_num in range(max_steps_for_test):
            action_sample_tuple = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action_sample_tuple)
            total_reward_test += reward
            
            if SCRIPT_LOG_LEVEL == "detailed" or (step_num + 1) % 50 == 0:
                action_desc = env.ACTION_MAP.get(action_sample_tuple[0])
                profit_tgt_desc = action_sample_tuple[1][0]
                print(f"  Test TickStep: {step_num+1}, Action: ({action_desc}, TgtP: {profit_tgt_desc:.3f}), Reward: {reward:.4f}, Term: {terminated}, Trunc: {truncated}, Eq: {info.get('equity',0):.2f}")
            
            if terminated or truncated:
                print(f"Episode finished at test tick-step {step_num+1}. Reason: Terminated={terminated}, Truncated={truncated}")
                break
        
        final_step_count = info.get('current_step', env.start_step) - env.start_step if info else max_steps_for_test
        print(f"\nTest run finished. Total tick-steps taken: {final_step_count}. Total reward: {total_reward_test:.2f}")
        print("Last 5 trades from history (if any):")
        for t_entry in env.trade_history[-5:]: 
            print(f"  {json.dumps({k: (f'{v:.4f}' if isinstance(v, float) else v) for k,v in t_entry.items()}, indent=None)}")
        env.close()

    except Exception as e:
        print(f"Error during SimpleTradingEnv (tick-based) test: {e}")
        traceback.print_exc()