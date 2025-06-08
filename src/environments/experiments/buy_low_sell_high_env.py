# src/environments/experiments/buy_low_sell_high_env.py
import numpy as np
import pandas as pd
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

# --- Configuration for this specific experiment ---
BLSH_ENV_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    # The benchmark TA to use for the 'buy low, sell high' heuristic.
    # Make sure this feature is included in your environment's kline_price_features.
    "buy_low_sell_high_benchmark_ta": "SMA_20",

    # The multiplier to apply to a profitable trade's PnL if it was
    # executed well (bought below benchmark, sold above). 1.5 = a 50% bonus.
    "buy_low_sell_high_bonus_multiplier": 1.5,
}


class BuyLowSellHighEnv(SimpleTradingEnv):
    """
    An experimental environment that extends SimpleTradingEnv to explicitly reward
    the strategy of "buying low and selling high" relative to a benchmark
    technical indicator (e.g., a moving average).

    It gives a bonus multiplier to profitable trades that were initiated
    below the benchmark and closed above it.
    """
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        # Merge the specific config with any provided config
        env_config = {**BLSH_ENV_CONFIG, **(config if config else {})}
        super().__init__(tick_df, kline_df_with_ta, env_config)

        # Add a new state variable to store the benchmark's price at the time of purchase
        self.benchmark_price_at_buy = 0.0

    def reset(self, seed=None, options=None):
        # Reset the parent environment and also our new state variable
        observation, info = super().reset(seed=seed, options=options)
        self.benchmark_price_at_buy = 0.0
        return observation, info

    def step(self, action_tuple):
        """
        This method overrides the parent's step to implement the custom reward logic.
        It replicates the base logic but adds the 'buy low, sell high' check.
        """
        discrete_action, _ = action_tuple
        reward, terminated, truncated = 0.0, False, False
        price = self.decision_prices[self.current_step]
        
        # --- BUY ACTION ---
        if discrete_action == 1:
            if not self.position_open:
                cost = (self.initial_balance * self.base_trade_amount_ratio) * (1 + self.commission_pct)
                if self.current_balance >= cost:
                    self.position_volume = (self.initial_balance * self.base_trade_amount_ratio) / price
                    self.current_balance -= cost
                    self.position_open, self.entry_price = True, price

                    # --- NEW: STORE BENCHMARK PRICE ON BUY ---
                    benchmark_feature = self.config.get("buy_low_sell_high_benchmark_ta")
                    if benchmark_feature and benchmark_feature in self.kline_feature_arrays:
                        # Find the corresponding k-line index for the current tick
                        kline_idx = self.kline_df_with_ta.index.get_indexer([self.tick_df.index[self.current_step]], method='ffill')[0]
                        if kline_idx != -1:
                            self.benchmark_price_at_buy = self.kline_feature_arrays[benchmark_feature][kline_idx]
            else:
                # Use proportional penalty for trying to buy when a position is open
                current_position_value = self.position_volume * price
                penalty_pct = self.config.get("penalty_invalid_action_pct_of_value", 0)
                reward += current_position_value * penalty_pct

        # --- SELL ACTION ---
        elif discrete_action == 2:
            if self.position_open:
                revenue = self.position_volume * price * (1 - self.commission_pct)
                pnl = revenue - (self.position_volume * self.entry_price)
                reward = pnl  # Start with the raw PnL as the base reward

                # --- NEW: APPLY "GOOD BOY" BONUS ---
                if pnl > 0:  # Only apply the bonus to profitable trades
                    benchmark_feature = self.config.get("buy_low_sell_high_benchmark_ta")
                    if benchmark_feature and self.benchmark_price_at_buy > 0:
                        kline_idx = self.kline_df_with_ta.index.get_indexer([self.tick_df.index[self.current_step]], method='ffill')[0]
                        if kline_idx != -1:
                            benchmark_price_at_sell = self.kline_feature_arrays[benchmark_feature][kline_idx]
                            
                            # Check if the "buy low, sell high" condition is met
                            if self.entry_price < self.benchmark_price_at_buy and price > benchmark_price_at_sell:
                                bonus_multiplier = self.config.get("buy_low_sell_high_bonus_multiplier", 1.0)
                                reward *= bonus_multiplier  # Apply the bonus multiplier
                
                # Update balance and reset position state
                self.current_balance += revenue
                self.position_open, self.position_volume, self.entry_price = False, 0.0, 0.0
                self.benchmark_price_at_buy = 0.0  # Reset benchmark price after selling
            else:
                # Use proportional penalty for trying to sell when no position is open
                standard_trade_value = self.initial_balance * self.config.get("base_trade_amount_ratio", 0.02)
                penalty_pct = self.config.get("penalty_invalid_action_pct_of_value", 0)
                reward += standard_trade_value * penalty_pct

        # --- HOLD ACTION & REMAINDER OF STEP LOGIC (Unchanged from base_env) ---
        elif discrete_action == 0:
            # ... (proportional hold penalties from base_env) ...
            if self.position_open:
                if price < self.entry_price:
                    current_position_value = self.position_volume * price
                    penalty_pct = self.config.get("penalty_hold_losing_position_pct_of_value", 0)
                    reward += current_position_value * penalty_pct
            else:
                standard_trade_value = self.initial_balance * self.config.get("base_trade_amount_ratio", 0.02)
                penalty_pct = self.config.get("penalty_hold_flat_position_pct_of_trade_value", 0)
                reward += standard_trade_value * penalty_pct

        # --- Advance Step and Check for Episode End ---
        self.current_step += 1
        equity = self.current_balance + (self.position_volume * price if self.position_open else 0)
        
        if equity < self.catastrophic_loss_limit:
            terminated = True
            reward += self.initial_balance * self.config.get("penalty_catastrophic_loss_pct", 0)

        if self.current_step > self.end_step:
            truncated = True
        
        if (terminated or truncated) and self.position_open:
            self.current_balance += self.position_volume * price * (1 - self.commission_pct)
            self.position_open = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()