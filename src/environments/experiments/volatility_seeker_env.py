# src/environments/experiments/volatility_seeker_env.py
import numpy as np
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

# --- Configuration for the Volatility Seeker Environment ---
VOLATILITY_SEEKER_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "reward_scaling_factor": 0.01, # Controls the steepness of the convex reward curve
    "reward_buy_open": 0.0,        # No reward for just opening a position
    "penalty_incorrect_action": -0.01 # Penalty for invalid actions
}

class VolatilitySeekerEnv(SimpleTradingEnv):
    """
    An environment that uses a convex reward strategy to encourage volatility.
    - When HOLDing a position, the reward increases quadratically as the price
      moves away from the entry price. This encourages seeking large price swings.
    - When SELLing to close a position, the reward is the realized profit or loss (PnL).
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        # Start with the environment's specific defaults
        final_config = VOLATILITY_SEEKER_CONFIG.copy()

        # If a config from YAML is provided, merge it, allowing it to override
        if config:
            final_config.update(config)
            
        super().__init__(tick_df, kline_df_with_ta, config=final_config)

    def _calculate_reward(self, discrete_action, price_for_action):
        # --- State: FLAT (No position is open) ---
        if not self.position_open:
            if discrete_action == 1: # BUY
                return self.config["reward_buy_open"]
            else: # Incorrect SELL or HOLD
                return self.config["penalty_incorrect_action"]

        # --- State: IN POSITION (A position is open) ---
        else:
            if discrete_action == 0: # HOLD
                # For holding, reward grows quadratically with distance from entry price
                price_distance = abs(price_for_action - self.entry_price)
                reward = self.config["reward_scaling_factor"] * (price_distance ** 2)
                return reward
            
            elif discrete_action == 2: # SELL
                # For selling, reward is the realized profit and loss
                realized_pnl = (price_for_action - self.entry_price) * self.position_volume
                return realized_pnl

            else: # Incorrect BUY
                return self.config["penalty_incorrect_action"]