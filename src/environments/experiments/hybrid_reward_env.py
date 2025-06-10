# src/environments/experiments/hybrid_reward_env.py
import numpy as np
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

# --- Configuration for the Hybrid Reward Environment ---
HYBRID_REWARD_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "reward_at_entry": 1.0,      # The maximum reward for holding at the entry price
    "decay_rate": 0.1,           # Controls how quickly the hold reward decreases
    "reward_buy_open": 0.0,      # No reward for just opening a position
    "penalty_incorrect_action": -0.01 # Penalty for invalid actions
}

class HybridRewardEnv(SimpleTradingEnv):
    """
    An environment that uses a hybrid reward strategy:
    - When HOLDing a position, the reward decays exponentially as the price
      moves away from the entry price. This encourages keeping the trade in a
      certain range.
    - When SELLing to close a position, the reward is the realized profit or loss (PnL)
      of the trade. This encourages profitable exits.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        # Start with the environment's specific defaults
        final_config = HYBRID_REWARD_CONFIG.copy()

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
                # For holding, reward decays based on distance from entry price
                price_distance = abs(price_for_action - self.entry_price)
                reward = self.config["reward_at_entry"] * np.exp(-self.config["decay_rate"] * price_distance)
                return reward
            
            elif discrete_action == 2: # SELL
                # For selling, reward is the realized profit and loss
                realized_pnl = (price_for_action - self.entry_price) * self.position_volume
                return realized_pnl

            else: # Incorrect BUY
                return self.config["penalty_incorrect_action"]