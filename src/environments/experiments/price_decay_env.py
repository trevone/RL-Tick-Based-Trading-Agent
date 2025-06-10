# src/environments/experiments/price_decay_env.py
import numpy as np
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

# --- Configuration for the Price Decay Reward Environment ---
PRICE_DECAY_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "reward_at_entry": 1.0,         # The maximum reward when price is at the entry point
    "decay_rate": 0.1,              # Controls how quickly the reward decreases as price moves away
    "reward_buy_open": 0.0,         # No reward for just opening a position
    "penalty_incorrect_action": -0.01 # Penalty for invalid actions
}

class PriceDecayEnv(SimpleTradingEnv):
    """
    An environment where the reward is based on how close the current price is
    to the entry price of a trade. The reward is highest at the entry price and
    decays exponentially as the price moves away in either direction.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        # Start with the environment's specific defaults
        final_config = PRICE_DECAY_CONFIG.copy()

        # If a config from YAML is provided, merge it, allowing it to override
        if config:
            final_config.update(config)
            
        super().__init__(tick_df, kline_df_with_ta, config=final_config)

    def _calculate_reward(self, discrete_action, price_for_action):
        # --- State: FLAT (No position is open) ---
        if not self.position_open:
            if discrete_action == 1: # BUY
                # No immediate reward for buying, the reward comes from how the price behaves after entry
                return self.config["reward_buy_open"]
            else: # Incorrect SELL or HOLD
                return self.config["penalty_incorrect_action"]

        # --- State: IN POSITION (A position is open) ---
        else:
            if discrete_action == 0: # HOLD
                # Calculate the distance from the entry price
                price_distance = abs(price_for_action - self.entry_price)
                
                # Calculate the reward using an exponential decay function
                reward = self.config["reward_at_entry"] * np.exp(-self.config["decay_rate"] * price_distance)
                return reward
            
            elif discrete_action == 2: # SELL
                # The reward for selling is also based on the exit price's proximity to the entry price.
                price_distance = abs(price_for_action - self.entry_price)
                reward = self.config["reward_at_entry"] * np.exp(-self.config["decay_rate"] * price_distance)
                return reward

            else: # Incorrect BUY
                return self.config["penalty_incorrect_action"]