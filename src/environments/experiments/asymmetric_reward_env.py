# src/environments/experiments/asymmetric_reward_env.py
import numpy as np
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

# --- Configuration for the Asymmetric Reward Environment ---
ASYMMETRIC_REWARD_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "profit_scaling_factor": 0.01, # Scales the convex reward for profits
    "loss_scaling_factor": 0.1,    # Scales the concave penalty for losses
    "reward_buy_open": 0.0,
    "penalty_incorrect_action": -0.01
}

class AsymmetricRewardEnv(SimpleTradingEnv):
    """
    An environment with an asymmetric reward function to encourage
    cutting losses and letting profits run.
    - When HOLDing a profitable position, the reward is CONVEX (accelerating gain).
    - When HOLDing a losing position, the penalty is CONCAVE (decelerating loss).
    - When SELLing, the reward is the standard realized PnL.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        final_config = ASYMMETRIC_REWARD_CONFIG.copy()
        if config:
            final_config.update(config)
        super().__init__(tick_df, kline_df_with_ta, config=final_config)

    def _calculate_reward(self, discrete_action, price_for_action):
        if not self.position_open:
            if discrete_action == 1: # BUY
                return self.config["reward_buy_open"]
            else: # Incorrect SELL or HOLD
                return self.config["penalty_incorrect_action"]
        else:
            if discrete_action == 0: # HOLD
                if price_for_action > self.entry_price:
                    # Convex reward for profits (quadratic)
                    profit_distance = price_for_action - self.entry_price
                    reward = self.config["profit_scaling_factor"] * (profit_distance ** 2)
                    return reward
                elif price_for_action < self.entry_price:
                    # Concave penalty for losses (using square root)
                    loss_distance = self.entry_price - price_for_action
                    # The penalty is negative
                    penalty = -self.config["loss_scaling_factor"] * np.sqrt(loss_distance)
                    return penalty
                else:
                    # No reward or penalty if the price is exactly at entry
                    return 0.0
            
            elif discrete_action == 2: # SELL
                # Reward for selling is the realized profit
                realized_pnl = (price_for_action - self.entry_price) * self.position_volume
                return realized_pnl

            else: # Incorrect BUY
                return self.config["penalty_incorrect_action"]