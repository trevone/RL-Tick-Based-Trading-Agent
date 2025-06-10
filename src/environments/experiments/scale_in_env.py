# src/environments/experiments/scale_in_env.py
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

SCALE_IN_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "reward_additional_buy": 0.1,    # Reward for adding to a profitable position
    "penalty_buy_losing": -0.2,      # Penalize adding to a losing position
    "reward_successful_close": 1.5,
}

class ScaleInEnv(SimpleTradingEnv):
    """
    An environment to teach the agent to "scale into" a position by
    making multiple buys to increase its size and average its entry price.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        final_config = SCALE_IN_CONFIG.copy()
        if config:
            final_config.update(config)
        super().__init__(tick_df, kline_df_with_ta, config=final_config)

    def _calculate_reward(self, discrete_action, price_for_action):
        # --- Action: BUY ---
        if discrete_action == 1:
            if self.position_open:
                # Reward/penalize based on whether we are adding to a winner or loser
                if price_for_action >= self.entry_price:
                    return self.config["reward_additional_buy"]
                else:
                    return self.config["penalty_buy_losing"]
            else:
                return 0.0 # No special reward for the first buy

        # --- Action: SELL ---
        elif discrete_action == 2:
            if self.position_open:
                pnl = (price_for_action - self.entry_price) * self.position_volume
                return self.config["reward_successful_close"] + pnl
            else:
                return -1.0 # Penalty for selling with no position

        # --- Action: HOLD ---
        else:
            return -0.01 # Small penalty to discourage inaction