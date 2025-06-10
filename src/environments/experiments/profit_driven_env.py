from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

PROFIT_DRIVEN_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "penalty_incorrect_action": -0.01,
    "reward_buy_open": 0.0, # No reward for just opening, profit is the goal
}

class ProfitDrivenEnv(SimpleTradingEnv):
    """
    An environment where the reward is directly tied to the financial
    outcome of each action, both realized and unrealized.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        # Start with the environment's specific defaults
        final_config = PROFIT_DRIVEN_CONFIG.copy()

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
                # Reward is the change in unrealized PnL from the last step.
                unrealized_pnl_change = (price_for_action - self.last_price) * self.position_volume
                return unrealized_pnl_change
            
            elif discrete_action == 2: # SELL
                # Reward is the total realized PnL for the entire trade.
                realized_pnl = (price_for_action - self.entry_price) * self.position_volume
                return realized_pnl

            else: # Incorrect BUY
                return self.config["penalty_incorrect_action"]