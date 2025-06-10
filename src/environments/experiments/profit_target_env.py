# src/environments/experiments/profit_target_env.py
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

# --- Configuration for the Profit Target Experiment ---
PROFIT_TARGET_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "profit_target_pct": 0.01,  # Target a 1% profit on trades.
    
    # --- Rewards and Penalties for this specific strategy ---
    "reward_hit_target": 2.0,           # High reward for selling at or above the target.
    "penalty_sell_miss": -1.0,          # Penalty for selling before hitting the target.
    "reward_hold_profitable": 0.01,     # Small reward for holding a profitable position below the target.
    "penalty_hold_past_target": -0.25,  # **Crucial**: Penalize holding when profit could be taken.
    "penalty_hold_unprofitable": -0.05, # Standard penalty for holding a losing position.
}

class ProfitTargetEnv(SimpleTradingEnv):
    """
    An experimental environment designed to teach the agent to sell
    once a specific profit target is reached.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        # Start with the environment's specific defaults
        final_config = PROFIT_TARGET_CONFIG.copy()

        # If a config from YAML is provided, merge it, allowing it to override
        if config:
            final_config.update(config)
            
        super().__init__(tick_df, kline_df_with_ta, config=final_config)

    def _calculate_reward(self, discrete_action, price_for_action):
        """
        Calculates rewards based on achieving a profit target.
        """
        profit_target_pct = self.config["profit_target_pct"]

        # --- State: FLAT (No position is open) ---
        if not self.position_open:
            if discrete_action == 1: # BUY
                return self.config.get("reward_buy_open", 0.0)
            else: # Incorrect SELL or HOLD
                return self.config.get("penalty_incorrect_action", -0.1)

        # --- State: IN POSITION (A position is open) ---
        else:
            # Calculate the current unrealized profit percentage
            current_profit_pct = (price_for_action - self.entry_price) / self.entry_price

            # --- Action: HOLD ---
            if discrete_action == 0:
                if current_profit_pct >= profit_target_pct:
                    # Penalize holding past the target. This teaches the agent to take profits.
                    return self.config["penalty_hold_past_target"]
                elif current_profit_pct > 0:
                    # Encourage holding while the trade is profitable but below the target.
                    return self.config["reward_hold_profitable"]
                else:
                    # Standard penalty for holding a losing position.
                    return self.config["penalty_hold_unprofitable"]
            
            # --- Action: SELL ---
            elif discrete_action == 2:
                if current_profit_pct >= profit_target_pct:
                    # High reward for taking profit at or above the target.
                    return self.config["reward_hit_target"] + current_profit_pct
                else:
                    # Penalize for selling before the target was reached (i.e., at a loss or small gain).
                    # The PnL is added, which will be negative for a loss.
                    return self.config["penalty_sell_miss"] + current_profit_pct
            
            # --- Action: BUY (Incorrect) ---
            else:
                return self.config["penalty_incorrect_action"]