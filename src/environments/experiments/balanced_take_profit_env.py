# src/environments/experiments/balanced_take_profit_env.py
from src.environments.base_env import SimpleTradingEnv

class BalancedTakeProfitEnv(SimpleTradingEnv):
    """
    This is the corrected and complete environment to encourage selling.
    - It provides a small reward for initiating a position.
    - It rewards any profitable sell with a scaled reward.
    - It provides ZERO reward for holding a profitable position to force a sell decision.
    """
    def _calculate_reward(self, discrete_action, price_for_action, was_position_open, is_position_now_open, entry_price_before_action):
        
        # --- Reward for a successful BUY action ---
        if discrete_action == 1 and not was_position_open and is_position_now_open:
            return 0.1 # Incentive to enter the market

        # --- Reward for a successful SELL action ---
        elif discrete_action == 2 and was_position_open and not is_position_now_open:
            profit_target_pct = self.config.get("profit_target_pct", 0.01)
            actual_profit_ratio = (price_for_action / entry_price_before_action - 1) if entry_price_before_action > 0 else 0
            
            if actual_profit_ratio > 0:
                # Scaled Reward: Starts at 0.2 for minimal profit and scales up towards 1.5 for hitting the target.
                scaling_factor = min(actual_profit_ratio / profit_target_pct, 1.0)
                return 0.2 + (scaling_factor * 1.3)
            else:
                # For a losing sell, use the proportional penalty
                loss_penalty_factor = self.config.get("loss_penalty_factor", 50)
                return actual_profit_ratio * loss_penalty_factor

        # --- Reward for a HOLD action ---
        elif discrete_action == 0 and was_position_open:
            pnl_ratio = (price_for_action / entry_price_before_action - 1) if entry_price_before_action > 0 else 0
            
            if pnl_ratio > 0:
                # THE CRITICAL CHANGE: No reward for passively holding a winning position.
                return 0.0
            else:
                # Keep the penalty for holding a losing position.
                return self.config.get("penalty_hold_losing_position", -0.0005)

        # For all other cases (holding flat, failed trades), use the base logic.
        return super()._calculate_reward(discrete_action, price_for_action, was_position_open, is_position_now_open, entry_price_before_action)