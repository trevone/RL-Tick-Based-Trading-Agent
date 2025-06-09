# src/environments/experiments/proactive_sell_env.py
from src.environments.base_env import SimpleTradingEnv

class ProactiveSellEnv(SimpleTradingEnv):
    """
    An environment designed to combat the "hold indefinitely" problem.
    - Holding a profitable position yields a neutral (zero) reward to encourage a decision.
    - The reward for a profitable sell is scaled, providing a smoother gradient.
    """
    def _calculate_reward(self, discrete_action, price_for_action, was_position_open, is_position_now_open, entry_price_before_action):
        # --- Reward for BUY action ---
        if discrete_action == 1 and not was_position_open and is_position_now_open:
            return 0.1 # Keep the incentive to enter the market

        # --- Reward for SELL action ---
        elif discrete_action == 2 and was_position_open and not is_position_now_open:
            profit_target_pct = self.config.get("profit_target_pct", 0.01)
            actual_profit_ratio = (price_for_action / entry_price_before_action - 1) if entry_price_before_action > 0 else 0
            
            if actual_profit_ratio > 0:
                # Scaled Reward: Starts at 0.2 for minimal profit and scales up to 1.5 for hitting the target.
                scaling_factor = min(actual_profit_ratio / profit_target_pct, 1.0)
                reward = 0.2 + (scaling_factor * 1.3)
                return reward
            else:
                return -1.0 # Strong penalty for any loss

        # --- CRITICAL FIX: Reward for HOLD action ---
        elif discrete_action == 0 and was_position_open:
            pnl_ratio = (price_for_action / entry_price_before_action - 1) if entry_price_before_action > 0 else 0
            
            if pnl_ratio > 0:
                # This is the key change: No reward for passively holding a profitable position.
                # This forces the agent to seek the reward by selling.
                return 0.0
            else:
                # Keep the penalty for holding a losing position.
                return self.config.get("penalty_hold_losing_position", -0.0005)

        # For all other cases (e.g., holding flat, failed trades), use the base logic.
        return super()._calculate_reward(discrete_action, price_for_action, was_position_open, is_position_now_open, entry_price_before_action)