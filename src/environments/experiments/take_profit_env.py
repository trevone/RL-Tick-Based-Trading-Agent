# src/environments/experiments/take_profit_env.py
from src.environments.base_env import SimpleTradingEnv

class TakeProfitEnv(SimpleTradingEnv):
    """
    An environment that rewards the agent for selling at a pre-defined profit target.
    This version inherits from the refactored SimpleTradingEnv and only overrides
    the reward calculation logic.
    """
    def _calculate_reward(self, discrete_action, price_for_action):
        """
        Overrides the base reward calculation.
        """
        # We only need to define custom logic for the SELL action.
        if discrete_action == 2:
            # Check if a position was open to sell.
            # Note: The parent 'step' method handles the actual selling.
            # Here, we just define the reward for that action.
            if self.position_open: # This check is implicitly about the state *before* the sell
                profit_target_pct = self.config.get("profit_target_pct", 0.01)
                actual_profit_ratio = (price_for_action / self.entry_price - 1)
                
                if actual_profit_ratio >= profit_target_pct:
                    return 1.5  # Strong reward for meeting target
                elif 0 < actual_profit_ratio < profit_target_pct:
                    return -0.25 # Penalty for selling too early
                else: # Loss
                    return -1.0 # Strong penalty for any loss
        
        # For all other actions (BUY, HOLD, or failed SELL),
        # let the base class calculate the reward using its default logic.
        return super()._calculate_reward(discrete_action, price_for_action)