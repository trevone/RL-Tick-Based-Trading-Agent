# src/environments/experiments/simple_sequence_env.py
from ..base_env import SimpleTradingEnv

class SimpleSequenceEnv(SimpleTradingEnv):
    """
    An environment that uses a very direct reward system to teach the
    fundamental trading sequence: BUY -> HOLD -> SELL.
    
    V2: Increases the penalty for inaction to force exploration.
    """
    def _calculate_reward(self, discrete_action, price_for_action):
        """
        Implements the user-specified reward structure, but with a stronger
        penalty for holding while flat to ensure the agent learns to trade.
        """
        # --- State: FLAT (No position is open) ---
        if not self.position_open:
            if discrete_action == 1: # BUY
                return 1.0  # Correct: Reward for buying
            elif discrete_action == 2: # SELL
                return -1.0 # Incorrect: Penalty for selling with no position
            else: # HOLD
                # *** KEY CHANGE HERE ***
                # Make the penalty for doing nothing much higher to force the agent
                # to explore and find the +1.0 reward for buying.
                return -0.25 

        # --- State: IN POSITION (A position is open) ---
        else:
            if discrete_action == 0: # HOLD
                # A small positive reward for correctly holding the position.
                return 0.1
            elif discrete_action == 1: # BUY
                return -1.0 # Incorrect: Penalty for buying again
            else: # SELL
                # Correct: Reward for selling to complete the sequence
                pnl = (price_for_action - self.entry_price) * self.position_volume
                return 1.0 + pnl