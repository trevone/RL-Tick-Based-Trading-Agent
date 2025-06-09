# src/environments/experiments/state_machine_env.py
import numpy as np
from src.environments.base_env import SimpleTradingEnv # Inherit to get obs/action space shapes

class StateMachineEnv(SimpleTradingEnv):
    """
    A from-scratch environment to teach the agent the trading state machine.
    This version uses a radically simplified observation space to force the agent
    to focus only on the core logic, ignoring market data entirely.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        # We call the parent to set up the observation and action space shapes
        # that the rest of the training pipeline expects.
        super().__init__(tick_df, kline_df_with_ta, config)
        
        # State variables for this specific task
        self.last_action = 0  # 0: Hold, 1: Buy, 2: Sell
        self.max_steps = 200 # Define a max episode length
        self.current_step_in_episode = 0
        if self.log_level in ["normal", "detailed"]:
            print("--- INITIALIZING StateMachineEnv (Simplified Observation) ---")

    def reset(self, seed=None, options=None):
        # We need to call the parent's reset to handle its internals
        super().reset(seed=seed)
        
        # Reset our specific state
        self.last_action = 0
        self.current_step_in_episode = 0
        self.position_open = False # Explicitly reset position status
        
        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        # Create a zero-filled observation of the correct shape that the PPO model expects.
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Now, we manually insert the ONLY information that matters for this task
        # into the last few slots of the observation array. The agent must learn
        # to ignore the zeros and focus on these critical state variables.
        is_position_open = 1.0 if self.position_open else 0.0
        last_action_was_trade = 1.0 if self.last_action in [1, 2] else 0.0
        
        # We use the third-to-last slot for the last action type itself, normalized.
        # This gives the agent a clear signal about its last move.
        # Hold: -1, Buy: 0, Sell: 1
        normalized_last_action = float(self.last_action - 1)
        
        obs[-3:] = [is_position_open, last_action_was_trade, normalized_last_action]
        return obs

    def step(self, action_tuple):
        # The agent still outputs a tuple, but we only care about the discrete part.
        discrete_action, _ = action_tuple
        
        was_position_open = self.position_open
        reward = 0.0
        
        # --- Rule Violations (Large Penalties) ---
        if discrete_action == 1 and was_position_open:
            reward = -1.0
        elif discrete_action == 2 and not was_position_open:
            reward = -1.0
        elif discrete_action == 2 and self.last_action == 1:
            reward = -1.0 # Penalize for selling immediately after buying

        # --- Correct Actions (Large Rewards) ---
        elif discrete_action == 1 and not was_position_open:
            reward = 1.0
            self.position_open = True
            self.last_action = 1
        elif discrete_action == 2 and was_position_open:
            reward = 1.0
            self.position_open = False
            self.last_action = 2

        # --- HOLD ACTION ---
        elif discrete_action == 0:
            if self.last_action in [1, 2]:
                reward = 0.5 # Reward for the mandatory hold after a trade
                self.last_action = 0
            else:
                reward = -0.01 # Small penalty to encourage eventual action
                self.last_action = 0
                
        self.current_step_in_episode += 1
        terminated = self.current_step_in_episode >= self.max_steps
        
        # We don't need to call the parent `_calculate_reward` as we've defined all logic here.
        # We also don't need to update portfolio value, as this env ignores profit.
        
        return self._get_observation(), reward, terminated, False, self._get_info()