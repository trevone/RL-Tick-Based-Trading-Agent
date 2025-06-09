# src/environments/experiments/state_machine_test_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StateMachineTestEnv(gym.Env):
    """
    A minimal, "sterile" environment to test if an agent can learn a simple
    trading state machine without the noise of market data.
    - Observation: Just [position_open, last_action_was_trade]
    - Actions: Discrete(3) -> 0:Hold, 1:Buy, 2:Sell
    - Rewards: Rule-based, +1 for correct actions, -1 for violations.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        # We accept the data arguments to match the loader, but ignore them.
        super().__init__()
        
        # Action space is now a simple Discrete space.
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [is_position_open (0 or 1), last_action_was_trade (0 or 1)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        # Initialize state variables
        self.position_open = False
        self.last_action_was_trade = False
        self.current_step = 0
        self.max_steps = 200 # Define a max episode length

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position_open = False
        self.last_action_was_trade = False
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            1.0 if self.position_open else 0.0,
            1.0 if self.last_action_was_trade else 0.0
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        terminated = False
        
        was_position_open = self.position_open

        # --- RULE VIOLATIONS (-1 Penalty) ---
        if action == 1 and was_position_open:
            reward = -1.0
        elif action == 2 and not was_position_open:
            reward = -1.0
        elif action in [1, 2] and self.last_action_was_trade:
            reward = -1.0
        
        # --- CORRECT ACTIONS (+1 Reward) ---
        elif action == 1 and not was_position_open:
            reward = 1.0
            self.position_open = True
            self.last_action_was_trade = True
        elif action == 2 and was_position_open:
            reward = 1.0
            self.position_open = False
            self.last_action_was_trade = True

        # --- HOLD ACTION ---
        elif action == 0:
            if self.last_action_was_trade:
                reward = 0.5 # Good hold after a trade
                self.last_action_was_trade = False
            else:
                reward = -0.01 # Nudge to take action
        
        # --- Episode Termination ---
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True # Use terminated instead of truncated for a definitive end

        return self._get_obs(), reward, terminated, False, {}