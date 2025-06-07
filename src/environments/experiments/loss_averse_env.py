# src/environments/experiments/loss_averse_env.py
import numpy as np
import pandas as pd
from ..base_env import SimpleTradingEnv

class LossAverseEnv(SimpleTradingEnv):
    """
    An experimental environment that inherits from SimpleTradingEnv but
    implements a stronger penalty for holding losing positions to
    encourage the agent to cut losses more quickly.
    """
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        # Initialize the parent class (our new base_env)
        super().__init__(tick_df, kline_df_with_ta, config)
        
        # This experiment uses a custom, stronger penalty
        self.loss_aversion_penalty = -0.01 # A stronger penalty than the default
        
        if self.log_level in ["normal", "detailed"]:
            print(f"--- INITIALIZING LOSS AVERSE EXPERIMENTAL ENV (ID: {self.env_id}) ---")
            print(f"  Custom Loss Aversion Penalty: {self.loss_aversion_penalty}")

    def step(self, action_tuple):
        # First, get the standard reward and results by calling the parent's step method
        # This is a clean way to build on top of existing logic
        obs, reward, terminated, truncated, info = super().step(action_tuple)
        
        discrete_action, _ = action_tuple
        if isinstance(discrete_action, np.integer): discrete_action = int(discrete_action)

        # --- CUSTOM EXPERIMENTAL LOGIC ---
        # If the agent chose to hold a position that is currently losing money,
        # we add our extra, stronger penalty.
        if discrete_action == 0 and self.position_open:
            current_price = info.get('current_tick_price', self.entry_price)
            if current_price < self.entry_price:
                reward += self.loss_aversion_penalty # Apply the extra penalty
        # --- END CUSTOM LOGIC ---

        return obs, reward, terminated, truncated, info