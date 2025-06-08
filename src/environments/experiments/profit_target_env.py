# src/environments/experiments/profit_target_env.py
import pandas as pd
import numpy as np # Make sure numpy is imported at the top for np.clip

from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

PROFIT_TARGET_EXP_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "profit_target_pct": 0.001, # This is the target profit percentage (e.g., 0.1%)
}

class ProfitTargetEnv(SimpleTradingEnv):
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        env_config = {**PROFIT_TARGET_EXP_CONFIG, **(config if config else {})}
        super().__init__(tick_df, kline_df_with_ta, env_config)

    def step(self, action_tuple):
        price = self.decision_prices[self.current_step]
        was_position_open_before_action = self.position_open
        entry_price_before_action = self.entry_price
        position_volume_before_action = self.position_volume

        obs, reward_from_parent, terminated, truncated, info = super().step(action_tuple)
        
        current_reward = reward_from_parent # Start with base reward (for Buy/Hold/Failed Sell)

        # Apply custom proportional reward ONLY if a successful SELL just occurred
        if info.get('position_open') is False and was_position_open_before_action is True:
            price = info['current_tick_price']
            
            # Calculate actual PnL ratio (percentage gain/loss relative to entry price)
            actual_pnl_ratio = (price / entry_price_before_action - 1) if entry_price_before_action > 1e-9 else 0.0

            # Calculate proportional component based on your desired mapping:
            # Maps actual_pnl_ratio=0 to -1 reward component
            # Maps actual_pnl_ratio=profit_target_pct to +1 reward component
            
            target_pct = self.config['profit_target_pct']
            
            if target_pct <= 1e-9: # Handle cases where target_pct is zero or extremely small to avoid division by zero
                proportional_component = -1.0 # If target is 0, and PnL is 0, reward is -1. If PnL is negative, it's very negative.
            else:
                proportional_component = (2 / target_pct) * actual_pnl_ratio - 1
            
            # --- CHANGE STARTS HERE ---
            # Removed all clipping to allow proportional_component to go arbitrarily high or low.
            # --- CHANGE ENDS HERE ---

            # Final reward for successful sell: Base reward (+1) + Proportional component
            current_reward = reward_from_parent + proportional_component

        # --- Common logic for advancing step and checking for termination ---
        
        terminated, final_reward, truncated = self._handle_episode_termination_and_cleanup(terminated, current_reward, truncated )
        
        return obs, current_reward, terminated, truncated, info