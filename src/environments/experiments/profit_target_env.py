# src/environments/experiments/profit_target_env.py
import pandas as pd
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

PROFIT_TARGET_EXP_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "profit_target_pct": 0.001,
    # This bonus is now a percentage of the initial trade value. 0.001 = 0.1% bonus.
    "reward_trade_completion_bonus_pct": 0.001, 
}

class ProfitTargetEnv(SimpleTradingEnv):
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        env_config = {**PROFIT_TARGET_EXP_CONFIG, **(config if config else {})}
        super().__init__(tick_df, kline_df_with_ta, env_config)

    def step(self, action_tuple):
        # Call parent's step to get basic env updates and initial reward.
        # The reward_from_parent could be -1 for incorrect BUY/SELL, or penalties for holding.
        obs, reward_from_parent, terminated, truncated, info = super().step(action_tuple)
        
        # Start with the reward from the parent (e.g., for incorrect actions or holding penalties).
        current_reward = reward_from_parent 

        # Apply custom proportional reward ONLY if a successful SELL just occurred.
        # 'info.get('position_open') is False' means position is now closed.
        # 'self.position_open is True' means position was open before this step.
        if info.get('position_open') is False and self.position_open is True:
            price = info['current_tick_price']
            
            # 1. Calculate the target price based on the desired profit_target_pct
            profit_target_price = self.entry_price * (1 + self.config['profit_target_pct'])
            
            # 2. Calculate the dollar amount of deviation from the target price
            deviation_dollars = (price - profit_target_price) * self.position_volume

            # 3. Normalize the dollar deviation by the initial balance.
            # This makes the reward proportional to your overall capital base.
            # Adding 1e-9 to avoid division by zero if initial_balance somehow becomes 0.
            normalized_deviation = deviation_dollars / (self.initial_balance + 1e-9)

            # 4. Scale the normalized deviation using configurable factors.
            # This brings the reward into a numerical range suitable for RL,
            # similar to your base env's +/-1 but reflecting magnitude.
            if normalized_deviation >= 0:
                current_reward += normalized_deviation * self.config['reward_scale_deviation_positive']
            else:
                current_reward += normalized_deviation * self.config['reward_scale_deviation_negative']
            
            # 5. Add a flat bonus for completing any trade. This encourages the agent to close positions.
            current_reward += self.config['reward_trade_completion_bonus_value']

            # 6. Add an additional penalty if the trade resulted in an actual financial loss (PnL < 0).
            # Calculate actual PnL in dollars
            pnl_dollars = (price - self.entry_price) * self.position_volume
            if pnl_dollars < 0:
                # This ensures a strong penalty for actual losses, even if the deviation from target was small.
                current_reward += (pnl_dollars / (self.initial_balance + 1e-9)) * self.config['reward_scale_actual_pnl_loss']

        return obs, current_reward, terminated, truncated, info