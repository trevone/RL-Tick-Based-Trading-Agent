# src/environments/experiments/loss_averse_env.py
import pandas as pd
from .profit_target_env import ProfitTargetEnv

class LossAverseEnv(ProfitTargetEnv):
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        super().__init__(tick_df, kline_df_with_ta, config)
        self.loss_aversion_penalty = -0.01

    def step(self, action_tuple):
        obs, reward, terminated, truncated, info = super().step(action_tuple)
        discrete_action, _ = action_tuple

        if discrete_action == 0 and self.position_open:
            current_price = info.get('current_tick_price', self.entry_price)
            if current_price < self.entry_price:
                reward += self.loss_aversion_penalty
        return obs, reward, terminated, truncated, info