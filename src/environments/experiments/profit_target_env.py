# src/environments/experiments/profit_target_env.py
import pandas as pd
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

PROFIT_TARGET_EXP_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "profit_target_pct": 0.002,
    "reward_factor_above_target": 50.0,
    "penalty_factor_below_target": 100.0,
    "reward_trade_completion_bonus": 1.0,
}

class ProfitTargetEnv(SimpleTradingEnv):
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        env_config = {**PROFIT_TARGET_EXP_CONFIG, **(config if config else {})}
        super().__init__(tick_df, kline_df_with_ta, env_config)

    def step(self, action_tuple):
        # First, get the basic reward (including holding penalties) from the parent
        obs, reward, terminated, truncated, info = super().step(action_tuple)

        # Only modify the reward if a sell just happened
        if info.get('position_open') is False and self.position_open is True: # A sell just occurred
             # The parent reward is the raw PnL. We will replace it.
             pnl = reward 
             if pnl <= 0:
                 reward = pnl # Keep the raw PnL for losses
             else:
                price = info['current_tick_price']
                actual_pnl_ratio = (price / self.entry_price - 1) if self.entry_price > 0 else 0
                performance = actual_pnl_ratio - self.config['profit_target_pct']
                reward = performance * (self.config['reward_factor_above_target'] if performance >= 0 else self.config['penalty_factor_below_target'])
                reward += self.config["reward_trade_completion_bonus"]

        return obs, reward, terminated, truncated, info