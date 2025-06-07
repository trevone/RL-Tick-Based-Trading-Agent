# src/environments/experiments/profit_target_env.py
import numpy as np
import pandas as pd
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

PROFIT_TARGET_EXP_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "profit_target_pct": 0.002,
    "reward_factor_above_target": 50.0,
    "penalty_factor_below_target": 100.0,
    "penalty_loss_trade_factor": 1.0,
    "reward_trade_completion_bonus": 1.0,
    "penalty_hold_losing_position": -0.0005,
    "penalty_hold_flat_position": -0.0001,
}

class ProfitTargetEnv(SimpleTradingEnv):
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        env_config = {**PROFIT_TARGET_EXP_CONFIG, **(config if config else {})}
        super().__init__(tick_df, kline_df_with_ta, env_config)
        self.current_desired_profit_target = 0.0

    def step(self, action_tuple):
        discrete_action, _ = action_tuple
        reward, terminated, truncated = 0.0, False, False
        price = self.decision_prices[self.current_step]

        if discrete_action == 1 and not self.position_open:
            super().step(action_tuple) # Let parent handle the buy
        elif discrete_action == 2 and self.position_open:
            revenue = self.position_volume * price * (1 - self.commission_pct)
            pnl = revenue - (self.position_volume * self.entry_price)
            self.current_balance += revenue
            actual_pnl_ratio = (price / self.entry_price - 1) if self.entry_price > 0 else 0
            
            if pnl <= 0:
                reward = pnl * self.config["penalty_loss_trade_factor"]
            else:
                performance = actual_pnl_ratio - self.config['profit_target_pct']
                reward = performance * (self.config['reward_factor_above_target'] if performance >= 0 else self.config['penalty_factor_below_target'])
            reward += self.config["reward_trade_completion_bonus"]
            
            self.position_open, self.position_volume, self.entry_price = False, 0.0, 0.0
        elif discrete_action == 0:
            if self.position_open and price < self.entry_price: reward += self.config["penalty_hold_losing_position"]
            elif not self.position_open: reward += self.config["penalty_hold_flat_position"]

        self.current_step += 1
        equity = self.current_balance + (self.position_volume * price if self.position_open else 0)
        if equity < self.catastrophic_loss_limit: terminated = True; reward += self.config["penalty_catastrophic_loss"]
        if self.current_step > self.end_step: truncated = True
        if (terminated or truncated) and self.position_open:
            self.current_balance += self.position_volume * price * (1 - self.commission_pct)
            self.position_open = False
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()