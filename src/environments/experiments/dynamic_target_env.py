# src/environments/experiments/dynamic_target_env.py
import pandas as pd
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

class DynamicTargetEnv(SimpleTradingEnv):
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        # We use the base config here, as this experiment defines its own logic
        super().__init__(tick_df, kline_df_with_ta, config)
        self.current_desired_profit_target = 0.0 # Agent-defined target

    def step(self, action_tuple):
        # This experiment fully overrides the step logic to implement its unique reward scheme
        discrete_action, profit_target_param_array = action_tuple
        agent_chosen_target = profit_target_param_array[0]
        reward, terminated, truncated = 0.0, False, False
        price = self.decision_prices[self.current_step]

        if discrete_action == 1 and not self.position_open:
            cost = (self.initial_balance * self.base_trade_amount_ratio) * (1 + self.commission_pct)
            if self.current_balance >= cost:
                self.position_volume = (self.initial_balance * self.base_trade_amount_ratio) / price
                self.current_balance -= cost
                self.position_open, self.entry_price = True, price
                self.current_desired_profit_target = agent_chosen_target # Store agent's target
        
        elif discrete_action == 2 and self.position_open:
            revenue = self.position_volume * price * (1 - self.commission_pct)
            pnl = revenue - (self.position_volume * self.entry_price)
            self.current_balance += revenue
            actual_pnl_ratio = (price / self.entry_price - 1) if self.entry_price > 0 else 0
            
            # The reward is judged against the agent's OWN chosen target
            performance = actual_pnl_ratio - self.current_desired_profit_target
            reward = performance # Simple reward: how much you beat your own goal
            
            self.position_open, self.position_volume, self.entry_price = False, 0.0, 0.0

        self.current_step += 1
        equity = self.current_balance + (self.position_volume * price if self.position_open else 0)
        if equity < self.catastrophic_loss_limit: terminated = True; reward += self.config["penalty_catastrophic_loss"]
        if self.current_step > self.end_step: truncated = True
        if (terminated or truncated) and self.position_open:
            self.current_balance += self.position_volume * price * (1 - self.commission_pct)
            self.position_open = False
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()