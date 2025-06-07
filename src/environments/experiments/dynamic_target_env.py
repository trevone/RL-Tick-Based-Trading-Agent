# src/environments/experiments/dynamic_target_env.py
import numpy as np
import pandas as pd
from ..base_env import SimpleTradingEnv

class DynamicTargetEnv(SimpleTradingEnv):
    """
    An experimental environment that uses the agent's own continuously
    chosen profit target to calculate the reward.

    This tests if the agent can learn not only when to trade, but also
    what a realistic profit target should be in different market conditions.
    """
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        # Initialize the parent class (our current base_env)
        super().__init__(tick_df, kline_df_with_ta, config)
        
        if self.log_level in ["normal", "detailed"]:
            print(f"--- INITIALIZING DYNAMIC TARGET REWARD ENV (ID: {self.env_id}) ---")

    def step(self, action_tuple):
        """
        Overrides the step method to use the agent's chosen profit target for reward calculation.
        """
        discrete_action, profit_target_param_array = action_tuple
        if isinstance(discrete_action, np.integer): discrete_action = int(discrete_action)
        
        # This is the profit target chosen by the agent for this potential trade
        agent_chosen_target = profit_target_param_array[0]

        reward = 0.0
        terminated = False
        truncated = False
        price = self.decision_prices[self.current_step]

        # --- ACTION: BUY ---
        # On a buy, we store the agent's chosen profit target for this specific trade.
        if discrete_action == 1 and not self.position_open:
            trade_size = self.initial_balance * self.base_trade_amount_ratio
            self.position_volume = trade_size / price
            cost = self.position_volume * price * (1 + self.commission_pct)
            self.current_balance -= cost
            self.position_open = True
            self.entry_price = price
            # Store the agent's target for this trade
            self.current_desired_profit_target = agent_chosen_target

        # --- ACTION: SELL ---
        elif discrete_action == 2 and self.position_open:
            revenue = self.position_volume * price * (1 - self.commission_pct)
            pnl = revenue - (self.position_volume * self.entry_price)
            self.current_balance += revenue
            
            actual_pnl_ratio = (price / self.entry_price - 1) if self.entry_price > 0 else 0
            
            # --- *** KEY DIFFERENCE *** ---
            # Instead of using a fixed target from the config, we use the one
            # the agent chose when it opened this position.
            profit_target = self.current_desired_profit_target 
            
            # The reward logic is the same, but it's now judged against the agent's own goal
            if pnl <= 0:
                reward = pnl * self.config.get("penalty_loss_trade_factor", 1.0)
            else:
                performance_vs_target = actual_pnl_ratio - profit_target
                if performance_vs_target >= 0:
                    reward = performance_vs_target * self.config['reward_factor_above_target']
                else:
                    reward = performance_vs_target * self.config['penalty_factor_below_target']
            
            self.position_open = False
            self.trade_history.append({'type': 'sell', 'step': self.current_step, 'price': price, 'pnl': pnl})

        # --- ACTION: HOLD ---
        elif discrete_action == 0:
            if self.position_open:
                if price < self.entry_price:
                    reward += self.config["penalty_hold_losing_position"]
            else:
                reward += self.config["penalty_hold_flat_position"]

        self.current_step += 1
        
        # Termination conditions...
        equity = self.current_balance + (self.position_volume * price if self.position_open else 0)
        if equity < self.catastrophic_loss_limit:
            terminated = True
            reward -= 100
        if self.current_step > self.end_step:
            truncated = True
            if self.position_open:
                self.current_balance += self.position_volume * price * (1 - self.commission_pct)
                self.position_open = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()