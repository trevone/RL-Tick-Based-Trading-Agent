# src/environments/experiments/decisive_trader_env.py
import numpy as np
from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

# --- Configuration for the Decisive Trader Environment ---
DECISIVE_TRADER_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    "time_decay_penalty_factor": -0.001, # The penalty per step for holding a position
    "reward_buy_open": 0.0,
    "penalty_incorrect_action": -0.01
}

class DecisiveTraderEnv(SimpleTradingEnv):
    """
    An environment that penalizes the agent for the duration of a trade.
    - It encourages efficient, decisive trades by applying a small, accumulating
      penalty at each step a position is held.
    - The final reward for a SELL is still the realized PnL, creating a
      trade-off between holding for more profit and minimizing the time penalty.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        final_config = DECISIVE_TRADER_CONFIG.copy()
        if config:
            final_config.update(config)
        super().__init__(tick_df, kline_df_with_ta, config=final_config)

    def reset(self, seed=None, options=None):
        # We need to add trade_start_step to the reset method
        self.trade_start_step = 0
        return super().reset(seed=seed, options=options)

    def _calculate_reward(self, discrete_action, price_for_action):
        if not self.position_open:
            if discrete_action == 1: # BUY
                return self.config["reward_buy_open"]
            else: # Incorrect SELL or HOLD
                return self.config["penalty_incorrect_action"]
        else:
            if discrete_action == 0: # HOLD
                # Calculate how many steps the trade has been open
                steps_in_trade = self.current_step - self.trade_start_step
                # Apply a penalty that increases with the duration of the trade
                time_penalty = steps_in_trade * self.config["time_decay_penalty_factor"]
                return time_penalty
            
            elif discrete_action == 2: # SELL
                # Reward for selling is the standard realized profit
                realized_pnl = (price_for_action - self.entry_price) * self.position_volume
                return realized_pnl

            else: # Incorrect BUY
                return self.config["penalty_incorrect_action"]

    def step(self, action_tuple):
        # We must override the step method to manage trade_start_step
        
        # This part is the same as the base class
        discrete_action, _ = action_tuple
        price = self.decision_prices[self.current_step]
        terminated, truncated = False, False
        reward = self._calculate_reward(discrete_action, price)

        # --- Execute Trade and Update State (with modification) ---
        if discrete_action == 1 and not self.position_open: # BUY
            cost = (self.initial_balance * self.base_trade_amount_ratio) * (1 + self.commission_pct)
            if self.current_balance >= cost:
                self.position_volume = (self.initial_balance * self.base_trade_amount_ratio) / price
                self.current_balance -= cost
                self.position_open, self.entry_price = True, price
                # *** MODIFICATION: Record when the trade started ***
                self.trade_start_step = self.current_step
        
        elif discrete_action == 2 and self.position_open: # SELL
            revenue = self.position_volume * price * (1 - self.commission_pct)
            self.current_balance += revenue
            self.position_open, self.position_volume, self.entry_price = False, 0.0, 0.0
       
        # --- This part is the same as the base class ---
        self.current_step += 1
        current_equity = self.current_balance + (self.position_volume * price if self.position_open else 0)

        if current_equity < self.catastrophic_loss_limit:
            terminated = True
            reward += self.config["penalty_catastrophic_loss"]

        if self.current_step > self.end_step:
            truncated = True
        
        if (terminated or truncated) and self.position_open:
            self.current_balance += self.position_volume * price * (1 - self.commission_pct)
            self.position_open = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()