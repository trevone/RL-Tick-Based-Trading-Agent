# src/environments/experiments/experimental_env.py
import numpy as np

# Inherit from the original SimpleTradingEnv to reuse its core functionality
from src.environments.base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

class ExperimentalTradingEnv(SimpleTradingEnv):
    """
    An experimental trading environment that inherits from SimpleTradingEnv
    but implements a different reward structure.
    """
    def __init__(self, tick_df, kline_df_with_ta, config=None):
        # Initialize the parent class with the same data and config
        super().__init__(tick_df, kline_df_with_ta, config)
        
        # Announce that the experimental environment is being used
        if self.log_level in ["normal", "detailed"]:
            print(f"--- INITIALIZING EXPERIMENTAL TRADING ENVIRONMENT (ID: {self.env_id}) ---")

    def step(self, action_tuple):
        """
        Overrides the step method to implement a custom reward logic.
        
        This example changes the reward for profitable sells to be a fixed
        value plus a bonus for meeting the profit target, removing the PnL scaling.
        This is just an example of how you could experiment.
        """
        discrete_action, profit_target_param = action_tuple
        if isinstance(discrete_action, np.integer):
            discrete_action = int(discrete_action)

        current_price_for_action = self.decision_prices[self.current_step]
        reward = 0.0
        terminated = False
        truncated = False
        trade_executed_this_step = False
        trade_details_for_log = {}

        # --- ACTION: SELL (with custom reward logic) ---
        if discrete_action == 2 and self.position_open:
            revenue_from_sale = self.position_volume * current_price_for_action
            commission_cost = revenue_from_sale * self.commission_pct
            net_revenue = revenue_from_sale - commission_cost
            cost_basis_of_position = self.entry_price * self.position_volume
            pnl_this_trade = net_revenue - cost_basis_of_position
            
            actual_pnl_ratio = (current_price_for_action / self.entry_price - 1) if self.entry_price > 1e-9 else 0

            # --- CUSTOM REWARD LOGIC ---
            if pnl_this_trade > 0:
                # Fixed reward for any profit, instead of scaled by PnL
                reward += 1.0 
                if self.log_level == "detailed":
                    print("  DEBUG (Experimental): Positive PnL detected. Awarding fixed profit reward.")
                
                # Bonus for meeting the original profit target
                if actual_pnl_ratio >= self.current_desired_profit_target:
                    reward += self.config.get("reward_sell_meets_target_bonus", 0.5)
            else:
                # Keep the original loss penalty
                reward += (pnl_this_trade / (self.initial_balance + 1e-9)) * self.config["penalty_sell_loss_factor"] + self.config["penalty_sell_loss_base"]
            # --- END CUSTOM REWARD LOGIC ---

            self.current_balance += net_revenue
            trade_details_for_log = {'type': 'sell', 'price': current_price_for_action, 'volume': self.position_volume,
                                     'balance': self.current_balance, 'pnl': pnl_this_trade, 'commission': commission_cost,
                                     'equity': self.current_balance, 'profit_target_aimed': self.current_desired_profit_target,
                                     'pnl_ratio_achieved': actual_pnl_ratio}
            self.position_open = False
            self.entry_price = 0.0
            self.position_volume = 0.0
            self.current_desired_profit_target = 0.0
            trade_executed_this_step = True
        
        # --- For all other actions (BUY, HOLD), call the parent's step method ---
        else:
            # To avoid re-writing all the logic, we can call the parent's step.
            # However, since the parent's step() also increments the step and gets observation,
            # it's often cleaner to copy the relevant logic sections from the parent that you *don't* want to change.
            # For this example, we'll just handle the other actions simply.
            # NOTE: A full implementation would copy the BUY and HOLD logic from base_env.py here.
            # This is a simplified example.
            if discrete_action == 1: # Simplified BUY
                reward += self.config["penalty_buy_position_already_open"] if self.position_open else 0.0
            elif discrete_action == 0: # Simplified HOLD
                reward += self.config["penalty_hold_flat_position"]

        if trade_executed_this_step:
            self.trade_history.append({'step': self.current_step, 'time': self.tick_df.index[self.current_step].isoformat(), **trade_details_for_log})

        # --- Common logic for advancing step and checking for termination ---
        self.current_step += 1
        current_equity = self.current_balance + (self.position_volume * current_price_for_action if self.position_open else 0)
        if current_equity < self.catastrophic_loss_limit:
            terminated = True
        if self.current_step > self.end_step:
            truncated = True
        
        observation = self._get_observation()
        info = self._get_info()

        return observation, np.nan_to_num(reward).item(), terminated, truncated, info