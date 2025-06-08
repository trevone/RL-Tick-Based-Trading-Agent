import pandas as pd
import numpy as np

from ..base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG # Import SimpleTradingEnv for base config
from .dynamic_target_env import DynamicTargetEnv # Inherit from DynamicTargetEnv

# Combine relevant config settings for this new environment
PROPORTIONAL_DYNAMIC_TARGET_EXP_CONFIG = {
    **DEFAULT_ENV_CONFIG, # Start with all base environment defaults
    
    # These are parameters that DynamicTargetEnv uses or new ones for proportional scaling.
    # Agent will choose 'profit_target_pct' via action, but a default might be used internally
    # or for configuration consistency.
    "profit_target_pct": 0.002, # Default profit target for calculations if needed
    
    "reward_trade_completion_bonus_value": 0.1, # A flat bonus for successfully completing a trade
    "reward_scale_deviation_positive": 50.0,    # Scales positive deviation from target (e.g., 1% deviation = 0.5 reward)
    "reward_scale_deviation_negative": 50.0,    # Scales negative deviation from target
    "reward_scale_actual_pnl_loss": 100.0,      # Stronger penalty factor for actual financial losses (PnL < 0)

    # You might want to override simplified penalties/rewards from DynamicTargetEnv's step
    # with more aligned values from DEFAULT_ENV_CONFIG or custom ones:
    "penalty_buy_position_already_open": -0.1, # Re-introduce/re-scale this from base
    "penalty_sell_no_position": -0.1,          # Re-introduce/re-scale this from base
    "penalty_hold_flat_position": -0.05,       # Re-introduce/re-scale this from base
    "penalty_hold_losing_position": -0.0005,
    "reward_hold_profitable_position": 0.0001,
}

class ProportionalDynamicTargetEnv(DynamicTargetEnv):
    def __init__(self, tick_df: pd.DataFrame, kline_df_with_ta: pd.DataFrame, config: dict = None):
        # Merge provided config with new defaults for this env
        env_config = {**PROPORTIONAL_DYNAMIC_TARGET_EXP_CONFIG, **(config if config else {})}
        # Initialize parent class (DynamicTargetEnv) with the merged config
        super().__init__(tick_df, kline_df_with_ta, env_config)

        # Confirm environment is being initialized (optional, for debugging)
        if self.log_level in ["normal", "detailed"]:
            print(f"--- INITIALIZING PROPORTIONAL DYNAMIC TARGET ENVIRONMENT (ID: {self.env_id}) ---")

    def step(self, action_tuple):
        discrete_action, profit_target_param_array = action_tuple # Get the agent's full action (discrete + continuous target)

        reward, terminated, truncated = 0.0, False, False
        price = self.decision_prices[self.current_step]
        
        # NOTE: This environment's step method explicitly re-implements BUY/HOLD logic
        # and then applies complex reward for SELL, similar to DynamicTargetEnv.
        # It does NOT call super().step() at the very beginning to get parent's rewards,
        # but rather integrates the logic from DynamicTargetEnv's step directly.

        trade_executed_this_step = False
        trade_details_for_log = {} # For logging/info dictionary if needed

        # --- Action: BUY (discrete_action == 1) ---
        if discrete_action == 1:
            if not self.position_open:
                cost = (self.initial_balance * self.base_trade_amount_ratio) * (1 + self.commission_pct)
                if self.current_balance >= cost:
                    self.position_volume = (self.initial_balance * self.base_trade_amount_ratio) / price
                    self.current_balance -= cost
                    self.position_open, self.entry_price = True, price
                    self.current_desired_profit_target = profit_target_param_array[0] # Capture agent's chosen target
                    reward = self.config.get("reward_open_buy_position", 0.001) # Use base reward for opening position
                else:
                    # Penalize for insufficient balance, as defined in config
                    reward = self.config.get("penalty_buy_insufficient_balance", -0.1)
            else:
                # Penalize for attempting to buy when already in a position
                reward = self.config.get("penalty_buy_position_already_open", -0.1)

        # --- Action: SELL (discrete_action == 2) - Primary focus for proportional reward ---
        elif discrete_action == 2:
            if self.position_open:
                revenue = self.position_volume * price * (1 - self.commission_pct)
                self.current_balance += revenue
                
                # --- Start Proportional Reward Logic (from previous discussions) ---
                # Calculate actual PnL ratio (percentage gain/loss relative to entry price)
                actual_pnl_ratio = (price / self.entry_price - 1) if self.entry_price > 1e-9 else 0.0
                
                # Calculate how much the actual PnL ratio deviated from the agent's chosen target
                deviation_from_target = actual_pnl_ratio - self.current_desired_profit_target

                # Scale the reward based on deviation. Positive deviation gets positive scaled reward, etc.
                if deviation_from_target >= 0:
                    reward = deviation_from_target * self.config['reward_scale_deviation_positive']
                else:
                    reward = deviation_from_target * self.config['reward_scale_deviation_negative']
                
                # Add a flat bonus for completing any trade (encourages closing positions)
                reward += self.config['reward_trade_completion_bonus_value']

                # Add an additional, potentially strong, penalty if the trade resulted in an actual financial loss
                if actual_pnl_ratio < 0:
                    # Scale the actual PnL dollars to make it a more significant penalty
                    pnl_dollars = (price - self.entry_price) * self.position_volume
                    reward += (pnl_dollars / (self.initial_balance + 1e-9)) * self.config['reward_scale_actual_pnl_loss']
                # --- End Proportional Reward Logic ---

                # Reset position-related state variables after a successful sell
                self.position_open, self.position_volume, self.entry_price = False, 0.0, 0.0
                self.current_desired_profit_target = 0.0 # Reset agent's target after closing position
                trade_executed_this_step = True # Flag for logging if needed
            else:
                # Penalize for attempting to sell when no position is open
                reward = self.config.get("penalty_sell_no_position", -0.1)

        # --- Action: HOLD (discrete_action == 0) ---
        elif discrete_action == 0:
            if self.position_open:
                # Apply penalties/rewards for holding an open position (from base_env config)
                if price < self.entry_price:
                    reward = self.config.get("penalty_hold_losing_position", -0.0005)
                else: # Holding a profitable position
                    reward = self.config.get("reward_hold_profitable_position", 0.0001)
            else:
                # Apply penalty for holding a flat position (from base_env config)
                reward = self.config.get("penalty_hold_flat_position", -0.0001)

        # --- Common logic for advancing step and checking for termination/truncation ---
        self.current_step += 1
        current_equity = self.current_balance + (self.position_volume * price if self.position_open else 0)

        # Catastrophic loss: Terminates the episode with a large penalty
        if current_equity < self.catastrophic_loss_limit:
            terminated = True
            reward += self.config["penalty_catastrophic_loss"]
        
        # Episode truncated if end of data is reached
        if self.current_step > self.end_step:
            truncated = True
        
        # If the episode ends (terminated or truncated) and there's an open position, close it out.
        if (terminated or truncated) and self.position_open:
            revenue_on_forced_close = self.position_volume * price * (1 - self.commission_pct)
            self.current_balance += revenue_on_forced_close
            self.position_open = False
            # No additional reward here, as the primary sell reward is designed for explicit closes.

        # Return the observation, calculated reward, termination flags, and info dictionary
        return self._get_observation(), reward, terminated, truncated, self._get_info()