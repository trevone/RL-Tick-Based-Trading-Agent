# tests/environments/test_base_env.py
import pytest
import pandas as pd
import numpy as np
import gymnasium as gym

# Import the environment and its default config from the new path
# These are the core components of the trading environment being tested.
from src.environments.base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

# Apply pytest-order marker. This module runs after data integrity and utils tests.
# This ensures that the environment's dependencies (data, wrappers) are tested first.
pytestmark = pytest.mark.order(3)

# --- Fixtures for common test data ---
# Fixtures provide reusable data setup for tests within this module.

@pytest.fixture
def mock_tick_data():
    """
    Provides mock tick data (Price, Quantity, IsBuyerMaker) for testing the environment.
    This simulates granular market data at a millisecond frequency.
    """
    np.random.seed(42) # Set seed for reproducibility
    num_ticks = 1000
    base_price = 100.0
    tick_dates = pd.date_range(start="2023-01-01 00:00:00", periods=num_ticks, freq='1ms', tz='UTC')
    df = pd.DataFrame(index=tick_dates)
    df['Price'] = base_price + np.cumsum(np.random.randn(num_ticks) * 0.01)
    df['Quantity'] = np.random.rand(1000) * 10 + 1
    df['IsBuyerMaker'] = np.random.choice([True, False], size=num_ticks)
    return df

@pytest.fixture
def mock_kline_data():
    """
    Provides mock K-line data (OHLCV) along with some basic Technical Analysis (TA) features.
    This simulates higher-frequency candle data that the environment uses for observation.
    """
    np.random.seed(42) # Set seed for reproducibility
    num_klines = 100 # Sufficient for a window size of 20
    kline_dates = pd.date_range(start="2023-01-01 00:00:00", periods=num_klines, freq='1h', tz='UTC')
    df = pd.DataFrame(index=kline_dates)
    df['Open'] = 100 + np.random.randn(num_klines) * 2
    df['High'] = df['Open'] + np.random.rand(num_klines) * 1
    df['Low'] = df['Open'] - np.random.rand(num_klines) * 1
    df['Close'] = df['Open'] + np.random.randn(num_klines) * 0.5
    df['Volume'] = np.random.rand(num_klines) * 100
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['RSI_7'] = np.random.rand(num_klines) * 100 # Mock RSI
    df['ATR'] = np.random.rand(num_klines) * 5 # Mock ATR
    df['MACD'] = np.random.randn(num_klines) # Mock MACD
    df['CDLDOJI'] = np.random.choice([0, 100, -100], size=num_klines) # Mock pattern
    df.fillna(0, inplace=True) # Fill any potential NaNs from rolling/TA for simplicity
    return df

@pytest.fixture
def trading_env(mock_tick_data, mock_kline_data):
    """
    Provides an instance of SimpleTradingEnv for testing.
    The environment is configured with mock data and suppressed logging for test clarity.
    """
    # Ensure environment config matches expected features in mock_kline_data
    env_config = DEFAULT_ENV_CONFIG.copy()
    env_config["tick_feature_window_size"] = 50
    env_config["kline_window_size"] = 20
    env_config["kline_price_features"] = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "RSI_7", "ATR", "MACD", "CDLDOJI"]
    env_config["log_level"] = "none" # Suppress verbose logging during tests
    return SimpleTradingEnv(tick_df=mock_tick_data, kline_df_with_ta=mock_kline_data, config=env_config)

# --- Test Environment Core Functionality ---
# These tests verify the basic initialization, reset behavior, and observation space
# of the SimpleTradingEnv.

def test_env_initialization(trading_env):
    """
    Test environment initialization and the properties of its observation and action spaces.
    Ensures initial state variables like balance and position are set correctly.
    """
    env = trading_env
    obs, info = env.reset()

    assert env.observation_space.shape is not None
    assert env.action_space is not None
    assert env.current_balance == env.initial_balance
    assert not env.position_open
    assert len(env.trade_history) == 1 # Only initial_balance event
    assert obs.shape == env.observation_space.shape
    assert isinstance(obs, np.ndarray)

def test_reset_functionality(trading_env):
    """
    Test the reset method of the environment.
    Verifies that the environment's state (balance, position, trade history) is reset
    to its initial configuration after simulating some steps.
    """
    env = trading_env
    initial_obs, initial_info = env.reset()

    # Simulate some steps
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)
    
    reset_obs, reset_info = env.reset() # Reset again

    assert np.all(initial_obs == reset_obs) # With fixed seed, initial obs should be same
    assert reset_info["current_balance"] == env.initial_balance
    assert not reset_info["position_open"]
    assert len(env.trade_history) == 1 # Trade history should be reset

# --- Test Trading Actions ---
# These tests verify the effects of 'Buy', 'Sell', and 'Hold' actions on the
# environment's state, balance, position, and trade history.

def test_buy_action(trading_env):
    """
    Test the buy action and its impact on balance, position state, and trade history.
    Verifies that a position is opened and balance decreases after a successful buy.
    """
    env = trading_env
    env.reset()
    initial_balance = env.current_balance
    current_price = env.decision_prices[env.current_step]

    # Action: Buy (1), with a profit target
    action = (1, np.array([0.01], dtype=np.float32))
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.position_open
    assert env.entry_price == current_price
    assert env.position_volume > 0
    assert env.current_balance < initial_balance # Balance should decrease after buy
    assert reward > 0 # Expect reward for opening buy position
    assert len(env.trade_history) == 2 # Initial + Buy trade
    assert info["position_open"] == True
    assert info["entry_price"] == current_price

def test_sell_action(trading_env):
    """
    Test the sell action and its impact after a prior buy.
    Verifies that a position is closed, balance increases (if profitable),
    and trade history is updated correctly.
    """
    env = trading_env
    env.reset()
    
    # First, buy to have a position
    buy_action = (1, np.array([0.01], dtype=np.float32))
    env.step(buy_action)
    
    # Store state before sell
    initial_position_volume = env.position_volume
    entry_price = env.entry_price
    balance_before_sell = env.current_balance
    
    # Advance a few steps to change price and then sell
    # Manipulate price to ensure a profit for the test case
    env.current_step += 5 # Advance steps for a new price point
    
    # Ensure current_step is within bounds for decision_prices
    if env.current_step >= len(env.decision_prices):
        env.current_step = len(env.decision_prices) - 1 # Adjust to last available tick
    
    current_price = entry_price * 1.05 # Guarantee 5% profit for this test scenario
    # Temporarily set the decision price for the current step in the environment's data
    original_price_at_step = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = current_price

    # Action: Sell (2)
    sell_action = (2, np.array([0.0], dtype=np.float32)) # Profit target param doesn't affect sell logic in this scenario
    obs, reward, terminated, truncated, info = env.step(sell_action)

    assert not env.position_open
    assert env.position_volume == 0
    assert env.entry_price == 0.0 # Entry price should reset to 0.0 after closing position
    assert env.current_balance > balance_before_sell # Balance should increase after a profitable sell
    assert 'sell' in [t['type'] for t in env.trade_history]
    assert len(env.trade_history) == 3 # Initial + Buy + Sell

    # Check reward for profit (should be positive now)
    pnl = (current_price - entry_price) * initial_position_volume * (1 - env.commission_pct)
    assert pnl > 0 # Ensure it was a profit
    assert reward > env.config["reward_sell_profit_base"]

    # Restore original price to avoid side effects on other tests (important for shared fixtures)
    env.decision_prices[env.current_step] = original_price_at_step


def test_hold_action(trading_env):
    """
    Test the hold action in both flat and open position states.
    Verifies that balance remains unchanged and position state is maintained as expected.
    """
    env = trading_env
    env.reset()
    initial_balance = env.current_balance
    
    # Hold when no position is open (flat position)
    action_hold_flat = (0, np.array([0.0], dtype=np.float32))
    obs, reward_flat, terminated, truncated, info = env.step(action_hold_flat)
    assert env.current_balance == initial_balance # Balance unchanged
    assert not env.position_open # Position remains closed
    assert reward_flat == env.config["penalty_hold_flat_position"] # Expect specific penalty for holding flat

    # Buy and then hold (hold with an open position)
    buy_action = (1, np.array([0.01], dtype=np.float32))
    env.step(buy_action)
    balance_after_buy = env.current_balance
    
    obs, reward_hold, terminated, truncated, info = env.step(action_hold_flat)
    assert env.current_balance == balance_after_buy # Balance unchanged (no trade executed)
    assert env.position_open # Position still open
    assert reward_hold != 0 # Should be reward_hold_profitable or penalty_hold_losing depending on price movement

# --- Test Episode Termination Conditions ---
# These tests verify that the environment correctly handles termination conditions
# such as catastrophic loss and end-of-data.

def test_catastrophic_loss(trading_env):
    """
    Test episode termination when equity drops below the catastrophic loss limit.
    Verifies that the episode terminates and a specific penalty is applied.
    """
    env = trading_env
    env.reset()
    
    # Configure the environment to easily hit the loss limit for testing
    env.initial_balance = 100.0 
    env.catastrophic_loss_limit = env.initial_balance * (1.0 - env.config["catastrophic_loss_threshold_pct"])
    
    # Simulate a buy action to correctly reduce current_balance and open a position.
    # This ensures equity calculation accurately reflects a live trade scenario.
    buy_price = env.decision_prices[env.current_step]
    env.base_trade_amount_ratio = 0.99 # Use almost all balance for trade for a dramatic loss
    buy_action = (1, np.array([0.01], dtype=np.float32))
    env.step(buy_action) # This will update current_balance, position_open, etc.

    # Now, manipulate price to trigger catastrophic loss.
    # Calculate the price that would cause equity to drop below limit, then set price slightly below.
    original_decision_prices = env.decision_prices.copy() # Store original prices for restoration
    
    # Ensure current_step is within bounds for decision_prices array
    if env.current_step >= len(env.decision_prices):
        env.current_step = len(env.decision_prices) - 1 # Adjust to last available tick
    
    # Calculate the exact price point required for equity to fall below the limit
    # equity = current_balance (after buy) + (position_volume * price)
    # price = (catastrophic_loss_limit - current_balance) / position_volume
    required_price_for_loss = (env.catastrophic_loss_limit - env.current_balance) / (env.position_volume + 1e-9) # Add epsilon to prevent division by zero
    # Set the price slightly below the required price to guarantee the loss condition
    price_at_loss = required_price_for_loss - 0.05 
    
    env.decision_prices[env.current_step] = price_at_loss # Apply the manipulated price
    
    # Step to trigger loss check
    action_hold = (0, np.array([0.0], dtype=np.float32))
    obs, reward, terminated, truncated, info = env.step(action_hold) 

    assert terminated # Verify that the episode terminated
    assert reward == pytest.approx(env.config["penalty_catastrophic_loss"] + env.config["penalty_hold_losing_position"], rel=1e-2) 
    assert 'sell_ruin_auto' in [t['type'] for t in env.trade_history] # Ensure auto-liquidation event is logged

    # Restore original prices to avoid side effects
    env.decision_prices = original_decision_prices


def test_episode_truncation_at_eof(trading_env):
    """
    Test episode truncation when the end of the data is reached.
    Verifies that the episode is truncated and any open position is automatically closed.
    """
    env = trading_env
    env.reset()

    # Set current_step close to end_step to trigger truncation soon
    env.current_step = env.end_step - 5 # A few steps before end of data
    
    # Play through remaining steps until truncation occurs
    for _ in range(10): # Will go past end_step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert truncated # Verify that the episode was truncated
    # Check if position was closed at EOF if it was open before truncation
    if env.position_open:
        assert 'sell_eof_auto' in [t['type'] for t in env.trade_history]

# --- Test Observation and Reward Clipping/Bonuses ---
# These tests verify specific aspects of observation processing and reward calculation,
# such as clipping values and applying profit target bonuses.

def test_observation_clipping(trading_env):
    """
    Test observation clipping behavior.
    Ensures that observation values (like normalized entry price or PnL ratio)
    are constrained within predefined low and high bounds.
    """
    env = trading_env
    env.reset()
    # Temporarily set extreme values in mock data to test clipping mechanism
    original_price = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = original_price * 1000 # Simulate a very high price

    # Force a position with extreme entry price to create extreme PnL features
    env.position_open = True
    env.entry_price = original_price / 1000 # Simulate a very low entry price
    env.position_volume = 1.0

    obs = env._get_observation() # Get the observation with extreme values

    # Define indices for relevant observation features
    obs_shape_ticks = env.tick_feature_window_size * env.num_tick_features_per_step
    obs_shape_klines = env.kline_window_size * env.num_kline_features_per_step
    
    norm_entry_price_idx = obs_shape_ticks + obs_shape_klines + 1
    unreal_pnl_ratio_idx = obs_shape_ticks + obs_shape_klines + 2

    # Verify that the observed features are clipped within the defined bounds
    assert obs[norm_entry_price_idx] <= env.config["obs_clip_high"]
    assert obs[unreal_pnl_ratio_idx] <= env.config["obs_clip_high"]
    assert obs[norm_entry_price_idx] >= env.config["obs_clip_low"]
    assert obs[unreal_pnl_ratio_idx] >= env.config["obs_clip_low"]

    # Restore environment state after the test
    env.decision_prices[env.current_step] = original_price
    env.position_open = False
    env.entry_price = 0.0
    env.position_volume = 0.0

def test_profit_target_bonus(trading_env):
    """
    Test the profit target bonus and penalty logic on sell actions.
    Verifies that the agent receives a bonus if a profit target is met or exceeded,
    and a penalty if profit is below target.
    """
    env = trading_env
    # Set a custom initial balance for this specific test for cleaner reward calculation
    env.initial_balance = 1000.0
    # FIX: Explicitly set catastrophic_loss_limit to a very low negative value to ensure it's not triggered in this test.
    # This value must be lower than any possible equity to prevent false positives.
    env.catastrophic_loss_limit = -10000.0 # Set to a very low negative number.
    env.reset()
    
    # Set up a buy action with a specific profit target
    # Use a price that results in a round `position_volume` to simplify PnL calculation
    buy_price = 10.0
    # Temporarily set the decision price for the current step in the environment's data
    original_price_at_step_buy = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = buy_price

    profit_target = 0.005 # 0.5% target for the test
    # Ensure `base_trade_amount_ratio` results in a whole number of units to avoid float issues in quantity
    env.base_trade_amount_ratio = 0.1 # This will result in 10 units for 1000 initial balance, 10.0 buy_price
    buy_action = (1, np.array([profit_target], dtype=np.float32))
    env.step(buy_action) # Execute the buy action

    # Restore original price after buy step
    env.decision_prices[env.current_step] = original_price_at_step_buy


    # Simulate price increase to just meet or slightly exceed the profit target
    env.current_step += 1 # Advance step to get a new price point
    
    # Ensure current_step is within bounds for decision_prices
    if env.current_step >= len(env.decision_prices):
        env.current_step = len(env.decision_prices) - 1

    # Price slightly above target
    target_sell_price_met = buy_price * (1 + profit_target + 0.0001)
    original_price_met_at_step = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = target_sell_price_met # Set manipulated price

    sell_action = (2, np.array([0.0], dtype=np.float32))
    obs_met, reward_met, terminated_met, truncated_met, info_met = env.step(sell_action)
    
    # Calculate the expected base reward without any bonus
    # Use the env's current values for position_volume and entry_price as they are set by the buy action
    pnl_met = (env.position_volume * target_sell_price_met * (1 - env.commission_pct)) - (env.position_volume * buy_price)
    expected_base_reward_met = env.config["reward_sell_profit_base"] + (pnl_met / (env.initial_balance + 1e-9)) * env.config["reward_sell_profit_factor"]
    
    assert reward_met > expected_base_reward_met # Verify that a bonus was received
    # Adjusted tolerance - increasing relative tolerance and adding absolute tolerance
    assert reward_met == pytest.approx(expected_base_reward_met + env.config["reward_sell_meets_target_bonus"], rel=1e-3, abs=1e-6)


    # Restore original price to avoid side effects
    env.decision_prices[env.current_step] = original_price_met_at_step

    # Reset environment and test price increase below target
    env.initial_balance = 1000.0 # Re-set for the second part of the test
    # FIX: Explicitly set catastrophic_loss_limit to a very low negative value for the second part as well
    env.catastrophic_loss_limit = -10000.0
    env.reset()
    
    # Re-buy to reset state with the same profit target
    # Use a price that results in a round `position_volume` to simplify PnL calculation
    buy_price = 10.0
    env.base_trade_amount_ratio = 0.1 # This will result in 10 units for 1000 initial balance, 10.0 buy_price
    original_price_at_step_buy = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = buy_price

    buy_action = (1, np.array([profit_target], dtype=np.float32))
    env.step(buy_action) # Execute the buy action

    # Restore original price after buy step
    env.decision_prices[env.current_step] = original_price_at_step_buy
    
    env.current_step += 1 # Advance step
    if env.current_step >= len(env.decision_prices):
        env.current_step = len(env.decision_prices) - 1

    # Price slightly below target
    target_sell_price_below = buy_price * (1 + profit_target - 0.0001)
    original_price_below_at_step = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = target_sell_price_below # Set manipulated price

    sell_action = (2, np.array([0.0], dtype=np.float32))
    obs_below, reward_below, terminated_below, truncated_below, info_below = env.step(sell_action)

    # Calculate expected base reward without any penalty
    pnl_below = (env.position_volume * target_sell_price_below * (1 - env.commission_pct)) - (env.position_volume * buy_price)
    expected_base_reward_below = env.config["reward_sell_profit_base"] + (pnl_below / (env.initial_balance + 1e-9)) * env.config["reward_sell_profit_factor"]

    assert reward_below < expected_base_reward_below # Verify that a penalty was applied
    assert reward_below == pytest.approx(expected_base_reward_below + env.config["penalty_sell_profit_below_target"], rel=1e-3, abs=1e-6)

    # Restore original price
    env.decision_prices[env.current_step] = original_price_below_at_step