# tests/environments/test_base_env.py
import pytest
import pandas as pd
import numpy as np
import gymnasium as gym

# Import the environment and its default config from the new path
from src.environments.base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG

@pytest.fixture
def mock_tick_data():
    """Provides mock tick data for testing."""
    np.random.seed(42)
    num_ticks = 1000
    base_price = 100.0
    tick_dates = pd.date_range(start="2023-01-01 00:00:00", periods=num_ticks, freq='1ms', tz='UTC')
    df = pd.DataFrame(index=tick_dates)
    df['Price'] = base_price + np.cumsum(np.random.randn(num_ticks) * 0.01)
    df['Quantity'] = np.random.rand(num_ticks) * 10 + 1
    df['IsBuyerMaker'] = np.random.choice([True, False], size=num_ticks)
    return df

@pytest.fixture
def mock_kline_data():
    """Provides mock k-line data with some basic TAs for testing."""
    np.random.seed(42)
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
    """Provides a SimpleTradingEnv instance for testing."""
    # Ensure environment config matches expected features in mock_kline_data
    env_config = DEFAULT_ENV_CONFIG.copy()
    env_config["tick_feature_window_size"] = 50
    env_config["kline_window_size"] = 20
    env_config["kline_price_features"] = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "RSI_7", "ATR", "MACD", "CDLDOJI"]
    env_config["log_level"] = "none" # Suppress verbose logging during tests
    return SimpleTradingEnv(tick_df=mock_tick_data, kline_df_with_ta=mock_kline_data, config=env_config)

def test_env_initialization(trading_env):
    """Test environment initialization and observation space."""
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
    """Test the reset method of the environment."""
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

def test_buy_action(trading_env):
    """Test the buy action and state changes."""
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
    """Test the sell action after a buy."""
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
    
    current_price = entry_price * 1.05 # Guarantee 5% profit
    # Temporarily set the decision price for the current step
    original_price_at_step = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = current_price

    # Action: Sell (2)
    sell_action = (2, np.array([0.0], dtype=np.float32)) # Profit target doesn't matter for sell in raw action
    obs, reward, terminated, truncated, info = env.step(sell_action)

    assert not env.position_open
    assert env.position_volume == 0
    assert env.entry_price == 0.0 # Entry price should reset to 0.0
    assert env.current_balance > balance_before_sell # Balance should increase after sell
    assert 'sell' in [t['type'] for t in env.trade_history]
    assert len(env.trade_history) == 3 # Initial + Buy + Sell

    # Check reward for profit (should be positive now)
    pnl = (current_price - entry_price) * initial_position_volume * (1 - env.commission_pct)
    assert pnl > 0 # Ensure it was a profit
    assert reward > env.config["reward_sell_profit_base"]

    # Restore original price
    env.decision_prices[env.current_step] = original_price_at_step


def test_hold_action(trading_env):
    """Test the hold action."""
    env = trading_env
    env.reset()
    initial_balance = env.current_balance
    
    # Hold when no position
    action_hold_flat = (0, np.array([0.0], dtype=np.float32))
    obs, reward_flat, terminated, truncated, info = env.step(action_hold_flat)
    assert env.current_balance == initial_balance # Balance unchanged
    assert not env.position_open
    assert reward_flat == env.config["penalty_hold_flat_position"]

    # Buy and then hold
    buy_action = (1, np.array([0.01], dtype=np.float32))
    env.step(buy_action)
    balance_after_buy = env.current_balance
    
    obs, reward_hold, terminated, truncated, info = env.step(action_hold_flat)
    assert env.current_balance == balance_after_buy # Balance unchanged
    assert env.position_open # Position still open
    assert reward_hold != 0 # Should be reward_hold_profitable or penalty_hold_losing

def test_catastrophic_loss(trading_env):
    """Test episode termination on catastrophic loss."""
    env = trading_env
    env.reset()
    env.initial_balance = 100.0 # Make it easy to hit loss limit
    env.catastrophic_loss_limit = env.initial_balance * (1.0 - env.config["catastrophic_loss_threshold_pct"])
    env.current_balance = env.initial_balance

    # Force a position
    env.position_open = True
    env.entry_price = 100.0
    env.position_volume = 1.0

    # Simulate price drop to trigger catastrophic loss at the current step's price
    original_decision_prices = env.decision_prices.copy()
    
    # Ensure current_step is within bounds for decision_prices
    if env.current_step >= len(env.decision_prices):
        env.current_step = len(env.decision_prices) - 1 # Adjust to last available tick

    price_at_loss = env.entry_price * (1.0 - env.config["catastrophic_loss_threshold_pct"] - 0.01) # Price drops below threshold
    env.decision_prices[env.current_step] = price_at_loss
    
    # Step to trigger loss
    action_hold = (0, np.array([0.0], dtype=np.float32))
    obs, reward, terminated, truncated, info = env.step(action_hold)

    assert terminated
    assert reward == env.config["penalty_catastrophic_loss"]
    assert 'sell_ruin_auto' in [t['type'] for t in env.trade_history] # Should auto-liquidate

    env.decision_prices = original_decision_prices # Restore

def test_episode_truncation_at_eof(trading_env):
    """Test episode truncation when end of data is reached."""
    env = trading_env
    env.reset()

    # Set current_step close to end_step
    env.current_step = env.end_step - 5 # A few steps before end
    
    # Play through remaining steps
    for _ in range(10): # Will go past end_step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert truncated
    # Check if position was closed at EOF if open
    if env.position_open:
        assert 'sell_eof_auto' in [t['type'] for t in env.trade_history]

def test_observation_clipping(trading_env):
    """Test observation clipping behavior."""
    env = trading_env
    env.reset()
    # Temporarily set extreme values in mock data to test clipping
    original_price = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = original_price * 1000 # Very high price

    # Force a position for pnl features
    env.position_open = True
    env.entry_price = original_price / 1000 # Very low entry price
    env.position_volume = 1.0

    obs = env._get_observation()

    # Check that relevant observation features are clipped
    # (assuming order: ticks, klines, is_open, norm_entry_price, unreal_pnl_ratio)
    obs_shape_ticks = env.tick_feature_window_size * env.num_tick_features_per_step
    obs_shape_klines = env.kline_window_size * env.num_kline_features_per_step
    
    norm_entry_price_idx = obs_shape_ticks + obs_shape_klines + 1
    unreal_pnl_ratio_idx = obs_shape_ticks + obs_shape_klines + 2

    assert obs[norm_entry_price_idx] <= env.config["obs_clip_high"]
    assert obs[unreal_pnl_ratio_idx] <= env.config["obs_clip_high"]
    assert obs[norm_entry_price_idx] >= env.config["obs_clip_low"]
    assert obs[unreal_pnl_ratio_idx] >= env.config["obs_clip_low"]

    # Restore data
    env.decision_prices[env.current_step] = original_price
    env.position_open = False
    env.entry_price = 0.0
    env.position_volume = 0.0

def test_profit_target_bonus(trading_env):
    """Test the profit target bonus/penalty on sell."""
    env = trading_env
    env.reset()

    # Set up a buy action with a specific profit target
    initial_balance = env.current_balance
    buy_price = env.decision_prices[env.current_step]
    profit_target = 0.005 # 0.5% target
    buy_action = (1, np.array([profit_target], dtype=np.float32))
    env.step(buy_action)

    # Simulate price increase to just meet target
    env.current_step += 1 # Advance step
    
    # Ensure current_step is within bounds for decision_prices
    if env.current_step >= len(env.decision_prices):
        env.current_step = len(env.decision_prices) - 1

    target_sell_price_met = buy_price * (1 + profit_target + 0.0001) # Slightly above target
    original_price_met_at_step = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = target_sell_price_met

    sell_action = (2, np.array([0.0], dtype=np.float32)) # Profit target param doesn't affect sell logic
    obs_met, reward_met, terminated_met, truncated_met, info_met = env.step(sell_action)
    
    # Calculate expected base reward without bonus
    pnl_met = (target_sell_price_met * env.position_volume * (1 - env.commission_pct)) - (buy_price * env.position_volume)
    expected_base_reward_met = env.config["reward_sell_profit_base"] + (pnl_met / (env.initial_balance + 1e-9)) * env.config["reward_sell_profit_factor"]
    
    assert reward_met > expected_base_reward_met # Should have received bonus
    assert reward_met == pytest.approx(expected_base_reward_met + env.config["reward_sell_meets_target_bonus"])

    # Restore original price
    env.decision_prices[env.current_step] = original_price_met_at_step

    # Reset and test price increase below target
    env.reset()
    env.step(buy_action) # Re-buy to reset state with same target
    
    env.current_step += 1 # Advance step
    if env.current_step >= len(env.decision_prices):
        env.current_step = len(env.decision_prices) - 1

    target_sell_price_below = buy_price * (1 + profit_target - 0.0001) # Slightly below target
    original_price_below_at_step = env.decision_prices[env.current_step]
    env.decision_prices[env.current_step] = target_sell_price_below

    obs_below, reward_below, terminated_below, truncated_below, info_below = env.step(sell_action)

    # Calculate expected base reward without penalty
    pnl_below = (target_sell_price_below * env.position_volume * (1 - env.commission_pct)) - (buy_price * env.position_volume)
    expected_base_reward_below = env.config["reward_sell_profit_base"] + (pnl_below / (env.initial_balance + 1e-9)) * env.config["reward_sell_profit_factor"]

    assert reward_below < expected_base_reward_below # Should have received penalty
    assert reward_below == pytest.approx(expected_base_reward_below + env.config["penalty_sell_profit_below_target"])

    # Restore original price
    env.decision_prices[env.current_step] = original_price_below_at_step