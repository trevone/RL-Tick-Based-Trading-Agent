# tests/environments/test_base_env.py
import pytest
import pandas as pd
import numpy as np
from src.environments.base_env import SimpleTradingEnv

@pytest.fixture
def mock_tick_data():
    np.random.seed(42)
    df = pd.DataFrame(index=pd.to_datetime(pd.RangeIndex(start=0, stop=1000, step=1), unit='ms', utc=True))
    df['Price'] = 100.0 + np.cumsum(np.random.randn(1000) * 0.01)
    df['Quantity'] = np.random.rand(1000) * 10 + 1
    return df

@pytest.fixture
def mock_kline_data():
    np.random.seed(42)
    df = pd.DataFrame(index=pd.to_datetime(pd.RangeIndex(start=0, stop=100*3600*1000, step=3600*1000), unit='ms', utc=True))
    df['Open'], df['High'], df['Low'], df['Close'], df['Volume'] = [100.0]*100, [101.0]*100, [99.0]*100, [100.5]*100, [10.0]*100
    return df

@pytest.fixture
def trading_env(mock_tick_data, mock_kline_data):
    config = {"tick_feature_window_size": 10, "kline_window_size": 5}
    return SimpleTradingEnv(tick_df=mock_tick_data, kline_df_with_ta=mock_kline_data, config=config)

def test_initialization(trading_env):
    assert trading_env.current_balance == trading_env.initial_balance
    assert not trading_env.position_open

def test_reset(trading_env):
    trading_env.step(trading_env.action_space.sample())
    obs, info = trading_env.reset()
    assert info['equity'] == trading_env.initial_balance
    assert not info['position_open']

def test_buy_action(trading_env):
    initial_balance = trading_env.current_balance
    price = trading_env.decision_prices[trading_env.current_step]
    obs, reward, _, _, info = trading_env.step((1, np.array([0.5])))
    assert info['position_open']
    assert info['equity'] < initial_balance
    assert reward == 0.0

def test_sell_action_profit(trading_env):
    trading_env.step((1, np.array([0.5]))) # Buy
    entry_price = trading_env.entry_price
    trading_env.decision_prices[trading_env.current_step] = entry_price * 1.01 # Simulate 1% profit
    obs, reward, _, _, info = trading_env.step((2, np.array([0.5]))) # Sell
    assert not info['position_open']
    assert reward > 0 # Should get completion bonus + profit bonus

def test_sell_action_loss(trading_env):
    trading_env.step((1, np.array([0.5]))) # Buy
    entry_price = trading_env.entry_price
    trading_env.decision_prices[trading_env.current_step] = entry_price * 0.99 # Simulate 1% loss
    obs, reward, _, _, info = trading_env.step((2, np.array([0.5]))) # Sell
    assert not info['position_open']
    assert reward < 0 # Should be negative PnL + completion bonus

def test_hold_action(trading_env):
    # Hold flat
    obs, reward, _, _, info = trading_env.step((0, np.array([0.5])))
    assert reward == trading_env.config["penalty_hold_flat_position"]
    # Hold losing
    trading_env.step((1, np.array([0.5]))) # Buy
    trading_env.decision_prices[trading_env.current_step] = trading_env.entry_price * 0.99
    obs, reward, _, _, info = trading_env.step((0, np.array([0.5])))
    assert reward == trading_env.config["penalty_hold_losing_position"]