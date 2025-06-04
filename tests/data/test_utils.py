# tests/data/test_utils.py
import pytest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import time # Added for mocking time.sleep

# Import functions from the new path
from src.data.utils import (
    _calculate_technical_indicators,
    _load_single_yaml_config,
    load_config,
    fetch_and_cache_kline_data,
    fetch_continuous_aggregate_trades,
    get_data_path_for_day,
    load_tick_data_for_range,
    load_kline_data_for_range,
    merge_configs,
    generate_config_hash,
    convert_to_native_types,
    get_relevant_config_for_hash,
    resolve_model_path,
    DATA_CACHE_DIR # Use the globally defined cache directory
)

# --- Fixtures for common test data ---
@pytest.fixture(scope="module")
def sample_kline_df():
    """Provides a sample DataFrame for TA calculation (only base OHLCV)."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq='1h', tz='UTC')
    df = pd.DataFrame(index=dates)
    df['Open'] = np.random.rand(100) * 100 + 1000
    df['High'] = df['Open'] + np.random.rand(100) * 5
    df['Low'] = df['Open'] - np.random.rand(100) * 5
    df['Close'] = df['Open'] + (np.random.rand(100) - 0.5) * 5
    df['Volume'] = np.random.rand(100) * 1000
    return df

@pytest.fixture
def mock_cache_dir(tmp_path):
    """Creates a temporary cache directory for tests."""
    test_cache_dir = tmp_path / "test_cache"
    test_cache_dir.mkdir()
    return str(test_cache_dir)

@pytest.fixture
def mock_logs_dir(tmp_path):
    """Creates a temporary logs directory for tests."""
    test_logs_dir = tmp_path / "test_logs"
    test_logs_dir.mkdir()
    (test_logs_dir / "training").mkdir()
    (test_logs_dir / "evaluation").mkdir()
    (test_logs_dir / "live_trading").mkdir()
    (test_logs_dir / "tensorboard_logs").mkdir()
    return str(test_logs_dir)

@pytest.fixture
def mock_configs_dir(tmp_path):
    """Creates a temporary config directory with default configs."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    # Create dummy config files
    (defaults_dir / "run_settings.yaml").write_text("run_settings:\n  log_dir_base: 'logs/'\n  model_name: 'test_agent'\n")
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 20\n  kline_price_features: ['Close']\n")
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.001\n  total_timesteps: 100000\n")
    # ADDED: sac_params.yaml for test_get_relevant_config_for_hash
    (defaults_dir / "sac_params.yaml").write_text("sac_params:\n  learning_rate: 0.0005\n  total_timesteps: 50000\n  buffer_size: 10000\n")
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings:\n  default_symbol: 'BTCUSDT'\n  testnet: True\n")
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size']\n  agent_params:\n    PPO: ['learning_rate', 'total_timesteps']\n    SAC: ['learning_rate', 'total_timesteps', 'buffer_size']\n  binance_settings: ['default_symbol']\n")
    
    # Create main config.yaml
    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n")
    
    return str(tmp_path)

# --- Tests for _calculate_technical_indicators ---
@pytest.mark.parametrize("feature, expected_col", [
    ("SMA_10", "SMA_10"),
    ("RSI_14", "RSI_14"),
    ("MACD", "MACD"),
    ("ATR", "ATR"),
    ("CDLDOJI", "CDLDOJI"),
    ("STOCH_K", "STOCH_K")
])
def test_calculate_technical_indicators_adds_features(sample_kline_df, feature, expected_col):
    """Test that specified TA features are added."""
    # Ensure TALIB is mocked, otherwise, this test depends on external library existence
    with patch('src.data.utils.TALIB_AVAILABLE', True), \
         patch('src.data.utils.talib.SMA', return_value=np.arange(len(sample_kline_df))), \
         patch('src.data.utils.talib.RSI', return_value=np.arange(len(sample_kline_df))), \
         patch('src.data.utils.talib.MACD', return_value=(np.arange(len(sample_kline_df)), None, None)), \
         patch('src.data.utils.talib.ATR', return_value=np.arange(len(sample_kline_df))), \
         patch('src.data.utils.talib.CDLDOJI', return_value=np.arange(len(sample_kline_df))), \
         patch('src.data.utils.talib.STOCH', return_value=(np.arange(len(sample_kline_df)), None)):
        
        df_processed = _calculate_technical_indicators(sample_kline_df, [feature])
        assert expected_col in df_processed.columns
        assert not df_processed[expected_col].isnull().any()

def test_calculate_technical_indicators_no_talib(sample_kline_df):
    """Test behavior when TA-Lib is not available."""
    with patch('src.data.utils.TALIB_AVAILABLE', False):
        # We explicitly request a TA feature here
        price_features_to_add = ["Close", "SMA_10"]
        df_processed = _calculate_technical_indicators(sample_kline_df, price_features_to_add)
        
        # When TALIB_AVAILABLE is False, _calculate_technical_indicators should ONLY return
        # the base OHLCV columns present in the input df_processed.
        # Since sample_kline_df only has OHLCV, and SMA_10 is a TA, SMA_10 should NOT be in columns.
        assert "SMA_10" not in df_processed.columns
        assert df_processed.columns.tolist() == ['Open', 'High', 'Low', 'Close', 'Volume'] # Only base features