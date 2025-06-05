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
# These are the utility functions from the data module that are being tested
# or are necessary for setting up mock environments for the tests.
from src.data.utils import (
    _calculate_technical_indicators, # Function to test TA calculations
    _load_single_yaml_config,        # Helper for loading individual YAMLs
    load_config,                     # Function to test layered config loading
    fetch_and_cache_kline_data,      # Function to test K-line data fetching and caching
    fetch_continuous_aggregate_trades, # Function to test aggregate trades fetching and caching
    get_data_path_for_day,           # Helper for generating cache file paths
    load_tick_data_for_range,        # Function to test loading tick data for a range
    load_kline_data_for_range,       # Function to test loading K-line data for a range
    merge_configs,                   # Helper for merging dictionaries
    generate_config_hash,            # Helper for creating config hashes
    convert_to_native_types,         # Helper for converting NumPy types to native Python types
    get_relevant_config_for_hash,    # Helper for extracting hash-relevant config parts
    resolve_model_path,              # Function to test model path resolution logic
    DATA_CACHE_DIR                   # Global constant for the data cache directory
)

# Apply pytest-order marker to ensure this module runs after data integrity tests.
# This ensures that fundamental data utilities are verified before more complex data operations.
pytestmark = pytest.mark.order(2)

# --- Fixtures for common test data ---
# Fixtures provide reusable test data or setup routines that are shared across multiple tests
# within this module. They help maintain test independence and reduce code duplication.

@pytest.fixture(scope="module")
def sample_kline_df():
    """
    Provides a sample DataFrame representing K-line data (OHLCV) for TA calculation tests.
    This fixture has a 'module' scope, meaning it's created once for all tests in this module.
    """
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
    """
    Creates a temporary directory to serve as a mock data cache during tests.
    This ensures that tests do not interfere with actual cached data and are isolated.
    """
    test_cache_dir = tmp_path / "test_cache"
    test_cache_dir.mkdir()
    return str(test_cache_dir)

@pytest.fixture
def mock_logs_dir(tmp_path):
    """
    Creates a temporary directory structure for mock logs during tests.
    This simulates the expected logging directory layout without writing to actual log paths.
    """
    test_logs_dir = tmp_path / "test_logs"
    test_logs_dir.mkdir()
    (test_logs_dir / "training").mkdir()
    (test_logs_dir / "evaluation").mkdir()
    (test_logs_dir / "live_trading").mkdir()
    (test_logs_dir / "tensorboard_logs").mkdir()
    return str(test_logs_dir)

@pytest.fixture
def mock_configs_dir(tmp_path):
    """
    Sets up a temporary configuration directory with minimal mock YAML files.
    This simulates the project's configuration hierarchy for testing config loading utilities.
    """
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    # Create dummy config files for testing `load_config` and `get_relevant_config_for_hash`
    (defaults_dir / "run_settings.yaml").write_text("run_settings:\n  log_dir_base: 'logs/'\n  model_name: 'test_agent'\n")
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 20\n  kline_price_features: ['Close']\n")
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.001\n  total_timesteps: 100000\n")
    (defaults_dir / "sac_params.yaml").write_text("sac_params:\n  learning_rate: 0.0005\n  total_timesteps: 50000\n  buffer_size: 10000\n")
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings:\n  default_symbol: 'BTCUSDT'\n  testnet: True\n")
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size']\n  agent_params:\n    PPO: ['learning_rate', 'total_timesteps']\n    SAC: ['learning_rate', 'total_timesteps', 'buffer_size']\n  binance_settings: ['default_symbol']\n")
    
    # Create main config.yaml for testing config loading with overrides
    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n")
    
    return str(tmp_path)

# --- Tests for _calculate_technical_indicators ---
# These tests verify the functionality of adding technical indicators to K-line data,
# including behavior when TA-Lib is or is not available.

@pytest.mark.parametrize("feature, expected_col", [
    ("SMA_10", "SMA_10"),
    ("RSI_14", "RSI_14"),
    ("MACD", "MACD"),
    ("ATR", "ATR"),
    ("CDLDOJI", "CDLDOJI"),
    ("STOCH_K", "STOCH_K")
])
def test_calculate_technical_indicators_adds_features(sample_kline_df, feature, expected_col):
    """
    Tests that specified TA features are correctly added to the DataFrame when TA-Lib is available.
    Mocks TA-Lib functions to control their return values and avoid external dependency issues.
    """
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
    """
    Tests the behavior of TA calculation when TA-Lib is not installed or available.
    It should gracefully skip TA calculations and only return base OHLCV features.
    """
    with patch('src.data.utils.TALIB_AVAILABLE', False):
        # We explicitly request a TA feature here (SMA_10) that should not be added
        price_features_to_add = ["Close", "SMA_10"]
        df_processed = _calculate_technical_indicators(sample_kline_df, price_features_to_add)
        
        # When TALIB_AVAILABLE is False, _calculate_technical_indicators should ONLY return
        # the base OHLCV columns present in the input df_processed.
        assert "SMA_10" not in df_processed.columns
        assert df_processed.columns.tolist() == ['Open', 'High', 'Low', 'Close', 'Volume'] # Only base features