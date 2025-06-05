# tests/data/test_utils.py
import pytest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone, timedelta
import time 
import yaml # For new tests involving config loading for binance_settings

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
    DATA_CACHE_DIR,                  
    RANGE_CACHE_SUBDIR,              # Import for constructing paths in tests
    _get_range_cache_path,           # Import for constructing paths in tests
    _generate_data_config_hash_key   # Import for constructing paths in tests
)

pytestmark = pytest.mark.order(2)

# --- Fixtures for common test data ---

@pytest.fixture(scope="module")
def sample_kline_df():
    dates = pd.date_range(start="2023-01-01", periods=100, freq='1h', tz='UTC')
    df = pd.DataFrame(index=dates)
    df['Open'] = np.random.rand(100) * 100 + 1000
    df['High'] = df['Open'] + np.random.rand(100) * 5
    df['Low'] = df['Open'] - np.random.rand(100) * 5
    df['Close'] = df['Open'] + (np.random.rand(100) - 0.5) * 5
    df['Volume'] = np.random.rand(100) * 1000
    # Add some TA features that might be requested
    df['SMA_10'] = df['Close'].rolling(10).mean().bfill().ffill()
    df['RSI_14'] = pd.Series(np.random.rand(100) * 100, index=dates).bfill().ffill()
    return df

@pytest.fixture
def mock_cache_dir(tmp_path):
    test_cache_dir = tmp_path / "test_cache"
    test_cache_dir.mkdir()
    # Create the RANGE_CACHE_SUBDIR inside the mock_cache_dir for tests
    (test_cache_dir / RANGE_CACHE_SUBDIR).mkdir(exist_ok=True)
    return str(test_cache_dir)

@pytest.fixture
def mock_logs_dir(tmp_path):
    test_logs_dir = tmp_path / "test_logs"
    test_logs_dir.mkdir()
    (test_logs_dir / "training").mkdir()
    (test_logs_dir / "evaluation").mkdir()
    (test_logs_dir / "live_trading").mkdir()
    (test_logs_dir / "tensorboard_logs").mkdir()
    return str(test_logs_dir)

@pytest.fixture
def mock_configs_dir(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()
    (defaults_dir / "run_settings.yaml").write_text("run_settings:\n  log_dir_base: 'logs/'\n  model_name: 'test_agent'\n")
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 20\n  kline_price_features: ['Close']\n  tick_resample_interval_ms: 1000\n") # Added tick_resample_interval_ms
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.001\n  total_timesteps: 100000\n")
    (defaults_dir / "sac_params.yaml").write_text("sac_params:\n  learning_rate: 0.0005\n  total_timesteps: 50000\n  buffer_size: 10000\n")
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings:\n  default_symbol: 'BTCUSDT'\n  testnet: True\n  historical_cache_dir: 'test_cache/'\n") # Point to a testable dir
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size', 'tick_resample_interval_ms']\n  agent_params:\n    PPO: ['learning_rate', 'total_timesteps']\n    SAC: ['learning_rate', 'total_timesteps', 'buffer_size']\n  binance_settings: ['default_symbol']\n")
    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n")
    return str(tmp_path)

# --- Helper for creating dummy DataFrames for testing caching ---
def create_dummy_df(start_time_str, num_periods, freq, columns, tz='UTC'):
    start_time = pd.to_datetime(start_time_str, utc=True)
    index = pd.date_range(start=start_time, periods=num_periods, freq=freq, tz=tz)
    data = {col: np.random.rand(num_periods) for col in columns}
    df = pd.DataFrame(data, index=index)
    return df

# --- Tests for _calculate_technical_indicators ---
@pytest.mark.parametrize("feature, expected_col", [
    ("SMA_10", "SMA_10"), ("RSI_14", "RSI_14"), ("MACD", "MACD"),
    ("ATR", "ATR"), ("CDLDOJI", "CDLDOJI"), ("STOCH_K", "STOCH_K")
])
def test_calculate_technical_indicators_adds_features(sample_kline_df, feature, expected_col):
    with patch('src.data.utils.TALIB_AVAILABLE', True), \
         patch('src.data.utils.talib.SMA', return_value=np.arange(len(sample_kline_df))), \
         patch('src.data.utils.talib.RSI', return_value=np.arange(len(sample_kline_df))), \
         patch('src.data.utils.talib.MACD', return_value=(np.arange(len(sample_kline_df)), None, None)), \
         patch('src.data.utils.talib.ATR', return_value=np.arange(len(sample_kline_df))), \
         patch('src.data.utils.talib.CDLDOJI', return_value=np.arange(len(sample_kline_df))), \
         patch('src.data.utils.talib.STOCH', return_value=(np.arange(len(sample_kline_df)), None)):
        df_processed = _calculate_technical_indicators(sample_kline_df.copy(), [feature]) # Pass copy
        assert expected_col in df_processed.columns
        assert not df_processed[expected_col].isnull().any()

def test_calculate_technical_indicators_no_talib(sample_kline_df):
    with patch('src.data.utils.TALIB_AVAILABLE', False):
        price_features_to_add = ["Close", "SMA_10"]
        df_processed = _calculate_technical_indicators(sample_kline_df.copy(), price_features_to_add) # Pass copy
        assert "SMA_10" not in df_processed.columns
        assert all(col in df_processed.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])


# --- NEW TEST CLASS FOR CACHING LOGIC ---
class TestLoadDataForRangeCaching:

    SYMBOL = "BTCUSDT"
    TICK_COLUMNS = ['Price', 'Quantity', 'IsBuyerMaker']
    KLINE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI_14'] # Example features
    BINANCE_SETTINGS = {"api_key": "test", "api_secret": "test", "testnet": True, "api_request_delay_seconds": 0.01}

    @pytest.fixture(autouse=True)
    def set_up(self, mock_cache_dir):
        self.mock_cache_dir = mock_cache_dir
        # Ensure range_cache subdirectory exists within the temp test cache
        os.makedirs(os.path.join(self.mock_cache_dir, RANGE_CACHE_SUBDIR), exist_ok=True)


    # --- Tests for load_tick_data_for_range ---
    @patch('src.data.utils.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_range_cache_hit(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-01 00:00:00"
        end_date_str = "2023-01-01 23:59:59"
        resample_ms = 1000

        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        expected_range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)
        
        dummy_range_df = create_dummy_df(start_date_str, 100, f"{resample_ms}ms", self.TICK_COLUMNS)

        mock_exists.side_effect = lambda path: path == expected_range_cache_path
        mock_read_parquet.side_effect = lambda path: dummy_range_df if path == expected_range_cache_path else pd.DataFrame()
        
        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str,
                                             cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS,
                                             tick_resample_interval_ms=resample_ms, log_level="none")

        mock_read_parquet.assert_called_once_with(expected_range_cache_path)
        mock_fetch_trades.assert_not_called()
        pd.testing.assert_frame_equal(result_df, dummy_range_df)

    @patch('src.data.utils.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_daily_resampled_hit(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-01 00:00:00"
        end_date_str = "2023-01-01 23:59:59" # Single day
        resample_ms = 1000

        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)
        daily_resampled_path = get_data_path_for_day("2023-01-01", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir, resample_interval_ms=resample_ms)
        
        dummy_daily_resampled_df = create_dummy_df(start_date_str, 86400, f"{resample_ms}ms", self.TICK_COLUMNS) # 1 day of 1s data

        def exists_side_effect(path):
            if path == range_cache_path: return False
            if path == daily_resampled_path: return True
            return False
        mock_exists.side_effect = exists_side_effect
        mock_read_parquet.side_effect = lambda path: dummy_daily_resampled_df if path == daily_resampled_path else pd.DataFrame()

        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str,
                                             cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS,
                                             tick_resample_interval_ms=resample_ms, log_level="none")
        
        mock_read_parquet.assert_any_call(daily_resampled_path)
        mock_fetch_trades.assert_not_called()
        # Check that new range cache was saved
        mock_to_parquet.assert_any_call(range_cache_path)
        pd.testing.assert_frame_equal(result_df, dummy_daily_resampled_df)


    @patch('src.data.utils.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_daily_raw_hit_resamples_and_caches(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-02 00:00:00"
        end_date_str = "2023-01-02 23:59:59" # Single day
        resample_ms = 1000

        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)
        daily_resampled_path = get_data_path_for_day("2023-01-02", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir, resample_interval_ms=resample_ms)
        daily_raw_path = get_data_path_for_day("2023-01-02", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir) # Raw

        dummy_daily_raw_df = create_dummy_df(start_date_str, 10000, "10ms", self.TICK_COLUMNS) # Higher frequency raw data

        def exists_side_effect(path):
            if path == range_cache_path: return False
            if path == daily_resampled_path: return False
            if path == daily_raw_path: return True
            return False
        mock_exists.side_effect = exists_side_effect
        mock_read_parquet.side_effect = lambda path: dummy_daily_raw_df if path == daily_raw_path else pd.DataFrame()

        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str,
                                             cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS,
                                             tick_resample_interval_ms=resample_ms, log_level="none")

        mock_read_parquet.assert_any_call(daily_raw_path)
        mock_fetch_trades.assert_not_called()
        # Assert that the new daily resampled cache was saved AND the new range cache was saved
        calls = [call(daily_resampled_path), call(range_cache_path)]
        mock_to_parquet.assert_has_calls(calls, any_order=True)
        assert not result_df.empty
        # Further check on result_df content if needed, e.g., number of rows matching resampled freq


    @patch('src.data.utils.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet') # Mock read_parquet even for miss case to avoid trying to read non-existent files
    @patch('os.path.exists')
    def test_tick_all_caches_miss_triggers_fetch(self, mock_exists, mock_read_parquet_miss, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-03 00:00:00"
        end_date_str = "2023-01-03 23:59:59" # Single day
        resample_ms = 1000

        # All cache files do not exist
        mock_exists.return_value = False
        mock_read_parquet_miss.return_value = pd.DataFrame() # Ensure read returns empty if somehow called

        # Mock the fetch function to return some data
        dummy_fetched_raw_df = create_dummy_df(start_date_str, 10000, "10ms", self.TICK_COLUMNS)
        mock_fetch_trades.return_value = dummy_fetched_raw_df

        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str,
                                             cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS,
                                             tick_resample_interval_ms=resample_ms, log_level="none")

        mock_fetch_trades.assert_called_once()
        # Assert that raw daily, resampled daily, and range caches were saved
        raw_daily_path = get_data_path_for_day("2023-01-03", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir)
        resampled_daily_path = get_data_path_for_day("2023-01-03", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir, resample_interval_ms=resample_ms)
        
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)

        # Note: fetch_continuous_aggregate_trades itself saves the raw_daily_path.
        # So, load_tick_data_for_range will save resampled_daily_path and range_cache_path.
        calls = [call(resampled_daily_path), call(range_cache_path)]
        mock_to_parquet.assert_has_calls(calls, any_order=True)
        assert not result_df.empty

    # --- Tests for load_kline_data_for_range ---
    @patch('src.data.utils.fetch_and_cache_kline_data') # This is the daily fetcher for klines
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_kline_range_cache_hit(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_daily_klines, mock_cache_dir):
        start_date_str = "2023-01-01 00:00:00"
        end_date_str = "2023-01-01 23:59:59"
        interval = "1h"
        price_features = self.KLINE_COLUMNS

        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "klines", "interval": interval, "features": sorted(price_features)}
        expected_range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, f"klines_{interval}", range_cache_params, mock_cache_dir)
        
        dummy_range_df = create_dummy_df(start_date_str, 24, interval, price_features)

        mock_exists.side_effect = lambda path: path == expected_range_cache_path
        mock_read_parquet.side_effect = lambda path: dummy_range_df if path == expected_range_cache_path else pd.DataFrame()
        
        result_df = load_kline_data_for_range(self.SYMBOL, start_date_str, end_date_str, interval, price_features,
                                              cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, log_level="none")

        mock_read_parquet.assert_called_once_with(expected_range_cache_path)
        mock_fetch_daily_klines.assert_not_called()
        pd.testing.assert_frame_equal(result_df, dummy_range_df)


    @patch('src.data.utils.fetch_and_cache_kline_data') # This is the daily fetcher
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet') # Mock read_parquet for completeness
    @patch('os.path.exists')
    def test_kline_range_cache_miss_triggers_daily_fetch(self, mock_exists, mock_read_parquet_miss, mock_to_parquet, mock_fetch_daily_klines, mock_cache_dir):
        start_date_str = "2023-01-02 00:00:00"
        end_date_str = "2023-01-02 23:59:59" # Single day
        interval = "1h"
        price_features = self.KLINE_COLUMNS

        mock_exists.return_value = False # No range cache, no daily caches
        mock_read_parquet_miss.return_value = pd.DataFrame()

        dummy_fetched_daily_df = create_dummy_df(start_date_str, 24, interval, price_features)
        mock_fetch_daily_klines.return_value = dummy_fetched_daily_df

        result_df = load_kline_data_for_range(self.SYMBOL, start_date_str, end_date_str, interval, price_features,
                                              cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, log_level="none")

        mock_fetch_daily_klines.assert_called_once() # Called for the one day in range
        
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "klines", "interval": interval, "features": sorted(price_features)}
        expected_range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, f"klines_{interval}", range_cache_params, mock_cache_dir)
        mock_to_parquet.assert_any_call(expected_range_cache_path) # Check that new range cache was saved
        assert not result_df.empty
        pd.testing.assert_frame_equal(result_df, dummy_fetched_daily_df)