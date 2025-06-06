# tests/data/test_data_loader.py
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, call
from datetime import datetime, timezone

# --- UPDATED IMPORTS ---
from src.data.data_loader import load_tick_data_for_range, load_kline_data_for_range
from src.data.path_manager import get_data_path_for_day, _get_range_cache_path
from src.data.feature_engineer import calculate_technical_indicators
# --- END UPDATED IMPORTS ---

def create_dummy_df(start_time_str, num_periods, freq, columns, tz='UTC'):
    start_time = pd.to_datetime(start_time_str, utc=True)
    index = pd.date_range(start=start_time, periods=num_periods, freq=freq, tz=tz)
    data = {col: np.random.rand(num_periods) for col in columns}
    return pd.DataFrame(data, index=index)

@pytest.fixture
def mock_cache_dir(tmp_path):
    """Creates a temporary root cache directory for testing."""
    test_cache_dir = tmp_path / "test_cache"
    test_cache_dir.mkdir()
    return str(test_cache_dir)

class TestLoadDataForRangeCaching:
    SYMBOL = "BTCUSDT"
    TICK_COLUMNS = ['Price', 'Quantity', 'IsBuyerMaker']
    KLINE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI_14']
    BINANCE_SETTINGS = {"api_key": "test", "api_secret": "test", "testnet": True, "api_request_delay_seconds": 0.01}

    @pytest.fixture(autouse=True)
    def set_up(self, mock_cache_dir):
        self.mock_cache_dir = mock_cache_dir

    @patch('src.data.data_loader.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_range_cache_hit(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-01 00:00:00"; end_date_str = "2023-01-01 23:59:59"; resample_ms = 1000
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        expected_range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)
        
        dummy_range_df = create_dummy_df(start_date_str, 100, f"{resample_ms}ms", self.TICK_COLUMNS)
        
        # This mock is now safe because _get_range_cache_path doesn't create directories
        mock_exists.return_value = True
        mock_read_parquet.return_value = dummy_range_df
        
        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, tick_resample_interval_ms=resample_ms, log_level="none")
        
        mock_read_parquet.assert_called_once_with(expected_range_cache_path)
        mock_fetch_trades.assert_not_called()
        pd.testing.assert_frame_equal(result_df, dummy_range_df)

    @patch('src.data.data_loader.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_all_caches_miss_triggers_fetch(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-03 00:00:00"; end_date_str = "2023-01-03 23:59:59"; resample_ms = 1000
        mock_exists.return_value = False
        mock_read_parquet.return_value = pd.DataFrame()
        dummy_fetched_raw_df = create_dummy_df(start_date_str, 10000, "10ms", self.TICK_COLUMNS)
        mock_fetch_trades.return_value = dummy_fetched_raw_df

        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, tick_resample_interval_ms=resample_ms, log_level="none")

        mock_fetch_trades.assert_called_once()
        resampled_daily_path = get_data_path_for_day("2023-01-03", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir, resample_interval_ms=resample_ms)
        range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}, mock_cache_dir)
        
        # Check that the code attempts to save the resampled daily and the full range caches
        mock_to_parquet.assert_has_calls([call(resampled_daily_path), call(range_cache_path)], any_order=True)
        assert not result_df.empty

    @patch('src.data.data_loader.fetch_and_cache_kline_data')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_kline_range_cache_hit(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_daily_klines, mock_cache_dir):
        start_date_str = "2023-01-01 00:00:00"; end_date_str = "2023-01-01 23:59:59"; interval = "1h"; price_features = self.KLINE_COLUMNS
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "klines", "interval": interval, "features": sorted(price_features)}
        expected_range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, f"klines_{interval}", range_cache_params, mock_cache_dir)
        
        dummy_range_df = create_dummy_df(start_date_str, 24, interval, price_features)
        
        mock_exists.return_value = True
        mock_read_parquet.return_value = dummy_range_df

        result_df = load_kline_data_for_range(self.SYMBOL, start_date_str, end_date_str, interval, price_features, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, log_level="none")
        
        mock_read_parquet.assert_called_once_with(expected_range_cache_path)
        mock_fetch_daily_klines.assert_not_called()
        pd.testing.assert_frame_equal(result_df, dummy_range_df)