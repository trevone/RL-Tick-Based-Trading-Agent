# tests/data/test_data_integrity.py

import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from src.data.data_validator import (
    parse_filename_for_metadata,
    validate_daily_data,
    interval_to_timedelta
)
from src.data.path_manager import get_data_path_for_day

pytestmark = pytest.mark.order(1)

@pytest.fixture
def mock_agg_trades_df():
    start_time = datetime(2024, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 1, 23, 59, 59, 999999, tzinfo=timezone.utc)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=1000, tz='UTC')
    df = pd.DataFrame({
        'Price': np.random.rand(1000) * 1000 + 20000,
        'Quantity': np.random.rand(1000) * 10 + 1,
        'IsBuyerMaker': np.random.choice([True, False], 1000)
    }, index=timestamps)
    return df

@pytest.fixture
def mock_kline_df():
    start_time = datetime(2024, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 1, 23, 0, 0, tzinfo=timezone.utc)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1h', tz='UTC')
    df = pd.DataFrame(index=timestamps)
    df['Open'] = 100 + np.random.randn(24) * 100 + 20000
    df['High'] = np.random.rand(24) * 10 + df['Open']
    df['Low'] = df['Open'] - np.random.rand(24) * 10
    df['Close'] = np.random.rand(24) * 10 + df['Open']
    df['Volume'] = np.random.rand(24) * 1000
    df['SMA_20'] = np.random.rand(24) * 100 + 20000
    df['RSI_14'] = np.random.rand(24) * 100
    df = df.astype(float)
    return df

@pytest.fixture
def mock_cache_dir_for_integrity(tmp_path):
    return tmp_path / "integrity_cache"

def test_parse_filename_agg_trades():
    filename = "bn_aggtrades_BTCUSDT_2024-05-01.parquet"
    metadata = parse_filename_for_metadata(filename)
    assert metadata is not None
    assert metadata['data_type'] == 'agg_trades'
    assert metadata['symbol'] == 'BTCUSDT'
    assert metadata['start_time_utc'] == datetime(2024, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert metadata['end_time_utc'] == datetime(2024, 5, 1, 23, 59, 59, 999999, tzinfo=timezone.utc)
    assert 'interval' not in metadata

def test_parse_filename_kline():
    filename = "bn_klines_BTCUSDT_1h_2024-05-01_close_rsi14_sma20.parquet"
    metadata = parse_filename_for_metadata(filename)
    assert metadata is not None
    assert metadata['data_type'] == 'kline'
    assert metadata['symbol'] == 'BTCUSDT'
    assert metadata['interval'] == '1h'
    assert metadata['start_time_utc'] == datetime(2024, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert metadata['end_time_utc'] == datetime(2024, 5, 1, 23, 59, 59, 999999, tzinfo=timezone.utc)
    assert set(metadata['features']) == set(['Close', 'RSI_14', 'SMA_20', 'Open', 'High', 'Low', 'Volume'])

def test_parse_filename_invalid():
    filename = "invalid_file.txt"
    metadata = parse_filename_for_metadata(filename)
    assert metadata is None

def test_validate_daily_data_agg_trades_valid(mock_agg_trades_df, mock_cache_dir_for_integrity):
    filepath = get_data_path_for_day('2024-05-01', 'BTCUSDT', 'agg_trades', cache_dir=str(mock_cache_dir_for_integrity))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mock_agg_trades_df.to_parquet(filepath)
    is_valid, msg = validate_daily_data(filepath)
    assert is_valid
    assert "All checks passed." in msg

def test_validate_daily_data_kline_valid(mock_kline_df, mock_cache_dir_for_integrity):
    filepath = get_data_path_for_day('2024-05-01', 'BTCUSDT', 'kline', '1h', ['RSI_14', 'SMA_20'], cache_dir=str(mock_cache_dir_for_integrity))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mock_kline_df.to_parquet(filepath)
    is_valid, msg = validate_daily_data(filepath)
    assert is_valid
    assert "All checks passed." in msg

def test_validate_daily_data_empty_file(mock_cache_dir_for_integrity):
    filepath = get_data_path_for_day('2024-05-01', 'BTCUSDT', 'agg_trades', cache_dir=str(mock_cache_dir_for_integrity))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        pass
    is_valid, msg = validate_daily_data(filepath)
    assert is_valid
    assert "DataFrame is empty (0 bytes)." in msg

def test_validate_daily_data_missing_column(mock_agg_trades_df, mock_cache_dir_for_integrity):
    filepath = get_data_path_for_day('2024-05-01', 'BTCUSDT', 'agg_trades', cache_dir=str(mock_cache_dir_for_integrity))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_bad = mock_agg_trades_df.drop(columns=['Quantity'])
    df_bad.to_parquet(filepath)
    is_valid, msg = validate_daily_data(filepath)
    assert not is_valid
    assert "Missing expected columns: {'Quantity'}" in msg

def test_validate_daily_data_wrong_dtype(mock_agg_trades_df, mock_cache_dir_for_integrity):
    filepath = get_data_path_for_day('2024-05-01', 'BTCUSDT', 'agg_trades', cache_dir=str(mock_cache_dir_for_integrity))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_bad = mock_agg_trades_df.copy()
    df_bad['Price'] = df_bad['Price'].astype(str)
    df_bad.to_parquet(filepath)
    is_valid, msg = validate_daily_data(filepath)
    assert not is_valid
    assert "dtype is 'object' but expected 'float64'" in msg

def test_validate_daily_data_non_monotonic_index(mock_agg_trades_df, mock_cache_dir_for_integrity):
    filepath = get_data_path_for_day('2024-05-01', 'BTCUSDT', 'agg_trades', cache_dir=str(mock_cache_dir_for_integrity))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_bad = mock_agg_trades_df.iloc[[0, 2, 1]].copy()
    df_bad.to_parquet(filepath)
    is_valid, msg = validate_daily_data(filepath)
    assert not is_valid
    assert "Index (Timestamp) is not monotonically increasing." in msg

def test_validate_daily_data_timestamp_range_error(mock_agg_trades_df, mock_cache_dir_for_integrity):
    filepath = get_data_path_for_day('2024-05-01', 'BTCUSDT', 'agg_trades', cache_dir=str(mock_cache_dir_for_integrity))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_bad_time = mock_agg_trades_df.copy()
    df_bad_time.index = df_bad_time.index + timedelta(days=5)
    df_bad_time.to_parquet(filepath)
    is_valid, msg = validate_daily_data(filepath)
    assert not is_valid
    assert "significantly after expected start from filename" in msg

def test_validate_daily_data_kline_interval_consistency(mock_kline_df, mock_cache_dir_for_integrity):
    filepath = get_data_path_for_day('2024-05-01', 'BTCUSDT', 'kline', '1h', ['Close'], cache_dir=str(mock_cache_dir_for_integrity))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_bad_interval = mock_kline_df.iloc[::2].copy()
    df_bad_interval.to_parquet(filepath)
    is_valid, msg = validate_daily_data(filepath)
    assert not is_valid
    assert "K-line intervals are not consistently" in msg