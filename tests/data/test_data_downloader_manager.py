# tests/data/test_data_downloader_manager.py
import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call, mock_open
from datetime import datetime, timezone, date

# Import the module to be tested and its dependencies
from src.data.data_downloader_manager import (
    download_and_manage_data,
    download_and_manage_kline_data,
    _ensure_log_file_path,
    _log_deletion_event,
    LOG_DIR_BASE_NAME,
    DATA_MANAGEMENT_LOG_FILENAME
)
from src.data.utils import DATA_CACHE_DIR as UTILS_DATA_CACHE_DIR

def create_dummy_df(columns=['Price', 'Quantity', 'IsBuyerMaker'], num_rows=5, all_zero=False):
    if all_zero:
        return pd.DataFrame(np.zeros((num_rows, len(columns))), columns=columns)
    return pd.DataFrame(np.random.rand(num_rows, len(columns)), columns=columns)

@pytest.fixture
def mock_tmp_path_for_downloader(tmp_path, monkeypatch):
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    (project_root / UTILS_DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr('src.data.data_downloader_manager._get_project_root', lambda: str(project_root))
    return project_root

class TestLoggingHelpers:
    def test_ensure_log_file_path_creates_dir_and_returns_path(self, mock_tmp_path_for_downloader):
        project_root = mock_tmp_path_for_downloader
        expected_log_dir = project_root / LOG_DIR_BASE_NAME
        expected_log_file = expected_log_dir / DATA_MANAGEMENT_LOG_FILENAME
        log_file_path = _ensure_log_file_path()
        assert os.path.isdir(expected_log_dir)
        assert log_file_path == str(expected_log_file)

    def test_log_deletion_event_writes_correct_format(self, mock_tmp_path_for_downloader):
        log_file_path = _ensure_log_file_path()
        test_file_path = "/dummy/cache/some_file.parquet"
        test_reason = "File was 0 bytes."
        if os.path.exists(log_file_path): os.remove(log_file_path)
        _log_deletion_event(test_file_path, test_reason)
        assert os.path.exists(log_file_path)
        with open(log_file_path, "r") as f: content = f.read()
        assert "Event:     DELETED CACHED FILE" in content
        assert f"File Path: {test_file_path}" in content
        assert f"Reason:    {test_reason}" in content

    def test_log_deletion_event_complex_reason(self, mock_tmp_path_for_downloader):
        log_file_path = _ensure_log_file_path()
        if os.path.exists(log_file_path): os.remove(log_file_path)
        test_file_path = "/dummy/cache/another.parquet"
        complex_reason = "Validation failed: One or more checks FAILED:\n  - Index is not UTC-aware.\n  - Missing columns: {'X'}"
        _log_deletion_event(test_file_path, complex_reason)
        with open(log_file_path, "r") as f: content = f.read()
        assert "Reason:    Validation Failed" in content
        assert "Details:" in content
        assert "Index is not UTC-aware." in content

@patch('src.data.data_downloader_manager._log_deletion_event')
@patch('src.data.data_downloader_manager.os.path.exists')
@patch('src.data.data_downloader_manager.os.path.getsize')
@patch('src.data.data_downloader_manager.os.remove')
@patch('src.data.data_downloader_manager.pd.read_parquet')
@patch('src.data.data_downloader_manager.get_data_path_for_day')
@patch('src.data.data_downloader_manager.fetch_and_cache_tick_data')
@patch('src.data.data_downloader_manager.validate_daily_data')
class TestDownloadAndManageAggTrades:

    def run_manager(self, mock_validate, mock_fetch_ticks, mock_get_data_path, 
                    mock_read_parquet, mock_remove, mock_getsize, mock_exists, # Order changed to match decorator order
                    exists_side_effect=None, # MODIFIED: Renamed and can be a list
                    getsize_behavior=100, read_parquet_behavior=create_dummy_df(), 
                    fetch_return=create_dummy_df(), validate_return=(True, "Valid.")):
        
        if exists_side_effect is not None:
            mock_exists.side_effect = exists_side_effect
        else: # Default if not provided, useful for tests where it's always True or False
            mock_exists.return_value = True 
            
        mock_getsize.return_value = getsize_behavior
        if isinstance(read_parquet_behavior, Exception):
            mock_read_parquet.side_effect = read_parquet_behavior
        else:
            mock_read_parquet.return_value = read_parquet_behavior
        
        # Make get_data_path_for_day return a consistent dummy path for the tests
        mock_get_data_path.return_value = os.path.join("dummy_cache_dir", "dummy_file.parquet")
        
        mock_fetch_ticks.return_value = fetch_return
        mock_validate.return_value = validate_return

        download_and_manage_data("2023-01-01", "2023-01-01", "BTCUSDT")

    def test_dmd_agg_trades_file_exists_valid_and_validated(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists, 
                         exists_side_effect=[True, True], # File exists before and after read attempt
                         getsize_behavior=100, 
                         read_parquet_behavior=create_dummy_df(),
                         validate_return=(True, "All checks passed."))

        mock_get_data_path.assert_called_with("2023-01-01", "BTCUSDT", data_type="agg_trades", cache_dir=UTILS_DATA_CACHE_DIR)
        mock_fetch_ticks.assert_not_called()
        mock_remove.assert_not_called()
        mock_log_del.assert_not_called()
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_file_exists_0_bytes_deletes_and_fetches(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists,
                         exists_side_effect=[True, True, True], # 1. Exists (for getsize), 2. Exists (after remove, for fetch re-creation check - this will be True after mock fetch implicitly "creates" it for validation path)
                         getsize_behavior=0) # 0 bytes

        mock_remove.assert_called_once()
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, "File was 0 bytes.")
        mock_fetch_ticks.assert_called_once()
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_file_unreadable_deletes_and_fetches(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        read_error = pd.errors.EmptyDataError("File is empty/corrupt")
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists,
                         exists_side_effect=[True, True, True], 
                         getsize_behavior=100,
                         read_parquet_behavior=read_error)

        mock_remove.assert_called_once()
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, f"File unreadable/corrupt: {read_error}")
        mock_fetch_ticks.assert_called_once()
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_file_empty_after_read_deletes_and_fetches(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists,
                         exists_side_effect=[True, True, True],
                         getsize_behavior=100,
                         read_parquet_behavior=pd.DataFrame())

        mock_remove.assert_called_once()
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, "File was empty after reading.")
        mock_fetch_ticks.assert_called_once()
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_file_not_exists_fetches( # One of the failing tests
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        # mock_exists should return False first, then True after the mocked fetch "creates" the file
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists,
                         exists_side_effect=[False, True]) # MODIFIED

        mock_remove.assert_not_called()
        mock_fetch_ticks.assert_called_once()
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_fetch_successful_invalid_data_deletes_and_logs( # One of the failing tests
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        validation_msg = "Validation failed: Bad data."
        # mock_exists should return False first (trigger fetch), then True (file "created" by fetch)
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists,
                         exists_side_effect=[False, True], # MODIFIED
                         validate_return=(False, validation_msg))

        mock_fetch_ticks.assert_called_once()
        mock_validate.assert_called_once()
        mock_remove.assert_called_once_with(mock_get_data_path.return_value)
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, f"Validation failed: {validation_msg}")


@patch('src.data.data_downloader_manager._log_deletion_event')
@patch('src.data.data_downloader_manager.os.path.exists')
@patch('src.data.data_downloader_manager.os.path.getsize')
@patch('src.data.data_downloader_manager.os.remove')
@patch('src.data.data_downloader_manager.pd.read_parquet')
@patch('src.data.data_downloader_manager.get_data_path_for_day')
@patch('src.data.data_downloader_manager.fetch_and_cache_kline_data') # Patched correctly
@patch('src.data.data_downloader_manager.validate_daily_data')
class TestDownloadAndManageKlines:

    def run_manager_kline(self, mock_validate, mock_fetch_klines, mock_get_data_path, 
                          mock_read_parquet, mock_remove, mock_getsize, mock_exists, # Order changed
                          exists_side_effect=None, # MODIFIED: Renamed and can be a list
                          getsize_behavior=100, read_parquet_behavior=create_dummy_df(columns=['Open', 'Close']), 
                          fetch_return=create_dummy_df(columns=['Open', 'Close']), validate_return=(True, "Valid.")):
        
        if exists_side_effect is not None:
            mock_exists.side_effect = exists_side_effect
        else:
            mock_exists.return_value = True

        mock_getsize.return_value = getsize_behavior
        if isinstance(read_parquet_behavior, Exception):
            mock_read_parquet.side_effect = read_parquet_behavior
        else:
            mock_read_parquet.return_value = read_parquet_behavior
        
        mock_get_data_path.return_value = os.path.join("dummy_cache_dir", "dummy_kline_file.parquet")
        mock_fetch_klines.return_value = fetch_return
        mock_validate.return_value = validate_return

        download_and_manage_kline_data("2023-01-01", "2023-01-01", "BTCUSDT", "1h", ["Open", "Close"])

    def test_dmd_klines_file_exists_valid_and_validated(
        self, mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        self.run_manager_kline(mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
                               mock_remove, mock_getsize, mock_exists,
                               exists_side_effect=[True, True], 
                               read_parquet_behavior=create_dummy_df(columns=['Open', 'Close']),
                               validate_return=(True, "All checks passed."))

        mock_get_data_path.assert_called_with("2023-01-01", "BTCUSDT", data_type="kline", interval="1h", price_features_to_add=["Open", "Close"], cache_dir=UTILS_DATA_CACHE_DIR)
        mock_fetch_klines.assert_not_called()
        mock_remove.assert_not_called()
        mock_log_del.assert_not_called()
        mock_validate.assert_called_once()

    def test_dmd_klines_file_not_exists_fetches( # One of the failing tests
        self, mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        self.run_manager_kline(mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
                               mock_remove, mock_getsize, mock_exists,
                               exists_side_effect=[False, True]) # MODIFIED

        mock_remove.assert_not_called()
        mock_fetch_klines.assert_called_once()
        mock_validate.assert_called_once()

    def test_dmd_klines_fetch_successful_invalid_data_deletes_and_logs( # One of the failing tests
        self, mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet, 
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_tmp_path_for_downloader
    ):
        validation_msg = "Validation failed: Klines bad."
        self.run_manager_kline(mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
                               mock_remove, mock_getsize, mock_exists,
                               exists_side_effect=[False, True], # MODIFIED
                               validate_return=(False, validation_msg))

        mock_fetch_klines.assert_called_once()
        mock_validate.assert_called_once()
        mock_remove.assert_called_once_with(mock_get_data_path.return_value)
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, f"Validation failed: {validation_msg}")