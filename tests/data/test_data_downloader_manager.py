# tests/data/test_data_downloader_manager.py
import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call, mock_open
from datetime import datetime, timezone, date
import shutil

# Import the module to be tested and its dependencies
from src.data.data_downloader_manager import (
    download_and_manage_data,
    download_and_manage_kline_data,
    _ensure_log_file_path,
    _log_deletion_event,
    LOG_DIR_BASE_NAME,
    DATA_MANAGEMENT_LOG_FILENAME,
)

def create_dummy_df(columns=['Price', 'Quantity', 'IsBuyerMaker'], num_rows=5, all_zero=False):
    if all_zero:
        return pd.DataFrame(np.zeros((num_rows, len(columns))), columns=columns)
    return pd.DataFrame(np.random.rand(num_rows, len(columns)), columns=columns)

@pytest.fixture
def mock_tmp_path_for_downloader(tmp_path, monkeypatch):
    project_root = tmp_path / "project_root_downloader"
    project_root.mkdir(exist_ok=True)
    monkeypatch.setattr('src.data.data_downloader_manager._get_project_root', lambda: str(project_root))
    return project_root

@pytest.fixture
def mock_downloader_configs_fixture(tmp_path, monkeypatch):
    """
    Creates a temporary configuration directory structure and yields the
    path that will be configured as 'historical_cache_dir'.
    This fixture also handles chdir for load_config to work correctly.
    """
    config_root_for_test = tmp_path / "downloader_test_configs"
    config_root_for_test.mkdir()

    defaults_dir = config_root_for_test / "configs" / "defaults"
    defaults_dir.mkdir(parents=True)

    test_cache_dir_via_config = str(tmp_path / "test_downloader_cache_from_config")
    os.makedirs(test_cache_dir_via_config, exist_ok=True)

    # UPDATED: historical_cache_dir is now in run_settings.yaml
    (defaults_dir / "run_settings.yaml").write_text(
        f"run_settings:\n  historical_cache_dir: '{test_cache_dir_via_config}'\n  log_level: 'none'\n"
    )
    (defaults_dir / "binance_settings.yaml").write_text(
        "binance_settings:\n  api_key: \"mock_api_key\"\n  api_secret: \"mock_api_secret\"\n  testnet: true\n  api_request_delay_seconds: 0.001\n"
    )
    (config_root_for_test / "config.yaml").write_text("# Main config for downloader tests (can be empty to use defaults)\n")

    original_cwd = os.getcwd()
    monkeypatch.chdir(config_root_for_test)
    
    yield test_cache_dir_via_config

    monkeypatch.chdir(original_cwd)

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


@patch('src.data.data_downloader_manager.load_configs_for_data_management')
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
                    mock_read_parquet, mock_remove, mock_getsize, mock_exists,
                    mock_log_del, mock_load_configs,
                    configured_cache_dir: str,
                    exists_side_effect=None,
                    getsize_behavior=100, read_parquet_behavior=create_dummy_df(),
                    fetch_return=create_dummy_df(), validate_return=(True, "Valid.")):

        # FIXED: Mock configuration now separates run_settings and binance_settings
        mock_load_configs.return_value = {
            "run_settings": {
                "historical_cache_dir": configured_cache_dir,
            },
            "binance_settings": {
                "api_key": "mock_api_key",
                "api_secret": "mock_api_secret",
                "testnet": True,
                "api_request_delay_seconds": 0.001
            }
        }

        if exists_side_effect is not None:
            mock_exists.side_effect = exists_side_effect
        else:
            mock_exists.return_value = True

        mock_getsize.return_value = getsize_behavior
        if isinstance(read_parquet_behavior, Exception):
            mock_read_parquet.side_effect = read_parquet_behavior
        else:
            mock_read_parquet.return_value = read_parquet_behavior

        mock_get_data_path.return_value = os.path.join(configured_cache_dir, "dummy_aggtrade_file.parquet")

        mock_fetch_ticks.return_value = fetch_return
        mock_validate.return_value = validate_return

        download_and_manage_data("2023-01-01", "2023-01-01", "BTCUSDT")

    def test_dmd_agg_trades_file_exists_valid_and_validated(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                         configured_cache_dir=test_cache_dir,
                         exists_side_effect=[True, True],
                         getsize_behavior=100,
                         read_parquet_behavior=create_dummy_df(),
                         validate_return=(True, "All checks passed."))

        mock_load_configs.assert_called_once()
        # FIXED: The downloader script now correctly gets cache_dir from config and passes it to utils
        mock_get_data_path.assert_called_with("2023-01-01", "BTCUSDT", data_type="agg_trades", cache_dir=test_cache_dir)
        mock_fetch_ticks.assert_not_called()
        mock_remove.assert_not_called()
        mock_log_del.assert_not_called()
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_file_exists_0_bytes_deletes_and_fetches(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                         configured_cache_dir=test_cache_dir,
                         exists_side_effect=[True, True, True],
                         getsize_behavior=0)

        mock_fetch_ticks.assert_called_once()
        args, kwargs = mock_fetch_ticks.call_args
        assert kwargs.get('cache_dir') == test_cache_dir
        mock_remove.assert_called_once()
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, "File was 0 bytes.")
        mock_validate.assert_called_once()


    def test_dmd_agg_trades_file_unreadable_deletes_and_fetches(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        read_error = pd.errors.EmptyDataError("File is empty/corrupt")
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                         configured_cache_dir=test_cache_dir,
                         exists_side_effect=[True, True, True],
                         getsize_behavior=100,
                         read_parquet_behavior=read_error)

        mock_fetch_ticks.assert_called_once()
        args, kwargs = mock_fetch_ticks.call_args
        assert kwargs.get('cache_dir') == test_cache_dir
        mock_remove.assert_called_once()
        mock_log_del.assert_called_once_with(os.path.join(test_cache_dir, "dummy_aggtrade_file.parquet"), f"File unreadable/corrupt: {read_error}")
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_file_empty_after_read_deletes_and_fetches(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                         configured_cache_dir=test_cache_dir,
                         exists_side_effect=[True, True, True],
                         getsize_behavior=100,
                         read_parquet_behavior=pd.DataFrame())

        mock_fetch_ticks.assert_called_once()
        args, kwargs = mock_fetch_ticks.call_args
        assert kwargs.get('cache_dir') == test_cache_dir
        mock_remove.assert_called_once()
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, "File was empty after reading.")
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_file_not_exists_fetches(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                         configured_cache_dir=test_cache_dir,
                         exists_side_effect=[False, True])

        mock_fetch_ticks.assert_called_once()
        args, kwargs = mock_fetch_ticks.call_args
        assert kwargs.get('cache_dir') == test_cache_dir
        mock_remove.assert_not_called()
        mock_validate.assert_called_once()

    def test_dmd_agg_trades_fetch_successful_invalid_data_deletes_and_logs(
        self, mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        validation_msg = "Validation failed: Bad data."
        self.run_manager(mock_validate, mock_fetch_ticks, mock_get_data_path, mock_read_parquet,
                         mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                         configured_cache_dir=test_cache_dir,
                         exists_side_effect=[False, True],
                         validate_return=(False, validation_msg))

        mock_fetch_ticks.assert_called_once()
        args, kwargs = mock_fetch_ticks.call_args
        assert kwargs.get('cache_dir') == test_cache_dir
        mock_validate.assert_called_once()
        mock_remove.assert_called_once_with(mock_get_data_path.return_value)
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, f"Validation failed: {validation_msg}")


@patch('src.data.data_downloader_manager.load_configs_for_data_management')
@patch('src.data.data_downloader_manager._log_deletion_event')
@patch('src.data.data_downloader_manager.os.path.exists')
@patch('src.data.data_downloader_manager.os.path.getsize')
@patch('src.data.data_downloader_manager.os.remove')
@patch('src.data.data_downloader_manager.pd.read_parquet')
@patch('src.data.data_downloader_manager.get_data_path_for_day')
@patch('src.data.data_downloader_manager.fetch_and_cache_kline_data')
@patch('src.data.data_downloader_manager.validate_daily_data')
class TestDownloadAndManageKlines:

    def run_manager_kline(self, mock_validate, mock_fetch_klines, mock_get_data_path,
                          mock_read_parquet, mock_remove, mock_getsize, mock_exists,
                          mock_log_del, mock_load_configs,
                          configured_cache_dir: str,
                          exists_side_effect=None,
                          getsize_behavior=100, read_parquet_behavior=create_dummy_df(columns=['Open', 'Close']),
                          fetch_return=create_dummy_df(columns=['Open', 'Close']), validate_return=(True, "Valid.")):

        # FIXED: Mock configuration now separates run_settings and binance_settings
        mock_load_configs.return_value = {
            "run_settings": {
                "historical_cache_dir": configured_cache_dir,
            },
            "binance_settings": {
                "api_key": "mock_api_key",
                "api_secret": "mock_api_secret",
                "testnet": True,
                "api_request_delay_seconds": 0.001
            }
        }

        if exists_side_effect is not None:
            mock_exists.side_effect = exists_side_effect
        else:
            mock_exists.return_value = True

        mock_getsize.return_value = getsize_behavior
        if isinstance(read_parquet_behavior, Exception):
            mock_read_parquet.side_effect = read_parquet_behavior
        else:
            mock_read_parquet.return_value = read_parquet_behavior

        mock_get_data_path.return_value = os.path.join(configured_cache_dir, "dummy_kline_file.parquet")
        mock_fetch_klines.return_value = fetch_return
        mock_validate.return_value = validate_return

        download_and_manage_kline_data("2023-01-01", "2023-01-01", "BTCUSDT", "1h", ["Open", "Close"])

    def test_dmd_klines_file_exists_valid_and_validated(
        self, mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        self.run_manager_kline(mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
                               mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                               configured_cache_dir=test_cache_dir,
                               exists_side_effect=[True, True],
                               read_parquet_behavior=create_dummy_df(columns=['Open', 'Close']),
                               validate_return=(True, "All checks passed."))

        mock_load_configs.assert_called_once()
        mock_get_data_path.assert_called_with("2023-01-01", "BTCUSDT", data_type="kline", interval="1h", price_features_to_add=["Open", "Close"], cache_dir=test_cache_dir)
        mock_fetch_klines.assert_not_called()
        mock_remove.assert_not_called()
        mock_log_del.assert_not_called()
        mock_validate.assert_called_once()

    def test_dmd_klines_file_not_exists_fetches(
        self, mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        self.run_manager_kline(mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
                               mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                               configured_cache_dir=test_cache_dir,
                               exists_side_effect=[False, True])

        mock_fetch_klines.assert_called_once()
        args, kwargs = mock_fetch_klines.call_args
        assert kwargs.get('cache_dir') == test_cache_dir
        mock_remove.assert_not_called()
        mock_validate.assert_called_once()

    def test_dmd_klines_fetch_successful_invalid_data_deletes_and_logs(
        self, mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
        mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
        mock_downloader_configs_fixture
    ):
        test_cache_dir = mock_downloader_configs_fixture
        validation_msg = "Validation failed: Klines bad."
        self.run_manager_kline(mock_validate, mock_fetch_klines, mock_get_data_path, mock_read_parquet,
                               mock_remove, mock_getsize, mock_exists, mock_log_del, mock_load_configs,
                               configured_cache_dir=test_cache_dir,
                               exists_side_effect=[False, True],
                               validate_return=(False, validation_msg))

        mock_fetch_klines.assert_called_once()
        args, kwargs = mock_fetch_klines.call_args
        assert kwargs.get('cache_dir') == test_cache_dir
        mock_validate.assert_called_once()
        mock_remove.assert_called_once_with(mock_get_data_path.return_value)
        mock_log_del.assert_called_once_with(mock_get_data_path.return_value, f"Validation failed: {validation_msg}")