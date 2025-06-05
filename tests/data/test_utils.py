# tests/data/test_utils.py
import pytest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone, timedelta
import time 
import yaml
import json # Added for hash tests
import hashlib # Added for hash tests

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
    RANGE_CACHE_SUBDIR,
    _get_range_cache_path,
    _generate_data_config_hash_key
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
    df['SMA_10'] = df['Close'].rolling(10).mean().bfill().ffill()
    df['RSI_14'] = pd.Series(np.random.rand(100) * 100, index=dates).bfill().ffill()
    return df

@pytest.fixture
def mock_cache_dir(tmp_path):
    test_cache_dir = tmp_path / "test_cache"
    test_cache_dir.mkdir()
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
def mock_configs_dir(tmp_path): # This fixture is essential for some new tests
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    # Create dummy config files similar to project structure
    (defaults_dir / "run_settings.yaml").write_text(
        "run_settings:\n  log_dir_base: 'logs/'\n  model_name: 'test_agent'\n"
    )
    (defaults_dir / "environment.yaml").write_text(
        "environment:\n  kline_window_size: 20\n  tick_resample_interval_ms: 1000\n  kline_price_features: ['Close', 'Volume']\n  initial_balance: 10000\n"
    )
    (defaults_dir / "ppo_params.yaml").write_text(
        "ppo_params:\n  learning_rate: 0.0003\n  n_steps: 2048\n  policy_kwargs: \"{'net_arch': [64, 64]}\"\n" # String for policy_kwargs
    )
    (defaults_dir / "sac_params.yaml").write_text(
        "sac_params:\n  learning_rate: 0.001\n  buffer_size: 100000\n"
    )
    (defaults_dir / "binance_settings.yaml").write_text(
        "binance_settings:\n  default_symbol: 'BTCUSDT'\n  historical_interval: '1h'\n"
    )
    # hash_keys.yaml is crucial for testing get_relevant_config_for_hash
    (defaults_dir / "hash_keys.yaml").write_text(
        "hash_config_keys:\n"
        "  environment:\n"
        "    - kline_window_size\n"
        "    - tick_resample_interval_ms\n"
        "    - initial_balance # Added for testing\n"
        "  agent_params:\n"
        "    PPO:\n"
        "      - learning_rate\n"
        "      - n_steps\n"
        "      - policy_kwargs\n" # Added for testing eval
        "    SAC:\n"
        "      - learning_rate\n"
        "  binance_settings:\n"
        "    - default_symbol\n"
    )
    # Main config.yaml to specify agent_type and potentially override
    (tmp_path / "config.yaml").write_text(
        "agent_type: 'PPO'\n"
        "environment:\n  initial_balance: 20000\n" # Override
    )
    return str(tmp_path) # Return root of temp config structure

def create_dummy_df(start_time_str, num_periods, freq, columns, tz='UTC'):
    start_time = pd.to_datetime(start_time_str, utc=True)
    index = pd.date_range(start=start_time, periods=num_periods, freq=freq, tz=tz)
    data = {col: np.random.rand(num_periods) for col in columns}
    df = pd.DataFrame(data, index=index)
    return df

# --- Existing Tests for _calculate_technical_indicators ---
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
        df_processed = _calculate_technical_indicators(sample_kline_df.copy(), [feature])
        assert expected_col in df_processed.columns
        assert not df_processed[expected_col].isnull().any()

def test_calculate_technical_indicators_no_talib(sample_kline_df):
    with patch('src.data.utils.TALIB_AVAILABLE', False):
        price_features_to_add = ["Close", "SMA_10"]
        df_processed = _calculate_technical_indicators(sample_kline_df.copy(), price_features_to_add)
        assert "SMA_10" not in df_processed.columns
        assert all(col in df_processed.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])


# --- Tests for Caching Logic (from previous step) ---
class TestLoadDataForRangeCaching:
    SYMBOL = "BTCUSDT"
    TICK_COLUMNS = ['Price', 'Quantity', 'IsBuyerMaker']
    KLINE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI_14']
    BINANCE_SETTINGS = {"api_key": "test", "api_secret": "test", "testnet": True, "api_request_delay_seconds": 0.01}

    @pytest.fixture(autouse=True)
    def set_up(self, mock_cache_dir):
        self.mock_cache_dir = mock_cache_dir
        os.makedirs(os.path.join(self.mock_cache_dir, RANGE_CACHE_SUBDIR), exist_ok=True)

    @patch('src.data.utils.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_range_cache_hit(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-01 00:00:00"; end_date_str = "2023-01-01 23:59:59"; resample_ms = 1000
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        expected_range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)
        dummy_range_df = create_dummy_df(start_date_str, 100, f"{resample_ms}ms", self.TICK_COLUMNS)
        mock_exists.side_effect = lambda path: path == expected_range_cache_path
        mock_read_parquet.side_effect = lambda path: dummy_range_df if path == expected_range_cache_path else pd.DataFrame()
        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, tick_resample_interval_ms=resample_ms, log_level="none")
        mock_read_parquet.assert_called_once_with(expected_range_cache_path)
        mock_fetch_trades.assert_not_called()
        pd.testing.assert_frame_equal(result_df, dummy_range_df)

    @patch('src.data.utils.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_daily_resampled_hit(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-01 00:00:00"; end_date_str = "2023-01-01 23:59:59"; resample_ms = 1000
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)
        daily_resampled_path = get_data_path_for_day("2023-01-01", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir, resample_interval_ms=resample_ms)
        dummy_daily_resampled_df = create_dummy_df(start_date_str, 86400, f"{resample_ms}ms", self.TICK_COLUMNS)
        def exists_side_effect(path): return path != range_cache_path and path == daily_resampled_path
        mock_exists.side_effect = exists_side_effect
        mock_read_parquet.side_effect = lambda path: dummy_daily_resampled_df if path == daily_resampled_path else pd.DataFrame()
        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, tick_resample_interval_ms=resample_ms, log_level="none")
        mock_read_parquet.assert_any_call(daily_resampled_path)
        mock_fetch_trades.assert_not_called()
        mock_to_parquet.assert_any_call(range_cache_path)
        pd.testing.assert_frame_equal(result_df, dummy_daily_resampled_df)

    @patch('src.data.utils.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_daily_raw_hit_resamples_and_caches(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-02 00:00:00"; end_date_str = "2023-01-02 23:59:59"; resample_ms = 1000
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)
        daily_resampled_path = get_data_path_for_day("2023-01-02", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir, resample_interval_ms=resample_ms)
        daily_raw_path = get_data_path_for_day("2023-01-02", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir)
        dummy_daily_raw_df = create_dummy_df(start_date_str, 10000, "10ms", self.TICK_COLUMNS)
        def exists_side_effect(path): return path != range_cache_path and path != daily_resampled_path and path == daily_raw_path
        mock_exists.side_effect = exists_side_effect
        mock_read_parquet.side_effect = lambda path: dummy_daily_raw_df if path == daily_raw_path else pd.DataFrame()
        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, tick_resample_interval_ms=resample_ms, log_level="none")
        mock_read_parquet.assert_any_call(daily_raw_path)
        mock_fetch_trades.assert_not_called()
        mock_to_parquet.assert_has_calls([call(daily_resampled_path), call(range_cache_path)], any_order=True)
        assert not result_df.empty

    @patch('src.data.utils.fetch_continuous_aggregate_trades')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_tick_all_caches_miss_triggers_fetch(self, mock_exists, mock_read_parquet_miss, mock_to_parquet, mock_fetch_trades, mock_cache_dir):
        start_date_str = "2023-01-03 00:00:00"; end_date_str = "2023-01-03 23:59:59"; resample_ms = 1000
        mock_exists.return_value = False
        mock_read_parquet_miss.return_value = pd.DataFrame()
        dummy_fetched_raw_df = create_dummy_df(start_date_str, 10000, "10ms", self.TICK_COLUMNS)
        mock_fetch_trades.return_value = dummy_fetched_raw_df
        result_df = load_tick_data_for_range(self.SYMBOL, start_date_str, end_date_str, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, tick_resample_interval_ms=resample_ms, log_level="none")
        mock_fetch_trades.assert_called_once()
        resampled_daily_path = get_data_path_for_day("2023-01-03", self.SYMBOL, "agg_trades", cache_dir=mock_cache_dir, resample_interval_ms=resample_ms)
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "ticks", "resample_ms": resample_ms}
        range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, "ticks", range_cache_params, mock_cache_dir)
        mock_to_parquet.assert_has_calls([call(resampled_daily_path), call(range_cache_path)], any_order=True)
        assert not result_df.empty

    @patch('src.data.utils.fetch_and_cache_kline_data')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_kline_range_cache_hit(self, mock_exists, mock_read_parquet, mock_to_parquet, mock_fetch_daily_klines, mock_cache_dir):
        start_date_str = "2023-01-01 00:00:00"; end_date_str = "2023-01-01 23:59:59"; interval = "1h"; price_features = self.KLINE_COLUMNS
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "klines", "interval": interval, "features": sorted(price_features)}
        expected_range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, f"klines_{interval}", range_cache_params, mock_cache_dir)
        dummy_range_df = create_dummy_df(start_date_str, 24, interval, price_features)
        mock_exists.side_effect = lambda path: path == expected_range_cache_path
        mock_read_parquet.side_effect = lambda path: dummy_range_df if path == expected_range_cache_path else pd.DataFrame()
        result_df = load_kline_data_for_range(self.SYMBOL, start_date_str, end_date_str, interval, price_features, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, log_level="none")
        mock_read_parquet.assert_called_once_with(expected_range_cache_path)
        mock_fetch_daily_klines.assert_not_called()
        pd.testing.assert_frame_equal(result_df, dummy_range_df)

    @patch('src.data.utils.fetch_and_cache_kline_data')
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_kline_range_cache_miss_triggers_daily_fetch(self, mock_exists, mock_read_parquet_miss, mock_to_parquet, mock_fetch_daily_klines, mock_cache_dir):
        start_date_str = "2023-01-02 00:00:00"; end_date_str = "2023-01-02 23:59:59"; interval = "1h"; price_features = self.KLINE_COLUMNS
        mock_exists.return_value = False
        mock_read_parquet_miss.return_value = pd.DataFrame()
        dummy_fetched_daily_df = create_dummy_df(start_date_str, 24, interval, price_features)
        mock_fetch_daily_klines.return_value = dummy_fetched_daily_df
        result_df = load_kline_data_for_range(self.SYMBOL, start_date_str, end_date_str, interval, price_features, cache_dir=mock_cache_dir, binance_settings=self.BINANCE_SETTINGS, log_level="none")
        mock_fetch_daily_klines.assert_called_once()
        range_cache_params = {"symbol": self.SYMBOL, "start_date": start_date_str, "end_date": end_date_str, "type": "klines", "interval": interval, "features": sorted(price_features)}
        expected_range_cache_path = _get_range_cache_path(self.SYMBOL, start_date_str, end_date_str, f"klines_{interval}", range_cache_params, mock_cache_dir)
        mock_to_parquet.assert_any_call(expected_range_cache_path)
        assert not result_df.empty
        pd.testing.assert_frame_equal(result_df, dummy_fetched_daily_df)

# --- NEW TESTS FOR CONFIG HASHING, MODEL PATH RESOLUTION ---

class TestConfigAndModelUtils:

    def test_convert_to_native_types(self):
        data = {
            "int": np.int64(5), "float": np.float64(3.14), "bool": np.bool_(True),
            "list_of_np": [np.int64(1), np.float64(2.2)],
            "array": np.array([1,2,3]),
            "timestamp": pd.Timestamp("2023-01-01T12:00:00Z")
        }
        native_data = convert_to_native_types(data)
        assert isinstance(native_data["int"], int)
        assert isinstance(native_data["float"], float)
        assert isinstance(native_data["bool"], bool)
        assert isinstance(native_data["list_of_np"][0], int)
        assert isinstance(native_data["list_of_np"][1], float)
        assert isinstance(native_data["array"], list)
        assert native_data["array"] == [1,2,3]
        assert native_data["timestamp"] == "2023-01-01T12:00:00+00:00"


    def test_generate_config_hash_consistency_and_difference(self):
        config1 = {"lr": 0.001, "arch": [64, 64], "gamma": 0.99}
        config2 = {"lr": 0.001, "arch": [64, 64], "gamma": 0.99}
        config3 = {"lr": 0.0001, "arch": [64, 64], "gamma": 0.99} # Different lr
        config4 = {"gamma": 0.99, "arch": [64, 64], "lr": 0.001} # Different order

        hash1 = generate_config_hash(config1, length=8)
        hash2 = generate_config_hash(config2, length=8)
        hash3 = generate_config_hash(config3, length=8)
        hash4 = generate_config_hash(config4, length=8)

        assert isinstance(hash1, str)
        assert len(hash1) == 8
        assert hash1 == hash2 # Consistency
        assert hash1 != hash3 # Difference
        assert hash1 == hash4 # Order invariance due to sort_keys in json.dumps

    def test_get_relevant_config_for_hash(self, mock_configs_dir, monkeypatch):
        monkeypatch.chdir(mock_configs_dir) # So load_config finds files in temp dir
        
        # Load the effective_config using the mocked config files
        # The mock_configs_dir fixture creates hash_keys.yaml, environment.yaml, ppo_params.yaml, etc.
        # and a main config.yaml that sets agent_type to PPO and overrides initial_balance.
        effective_config = load_config(main_config_path="config.yaml", 
                                       default_config_paths=[
                                           "configs/defaults/environment.yaml",
                                           "configs/defaults/ppo_params.yaml",
                                           "configs/defaults/binance_settings.yaml",
                                           "configs/defaults/hash_keys.yaml", # This defines what's relevant
                                       ])
        
        relevant_config = get_relevant_config_for_hash(effective_config)

        # Based on mock_configs_dir/defaults/hash_keys.yaml:
        # environment: kline_window_size, tick_resample_interval_ms, initial_balance
        # ppo_params: learning_rate, n_steps, policy_kwargs
        # binance_settings: default_symbol
        
        assert "environment" in relevant_config
        assert relevant_config["environment"]["kline_window_size"] == 20 # from defaults/environment.yaml
        assert relevant_config["environment"]["tick_resample_interval_ms"] == 1000 # from defaults/environment.yaml
        assert relevant_config["environment"]["initial_balance"] == 20000 # from main config.yaml (override)
        assert "kline_price_features" not in relevant_config["environment"] # Not in hash_keys

        assert "ppo_params" in relevant_config
        assert relevant_config["ppo_params"]["learning_rate"] == 0.0003 # from defaults/ppo_params.yaml
        assert relevant_config["ppo_params"]["n_steps"] == 2048
        assert relevant_config["ppo_params"]["policy_kwargs"] == {'net_arch': [64, 64]} # Eval'd from string
        assert "gamma" not in relevant_config["ppo_params"] # Not in hash_keys for PPO

        assert "binance_settings" in relevant_config
        assert relevant_config["binance_settings"]["default_symbol"] == "BTCUSDT"
        assert "historical_interval" not in relevant_config["binance_settings"]

        # Test with a different agent_type if hash_keys for SAC were different
        effective_config_sac = effective_config.copy()
        effective_config_sac["agent_type"] = "SAC"
        # Assume sac_params are loaded if testing this path properly
        effective_config_sac["sac_params"] = {"learning_rate": 0.001, "buffer_size": 50000}
        # hash_keys.yaml has SAC: [learning_rate]
        relevant_config_sac = get_relevant_config_for_hash(effective_config_sac)
        assert "sac_params" in relevant_config_sac
        assert relevant_config_sac["sac_params"]["learning_rate"] == 0.001
        assert "buffer_size" not in relevant_config_sac["sac_params"] # Not in hash_keys for SAC

    @patch('src.data.utils.os.path.exists')
    @patch('src.data.utils.get_relevant_config_for_hash')
    @patch('src.data.utils.generate_config_hash')
    def test_resolve_model_path_explicit_path_found(self, mock_gen_hash, mock_get_rel_conf, mock_exists, tmp_path):
        model_zip = tmp_path / "explicit_model.zip"
        model_zip.touch() # Create dummy file
        
        effective_config = {"run_settings": {"model_path": str(model_zip)}}
        mock_exists.return_value = True # For the explicit path

        model_path, alt_path = resolve_model_path(effective_config)
        assert model_path == str(model_zip)
        mock_exists.assert_called_once_with(str(model_zip))
        mock_get_rel_conf.assert_not_called() # Should not attempt reconstruction
        mock_gen_hash.assert_not_called()

    @patch('src.data.utils.os.path.exists')
    @patch('src.data.utils.get_relevant_config_for_hash')
    @patch('src.data.utils.generate_config_hash')
    def test_resolve_model_path_reconstruction_finds_best_model(self, mock_gen_hash, mock_get_rel_conf, mock_exists, tmp_path):
        log_dir_base = tmp_path / "logs_resolve"
        run_settings = {
            "model_path": None, 
            "model_name": "test_agent_recon",
            "log_dir_base": str(log_dir_base)
        }
        effective_config = {
            "run_settings": run_settings,
            "hash_config_keys": {}, # Assume get_relevant_config_for_hash will be mocked
            "agent_type": "PPO", # Or any other, just for path construction
            "environment": {}, "ppo_params": {} # Required for get_relevant_config_for_hash call
        }

        mock_get_rel_conf.return_value = {"some_key": "some_value"} # Dummy relevant config
        mock_gen_hash.return_value = "abcdef" # Dummy hash
        
        expected_run_dir = log_dir_base / "training" / "abcdef_test_agent_recon"
        expected_best_model_path = expected_run_dir / "best_model" / "best_model.zip"
        
        # os.path.exists should return True only for the best_model.zip
        mock_exists.side_effect = lambda path: path == str(expected_best_model_path)

        model_path, alt_path = resolve_model_path(effective_config)
        
        assert model_path == str(expected_best_model_path)
        mock_get_rel_conf.assert_called_once_with(effective_config)
        mock_gen_hash.assert_called_once_with({"some_key": "some_value"})
        # Check that os.path.exists was called for the best model path
        mock_exists.assert_any_call(str(expected_best_model_path))


# --- Tests for Config Loading Utilities (Example) ---
class TestConfigLoading:
    def test_load_single_yaml_config_valid(self, tmp_path):
        yaml_content = "key: value\nnested:\n  n_key: 123"
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        config = _load_single_yaml_config(str(p))
        assert config == {"key": "value", "nested": {"n_key": 123}}

    def test_load_single_yaml_config_missing(self):
        config = _load_single_yaml_config("non_existent_file.yaml")
        assert config == {}

    def test_merge_configs(self):
        default = {"a": 1, "b": {"x": 10, "y": 20}, "d": 100}
        loaded = {"b": {"y": 25, "z": 30}, "c": 3}
        merged = merge_configs(default, loaded)
        expected = {"a": 1, "b": {"x": 10, "y": 25, "z": 30}, "c": 3, "d": 100}
        assert merged == expected

    def test_load_config_integration(self, mock_configs_dir, monkeypatch):
        monkeypatch.chdir(mock_configs_dir) # Set CWD to where config.yaml is
        
        # mock_configs_dir creates:
        # ./config.yaml (agent_type: PPO, environment.initial_balance: 20000)
        # ./configs/defaults/environment.yaml (kline_window_size: 20, initial_balance: 10000)
        # ./configs/defaults/ppo_params.yaml (learning_rate: 0.0003)
        # ./configs/defaults/binance_settings.yaml (default_symbol: BTCUSDT)
        # ./configs/defaults/hash_keys.yaml (...)
        # ./configs/defaults/run_settings.yaml (...)


        default_paths = [
            "configs/defaults/run_settings.yaml",
            "configs/defaults/environment.yaml",
            "configs/defaults/ppo_params.yaml",
            "configs/defaults/binance_settings.yaml",
            "configs/defaults/hash_keys.yaml"
        ]
        effective_config = load_config(main_config_path="config.yaml", default_config_paths=default_paths)

        assert effective_config["agent_type"] == "PPO" # From main config.yaml
        assert effective_config["run_settings"]["model_name"] == "test_agent" # From defaults
        assert effective_config["environment"]["kline_window_size"] == 20 # From defaults
        assert effective_config["environment"]["initial_balance"] == 20000 # Overridden by main config.yaml
        assert effective_config["ppo_params"]["learning_rate"] == 0.0003 # From defaults
        assert "sac_params" not in effective_config # Not loaded