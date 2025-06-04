# tests/data/test_utils.py
import pytest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

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
    """Provides a sample DataFrame for TA calculation."""
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
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings:\n  default_symbol: 'BTCUSDT'\n  testnet: True\n")
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size']\n  agent_params:\n    PPO: ['learning_rate']\n  binance_settings: ['default_symbol']\n")
    
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
        df_processed = _calculate_technical_indicators(sample_kline_df, ["SMA_10"])
        assert "SMA_10" not in df_processed.columns # Should not add if TA-Lib isn't there
        assert 'Close' in df_processed.columns # Base columns remain

# --- Tests for config loading ---
def test_load_single_yaml_config_exists(tmp_path):
    """Test _load_single_yaml_config when file exists."""
    test_file = tmp_path / "test.yaml"
    test_file.write_text("key: value\nlist: [1, 2]\n")
    config = _load_single_yaml_config(str(test_file))
    assert config == {"key": "value", "list": [1, 2]}

def test_load_single_yaml_config_not_exists(tmp_path):
    """Test _load_single_yaml_config when file does not exist."""
    config = _load_single_yaml_config(str(tmp_path / "non_existent.yaml"))
    assert config == {}

def test_load_config_merging(mock_configs_dir):
    """Test load_config with multiple default paths and a main config."""
    # The fixture already creates configs/defaults/run_settings.yaml, environment.yaml, ppo_params.yaml, config.yaml
    # and main config overrides agent_type
    
    # We will test a full load from the mock directory
    default_paths = [
        os.path.join(mock_configs_dir, "configs", "defaults", "run_settings.yaml"),
        os.path.join(mock_configs_dir, "configs", "defaults", "environment.yaml"),
        os.path.join(mock_configs_dir, "configs", "defaults", "ppo_params.yaml"),
        os.path.join(mock_configs_dir, "configs", "defaults", "binance_settings.yaml"),
        os.path.join(mock_configs_dir, "configs", "defaults", "hash_keys.yaml"),
    ]
    
    effective_config = load_config(main_config_path=os.path.join(mock_configs_dir, "config.yaml"),
                                   default_config_paths=default_paths)
    
    assert effective_config["agent_type"] == "PPO"
    assert effective_config["run_settings"]["model_name"] == "test_agent"
    assert effective_config["environment"]["kline_window_size"] == 20
    assert effective_config["ppo_params"]["learning_rate"] == 0.001
    assert effective_config["binance_settings"]["testnet"] == True


# --- Tests for data fetching and caching ---
@patch('src.data.utils.Client')
@patch('src.data.utils.os.makedirs')
@patch('src.data.utils.pd.read_parquet')
@patch('src.data.utils.pd.DataFrame.to_parquet')
@patch('src.data.utils.os.path.exists', return_value=False) # Ensure no cache exists initially
@patch('src.data.utils._calculate_technical_indicators', side_effect=lambda df, features: df) # Mock TA calculation
def test_fetch_and_cache_kline_data_fetch_success(
    mock_to_parquet, mock_read_parquet, mock_makedirs, mock_client_class, mock_path_exists, mock_kline_df,
    mock_cache_dir
):
    """Test successful fetching and caching of kline data."""
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    mock_client_instance.get_historical_klines.return_value = [
        [1672531200000, "100.0", "101.0", "99.0", "100.5", "100", 1672534799999, "0", 0, "0", "0", "0"],
        [1672534800000, "100.5", "102.0", "100.0", "101.5", "150", 1672538399999, "0", 0, "0", "0", "0"]
    ]

    df = fetch_and_cache_kline_data(
        symbol='BTCUSDT', interval='1h', start_date_str='2023-01-01', end_date_str='2023-01-01',
        cache_dir=mock_cache_dir, price_features_to_add=[], api_key='test', api_secret='test'
    )
    
    assert not df.empty
    mock_client_instance.get_historical_klines.assert_called_once()
    mock_to_parquet.assert_called_once()

@patch('src.data.utils.Client')
@patch('src.data.utils.os.makedirs')
@patch('src.data.utils.pd.read_parquet')
@patch('src.data.utils.pd.DataFrame.to_parquet')
@patch('src.data.utils.os.path.exists', return_value=False) # Ensure no cache exists initially
def test_fetch_continuous_aggregate_trades_success(
    mock_to_parquet, mock_read_parquet, mock_makedirs, mock_client_class, mock_path_exists,
    mock_cache_dir
):
    """Test successful fetching and caching of aggregate trades data."""
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    mock_client_instance.get_aggregate_trades.side_effect = [
        [{'T': 1672531200000, 'p': '100.0', 'q': '1.0', 'm': True, 'a': 1}],
        [{'T': 1672531200001, 'p': '100.1', 'q': '1.5', 'm': False, 'a': 2}],
        [] # Simulate end of data
    ]

    df = fetch_continuous_aggregate_trades(
        symbol='BTCUSDT', start_date_str='2023-01-01 00:00:00', end_date_str='2023-01-01 00:00:02',
        cache_dir=mock_cache_dir, api_key='test', api_secret='test'
    )

    assert not df.empty
    assert len(df) == 2 # Two trades fetched
    mock_client_instance.get_aggregate_trades.assert_called()
    mock_to_parquet.assert_called_once()

def test_get_data_path_for_day():
    """Test correct file path generation."""
    path_agg = get_data_path_for_day('2023-01-01', 'BTCUSDT', 'agg_trades', cache_dir='test_cache')
    assert path_agg == os.path.join('test_cache', 'bn_aggtrades_BTCUSDT_2023-01-01.parquet')

    path_kline = get_data_path_for_day('2023-01-01', 'BTCUSDT', 'kline', '1h', ['Close', 'RSI_14'], cache_dir='test_cache')
    # Features should be sorted and normalized
    assert path_kline == os.path.join('test_cache', 'bn_klines_BTCUSDT_1h_2023-01-01_close_rsi14.parquet')


@patch('src.data.utils.fetch_continuous_aggregate_trades')
@patch('src.data.utils.os.path.exists')
@patch('src.data.utils.pd.read_parquet')
def test_load_tick_data_for_range_missing_day(mock_read_parquet, mock_path_exists, mock_fetch_trades, mock_cache_dir):
    """Test loading tick data, including fetching a missing day."""
    # Simulate first day exists, second day missing, third day exists
    mock_path_exists.side_effect = lambda x: '2023-01-01' in x or '2023-01-03' in x

    # Mock data for existing files and fetched data
    df_day1 = pd.DataFrame([100], index=pd.to_datetime(['2023-01-01 12:00:00'], tz='UTC'), columns=['Price'])
    df_day2_fetched = pd.DataFrame([101], index=pd.to_datetime(['2023-01-02 12:00:00'], tz='UTC'), columns=['Price'])
    df_day3 = pd.DataFrame([102], index=pd.to_datetime(['2023-01-03 12:00:00'], tz='UTC'), columns=['Price'])

    mock_read_parquet.side_effect = [df_day1, df_day3] # For existing files
    mock_fetch_trades.return_value = df_day2_fetched # For missing day

    df = load_tick_data_for_range('BTCUSDT', '2023-01-01', '2023-01-03', cache_dir=mock_cache_dir)

    assert len(df) == 3
    assert df.index.min().date() == datetime(2023, 1, 1).date()
    assert df.index.max().date() == datetime(2023, 1, 3).date()
    mock_fetch_trades.assert_called_once() # Only called for the missing day


# --- Tests for config utilities ---
def test_merge_configs():
    """Test recursive merging of dictionaries."""
    default = {'a': 1, 'b': {'x': 10, 'y': 20}}
    override = {'b': {'y': 25, 'z': 30}, 'c': 3}
    merged = merge_configs(default, override)
    assert merged == {'a': 1, 'b': {'x': 10, 'y': 25, 'z': 30}, 'c': 3}

def test_generate_config_hash():
    """Test hash generation for consistency."""
    config1 = {'param1': 1, 'param2': 'abc'}
    config2 = {'param2': 'abc', 'param1': 1} # Same content, different order
    config3 = {'param1': 1, 'param2': 'xyz'}

    hash1 = generate_config_hash(config1)
    hash2 = generate_config_hash(config2)
    hash3 = generate_config_hash(config3)

    assert hash1 == hash2 # Order-independent hashing
    assert hash1 != hash3 # Different content, different hash

def test_convert_to_native_types():
    """Test conversion of numpy types to native Python types."""
    np_int = np.int64(1)
    np_float = np.float32(1.5)
    np_bool = np.bool_(True)
    np_array = np.array([1, 2, 3])
    pd_ts = pd.Timestamp('2023-01-01', tz='UTC')

    data = {
        'int': np_int,
        'float': np_float,
        'bool': np_bool,
        'list': [np_int, np_float],
        'array': np_array,
        'timestamp': pd_ts
    }
    converted_data = convert_to_native_types(data)

    assert isinstance(converted_data['int'], int)
    assert isinstance(converted_data['float'], float)
    assert isinstance(converted_data['bool'], bool)
    assert isinstance(converted_data['list'][0], int)
    assert isinstance(converted_data['list'][1], float)
    assert isinstance(converted_data['array'], list)
    assert isinstance(converted_data['timestamp'], str)


def test_get_relevant_config_for_hash(mock_configs_dir):
    """Test extraction of relevant config for hashing based on agent_type."""
    # Load effective config as train_agent would
    default_paths = [
        os.path.join(mock_configs_dir, "configs", "defaults", "run_settings.yaml"),
        os.path.join(mock_configs_dir, "configs", "defaults", "environment.yaml"),
        os.path.join(mock_configs_dir, "configs", "defaults", "ppo_params.yaml"),
        os.path.join(mock_configs_dir, "configs", "defaults", "binance_settings.yaml"),
        os.path.join(mock_configs_dir, "configs", "defaults", "hash_keys.yaml"),
    ]
    effective_config_ppo = load_config(main_config_path=os.path.join(mock_configs_dir, "config.yaml"),
                                       default_config_paths=default_paths)
    
    # Simulate config.yaml setting SAC
    shutil.copyfile(os.path.join(mock_configs_dir, "configs", "defaults", "sac_params.yaml"), os.path.join(mock_configs_dir, "sac_params.yaml"))
    with open(os.path.join(mock_configs_dir, "config.yaml"), 'w') as f:
        f.write("agent_type: 'SAC'\nsac_params:\n  learning_rate: 0.0005\n")
    
    effective_config_sac = load_config(main_config_path=os.path.join(mock_configs_dir, "config.yaml"),
                                       default_config_paths=default_paths + [os.path.join(mock_configs_dir, "sac_params.yaml")])


    # Test for PPO agent_type
    relevant_ppo = get_relevant_config_for_hash(effective_config_ppo)
    assert 'agent_params' in relevant_ppo
    assert 'PPO' in relevant_ppo['agent_params']
    assert relevant_ppo['agent_params']['PPO']['learning_rate'] == 0.001
    assert 'total_timesteps' in relevant_ppo['agent_params']['PPO'] # Should be in default PPO params
    assert 'kline_window_size' in relevant_ppo['environment'] # From env hash keys

    # Test for SAC agent_type
    relevant_sac = get_relevant_config_for_hash(effective_config_sac)
    assert 'agent_params' in relevant_sac
    assert 'SAC' in relevant_sac['agent_params']
    assert relevant_sac['agent_params']['SAC']['learning_rate'] == 0.0005 # Should be overridden
    assert 'buffer_size' in relevant_sac['agent_params']['SAC'] # From SAC default params


@patch('src.data.utils.os.path.exists')
@patch('src.data.utils.get_relevant_config_for_hash')
@patch('src.data.utils.generate_config_hash', return_value='test_hash')
def test_resolve_model_path_explicit_exists(
    mock_gen_hash, mock_get_relevant_config, mock_path_exists
):
    """Test resolving model path when an explicit path is provided and exists."""
    mock_path_exists.side_effect = lambda x: x == "path/to/model.zip" # Only mock specific file existence
    
    effective_config = {
        "run_settings": {
            "model_path": "path/to/model.zip",
            "alt_model_path": "path/to/alt_model.zip"
        }
    }
    
    model_path, alt_path = resolve_model_path(effective_config)
    assert model_path == "path/to/model.zip"
    assert alt_path == "path/to/alt_model.zip"
    mock_get_relevant_config.assert_not_called() # Should not call hash generation

@patch('src.data.utils.os.path.exists')
@patch('src.data.utils.get_relevant_config_for_hash', return_value={'env': 'test_env'})
@patch('src.data.utils.generate_config_hash', return_value='test_hash')
def test_resolve_model_path_reconstruct_best_model(
    mock_gen_hash, mock_get_relevant_config, mock_path_exists
):
    """Test resolving model path by reconstruction, finding best_model."""
    # Mock for reconstructed path exists, explicit path does not
    mock_path_exists.side_effect = lambda x: x == os.path.join("logs", "training", "test_hash_my_agent", "best_model", "best_model.zip")
    
    effective_config = {
        "run_settings": {
            "model_path": "non_existent_path.zip",
            "model_name": "my_agent",
            "log_dir_base": "logs/", # This is internal, but was previously used for path structure logic
        },
        "hash_config_keys": {} # Minimal for get_relevant_config to return value
    }
    
    model_path, alt_path = resolve_model_path(effective_config)
    assert model_path == os.path.join("logs", "training", "test_hash_my_agent", "best_model", "best_model.zip")
    assert alt_path is None # Alt path not found
    mock_get_relevant_config.assert_called_once()
    mock_gen_hash.assert_called_once()