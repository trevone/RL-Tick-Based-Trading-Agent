# tests/agents/test_train_agent.py
import pytest
import os
import shutil
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import the train_agent function and other necessary components
from src.agents.train_agent import train_agent
from src.data import path_manager

# --- Fixtures ---
@pytest.fixture
def mock_config_dir(tmp_path):
    """Creates a temporary config directory with valid default configs."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    # FIX: Use multiline strings to ensure valid YAML formatting.
    (defaults_dir / "run_settings.yaml").write_text(f"""
run_settings:
  log_dir_base: '{str(tmp_path / 'test_logs')}/'
  model_name: 'test_agent'
  log_level: 'normal'
  eval_freq_episodes: 1
  n_evaluation_episodes: 1
  default_symbol: 'BTCUSDT'
  historical_interval: '1h'
  historical_cache_dir: '{str(tmp_path / 'test_data_cache')}/'
  start_date_train: '2024-01-01 00:00:00'
  end_date_train: '2024-01-01 23:59:59'
  start_date_eval: '2024-01-02 00:00:00'
  end_date_eval: '2024-01-02 23:59:59'
""")
    (defaults_dir / "environment.yaml").write_text("""
environment:
  kline_window_size: 1
  tick_feature_window_size: 1
  kline_price_features: ['Close']
  tick_features_to_use: ['Price']
  tick_resample_interval_ms: 1000
""")
    (defaults_dir / "ppo_params.yaml").write_text("""
ppo_params:
  learning_rate: 0.001
  total_timesteps: 100
  policy_kwargs: "{'net_arch': [32]}"
""")
    (defaults_dir / "sac_params.yaml").write_text("""
sac_params:
  learning_rate: 0.0005
  total_timesteps: 50
  buffer_size: 10000
""")
    (defaults_dir / "binance_settings.yaml").write_text("""
binance_settings:
  testnet: true
""")
    (defaults_dir / "evaluation_data.yaml").write_text("# This file is empty\n")
    (defaults_dir / "hash_keys.yaml").write_text("""
hash_config_keys:
  environment: ['kline_window_size', 'tick_resample_interval_ms']
  agent_params:
    PPO: ['learning_rate']
    SAC: ['learning_rate', 'buffer_size']
  run_settings: ['default_symbol']
""")

    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n")
    
    return str(tmp_path)

@pytest.fixture
def mock_data_loader():
    """Mocks data loading functions and yields the mock objects."""
    with patch('src.agents.train_agent.load_kline_data_for_range') as mock_load_kline, \
         patch('src.agents.train_agent.load_tick_data_for_range') as mock_load_tick:
        
        mock_kline_df = pd.DataFrame(
            {'Open': [100.0]*2, 'High': [101.0]*2, 'Low': [99.0]*2, 'Close': [100.5]*2, 'Volume': [10.0]*2},
            index=pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 01:00:00'], utc=True)
        )
        mock_tick_df = pd.DataFrame(
            {'Price': [100.0]*2, 'Quantity': [1.0]*2, 'IsBuyerMaker': [False]*2},
            index=pd.to_datetime(['2024-01-01 00:00:00.000', '2024-01-01 00:00:00.001'], utc=True)
        )
        mock_load_kline.return_value = mock_kline_df
        mock_load_tick.return_value = mock_tick_df
        yield mock_load_kline, mock_load_tick

@pytest.fixture
def mock_sb3_models():
    """Mocks Stable Baselines3 model classes to prevent actual training."""
    with patch('src.agents.train_agent.PPO') as mock_ppo, \
         patch('src.agents.train_agent.SAC') as mock_sac, \
         patch('src.agents.train_agent.DDPG') as mock_ddpg, \
         patch('src.agents.train_agent.A2C') as mock_a2c, \
         patch('src.agents.train_agent.SB3_CONTRIB_AVAILABLE', True), \
         patch('src.agents.train_agent.RecurrentPPO', create=True) as mock_recurrent_ppo:
        
        model_instance_mock = MagicMock()
        mock_ppo.return_value = model_instance_mock
        mock_sac.return_value = model_instance_mock
        mock_ddpg.return_value = model_instance_mock
        mock_a2c.return_value = model_instance_mock
        mock_recurrent_ppo.return_value = model_instance_mock
        yield {
            "PPO": mock_ppo, "SAC": mock_sac, "DDPG": mock_ddpg,
            "A2C": mock_a2c, "RecurrentPPO": mock_recurrent_ppo
        }

@pytest.fixture(autouse=True)
def setup_teardown_dirs(tmp_path, monkeypatch):
    """Sets up temporary log/data_cache directories and cleans up."""
    project_root = tmp_path
    logs_dir_for_runs = project_root / "test_logs" / "training"
    logs_dir_for_runs.mkdir(parents=True, exist_ok=True)
    tensorboard_logs_base = project_root / "logs" / "tensorboard_logs"
    tensorboard_logs_base.mkdir(parents=True, exist_ok=True)
    default_data_cache_dir = project_root / "data_cache"
    default_data_cache_dir.mkdir(parents=True, exist_ok=True)
    test_data_cache_from_config = project_root / "test_data_cache"
    test_data_cache_from_config.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(path_manager, 'DATA_CACHE_DIR', str(default_data_cache_dir))
    yield


# --- Tests for train_agent ---

def test_train_agent_ppo_setup(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that train_agent sets up PPO correctly."""
    final_metric = train_agent(log_to_file=False)
    
    assert final_metric == -np.inf
    mock_sb3_models["PPO"].assert_called_once()
    args, kwargs = mock_sb3_models["PPO"].call_args
    assert kwargs['learning_rate'] == 0.001 
    mock_sb3_models["PPO"].return_value.learn.assert_called_once()
    learn_args, learn_kwargs = mock_sb3_models["PPO"].return_value.learn.call_args
    assert learn_kwargs['total_timesteps'] == 100

def test_train_agent_sac_setup(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that train_agent sets up SAC correctly with an override."""
    with open(os.path.join(mock_config_dir, "config.yaml"), 'w') as f:
        f.write("agent_type: 'SAC'\n")
    
    final_metric = train_agent(log_to_file=False) 
    
    assert final_metric == -np.inf
    mock_sb3_models["SAC"].assert_called_once()
    args, kwargs = mock_sb3_models["SAC"].call_args
    assert kwargs['learning_rate'] == 0.0005 
    assert kwargs['buffer_size'] == 10000
    mock_sb3_models["SAC"].return_value.learn.assert_called_once()
    learn_args, learn_kwargs = mock_sb3_models["SAC"].return_value.learn.call_args
    assert learn_kwargs['total_timesteps'] == 50

def test_train_agent_eval_callback_creation(mock_config_dir, mock_data_loader, mock_sb3_models, tmp_path):
    """Test that EvalCallback is set up and its result is returned by train_agent if log_to_file=True."""
    log_base = tmp_path / "eval_callback_test_logs"
    config_override = {"run_settings": {"log_dir_base": str(log_base)}}

    with patch('src.agents.train_agent.EvalCallback', autospec=True) as MockEvalCallback:
        mock_eval_callback_instance = MagicMock()
        mock_eval_callback_instance.best_mean_reward = 123.45
        MockEvalCallback.return_value = mock_eval_callback_instance
            
        final_metric = train_agent(config_override=config_override, log_to_file=True) 
            
        MockEvalCallback.assert_called_once()
        args, kwargs = MockEvalCallback.call_args
        assert kwargs['eval_freq'] > 0
        assert kwargs['log_path'] is not None
        assert kwargs['best_model_save_path'] is not None
        assert final_metric == 123.45

def test_train_agent_no_eval_callback_if_no_data(mock_config_dir, mock_sb3_models, tmp_path):
    """Test that EvalCallback is skipped if eval data is empty, even if log_to_file=True."""
    log_base = tmp_path / "no_eval_data_test_logs"
    config_override = {"run_settings": {"log_dir_base": str(log_base)}}
    
    with patch('src.agents.train_agent.load_kline_data_for_range') as mock_load_kline, \
         patch('src.agents.train_agent.load_tick_data_for_range') as mock_load_tick:
        
        mock_kline_df_train = pd.DataFrame(
            {'Open': [100.0]*2, 'High': [101.0]*2, 'Low': [99.0]*2, 'Close': [100.5]*2, 'Volume': [10.0]*2},
            index=pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 01:00:00'], utc=True)
        )
        mock_tick_df_train = pd.DataFrame(
            {'Price': [100.0]*2, 'Quantity': [1.0]*2, 'IsBuyerMaker': [False]*2},
            index=pd.to_datetime(['2024-01-01 00:00:00.000', '2024-01-01 00:00:00.001'], utc=True)
        )
        
        mock_load_kline.side_effect = [mock_kline_df_train, pd.DataFrame()]
        mock_load_tick.side_effect = [mock_tick_df_train, pd.DataFrame()]
        
        with patch('src.agents.train_agent.EvalCallback') as MockEvalCallbackConstructor:
            final_metric = train_agent(config_override=config_override, log_to_file=True)
            MockEvalCallbackConstructor.assert_not_called()
            assert final_metric == -np.inf

def test_train_agent_uses_correct_cache_dir(mock_config_dir, mock_data_loader, mock_sb3_models, tmp_path):
    """
    Tests that train_agent passes the 'historical_cache_dir' from config
    to the data loading functions.
    """
    mock_load_kline, mock_load_tick = mock_data_loader
    custom_cache_dir = str(tmp_path / "my_unique_test_cache")
    config_override = {
        "run_settings": {
            "historical_cache_dir": custom_cache_dir,
            "log_level": "none"
        },
        "ppo_params": {"total_timesteps": 10},
    }
    
    train_agent(config_override=config_override, log_to_file=False)

    assert mock_load_kline.call_count > 0
    for k_call_args in mock_load_kline.call_args_list:
        _, kwargs = k_call_args
        assert kwargs.get("cache_dir") == custom_cache_dir

    assert mock_load_tick.call_count > 0
    for t_call_args in mock_load_tick.call_args_list:
        _, kwargs = t_call_args
        assert kwargs.get("cache_dir") == custom_cache_dir

@pytest.mark.parametrize("log_level_setting", ["detailed", "normal", "none"])
def test_train_agent_propagates_log_level(mock_config_dir, mock_data_loader, mock_sb3_models, log_level_setting, tmp_path):
    """
    Tests that train_agent passes the 'log_level' from config
    to the data loading functions when log_to_file=True.
    If log_to_file=False, it should pass "none".
    """
    mock_load_kline, mock_load_tick = mock_data_loader
    test_specific_log_dir_base = str(tmp_path / f"logs_prop_{log_level_setting}")
    config_override = {
        "run_settings": {
            "log_level": log_level_setting,
            "log_dir_base": test_specific_log_dir_base
        },
        "ppo_params": {"total_timesteps": 10}
    }

    # Scenario 1: log_to_file = True
    train_agent(config_override=config_override, log_to_file=True)
    assert mock_load_kline.call_count >= 1
    for k_call_args in mock_load_kline.call_args_list:
        _, kwargs = k_call_args
        assert kwargs.get("log_level") == log_level_setting
    assert mock_load_tick.call_count >= 1
    for t_call_args in mock_load_tick.call_args_list:
        _, kwargs = t_call_args
        assert kwargs.get("log_level") == log_level_setting

    # Scenario 2: log_to_file = False
    mock_load_kline.reset_mock()
    mock_load_tick.reset_mock()
    train_agent(config_override=config_override, log_to_file=False)
    expected_propagated_log_level = "none"
    assert mock_load_kline.call_count >= 1
    for k_call_args in mock_load_kline.call_args_list:
        _, kwargs = k_call_args
        assert kwargs.get("log_level") == expected_propagated_log_level
    assert mock_load_tick.call_count >= 1
    for t_call_args in mock_load_tick.call_args_list:
        _, kwargs = t_call_args
        assert kwargs.get("log_level") == expected_propagated_log_level