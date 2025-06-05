# tests/agents/test_train_agent.py
import pytest
import os
import shutil
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import the train_agent function from the new path
from src.agents.train_agent import train_agent, load_default_configs_for_training
from src.environments.base_env import DEFAULT_ENV_CONFIG # For checking default rewards

# --- Fixtures ---
@pytest.fixture
def mock_config_dir(tmp_path):
    """Creates a temporary config directory with default configs."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    (defaults_dir / "run_settings.yaml").write_text("run_settings:\n  log_dir_base: 'logs/'\n  model_name: 'test_agent'\n  log_level: 'none'\n  eval_freq_episodes: 1\n  n_evaluation_episodes: 1\n")
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 1\n  tick_feature_window_size: 1\n  kline_price_features: ['Close']\n  tick_features_to_use: ['Price']\n")
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.001\n  total_timesteps: 100\n  policy_kwargs: \"{'net_arch': [32]}\"\n")
    (defaults_dir / "sac_params.yaml").write_text("sac_params:\n  learning_rate: 0.0005\n  total_timesteps: 50\n  buffer_size: 10000\n")
    (defaults_dir / "binance_settings.yaml").write_text(
        "binance_settings:\n"
        "  default_symbol: 'BTCUSDT'\n"
        "  historical_interval: '1h'\n"
        "  historical_cache_dir: 'data_cache/'\n"
        "  start_date_kline_data: '2024-01-01 00:00:00'\n"
        "  end_date_kline_data: '2024-01-01 23:59:59'\n"
        "  start_date_tick_data: '2024-01-01 00:00:00'\n"
        "  end_date_tick_data: '2024-01-01 23:59:59'\n"
    )
    (defaults_dir / "evaluation_data.yaml").write_text(
        "evaluation_data:\n"
        "  start_date_eval: '2024-01-02 00:00:00'\n"
        "  end_date_eval: '2024-01-02 23:59:59'\n"
        "  start_date_kline_eval: '2024-01-02 00:00:00'\n"
        "  end_date_kline_eval: '2024-01-02 23:59:59'\n"
        "  start_date_tick_eval: '2024-01-02 00:00:00'\n"
        "  end_date_tick_eval: '2024-01-02 23:59:59'\n"
    )
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size']\n  agent_params:\n    PPO: ['learning_rate']\n    SAC: ['learning_rate', 'buffer_size']\n  binance_settings: ['default_symbol']\n")

    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n")
    
    return str(tmp_path)

@pytest.fixture
def mock_data_loader():
    """Mocks data loading functions."""
    with patch('src.agents.train_agent.load_kline_data_for_range') as mock_load_kline, \
         patch('src.agents.train_agent.load_tick_data_for_range') as mock_load_tick:
        
        mock_kline_df = pd.DataFrame(
            {'Open': [100.0], 'High': [101.0], 'Low': [99.0], 'Close': [100.5], 'Volume': [10.0]},
            index=pd.to_datetime(['2024-01-01 00:00:00'], utc=True)
        )
        mock_tick_df = pd.DataFrame(
            {'Price': [100.0], 'Quantity': [1.0]},
            index=pd.to_datetime(['2024-01-01 00:00:00'], utc=True)
        )

        mock_load_kline.return_value = mock_kline_df
        mock_load_tick.return_value = mock_tick_df
        
        yield

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
            "PPO": mock_ppo,
            "SAC": mock_sac,
            "DDPG": mock_ddpg,
            "A2C": mock_a2c,
            "RecurrentPPO": mock_recurrent_ppo
        }

@pytest.fixture(autouse=True)
def setup_teardown_dirs(tmp_path, monkeypatch):
    """Sets up temporary log/data_cache directories and cleans up."""
    project_root = tmp_path

    logs_dir = project_root / "logs"
    logs_dir.mkdir()
    (logs_dir / "training").mkdir()
    (logs_dir / "tensorboard_logs").mkdir()

    data_cache_dir = project_root / "data_cache"
    data_cache_dir.mkdir()

    monkeypatch.chdir(project_root)
    
    import src.data.utils
    monkeypatch.setattr(src.data.utils, 'DATA_CACHE_DIR', str(data_cache_dir))

    yield

# --- Tests for train_agent ---

def test_train_agent_ppo_setup(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that train_agent sets up PPO correctly."""
    final_metric = train_agent(log_to_file=False)
    
    # When model.learn() is on a MagicMock, EvalCallback won't update its best_mean_reward.
    # So, train_agent returns its default -np.inf.
    assert final_metric == -np.inf

    mock_sb3_models["PPO"].assert_called_once()
    args, kwargs = mock_sb3_models["PPO"].call_args
    assert kwargs['learning_rate'] == 0.001 
    
    mock_sb3_models["PPO"].return_value.learn.assert_called_once()
    learn_args, learn_kwargs = mock_sb3_models["PPO"].return_value.learn.call_args
    assert learn_kwargs['total_timesteps'] == 100 
    assert learn_kwargs['tb_log_name'] is None 
    assert learn_kwargs['progress_bar'] is True 
    assert learn_kwargs['callback'] is not None 
    assert len(learn_kwargs['callback']) > 0


def test_train_agent_sac_setup(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that train_agent sets up SAC correctly with an override."""
    with open(os.path.join(mock_config_dir, "config.yaml"), 'w') as f:
        f.write("agent_type: 'SAC'\n")
    
    override_config = {
        "agent_type": "SAC",
        "sac_params": {"learning_rate": 0.0005, "total_timesteps": 50, "buffer_size": 10000}
    }

    final_metric = train_agent(config_override=override_config, log_to_file=False)
    
    assert final_metric == -np.inf

    mock_sb3_models["SAC"].assert_called_once()
    args, kwargs = mock_sb3_models["SAC"].call_args
    assert kwargs['learning_rate'] == 0.0005
    assert kwargs['buffer_size'] == 10000
    
    mock_sb3_models["SAC"].return_value.learn.assert_called_once()
    learn_args, learn_kwargs = mock_sb3_models["SAC"].return_value.learn.call_args
    assert learn_kwargs['total_timesteps'] == 50
    assert learn_kwargs['tb_log_name'] is None
    assert learn_kwargs['callback'] is not None
    assert len(learn_kwargs['callback']) > 0


def test_train_agent_eval_callback_creation(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that EvalCallback is set up and its result is returned by train_agent."""
    with patch('stable_baselines3.common.callbacks.EvalCallback', autospec=True) as MockEvalCallback:
        mock_eval_callback_instance = MagicMock()
        mock_eval_callback_instance.best_mean_reward = 123.45 
        MockEvalCallback.return_value = mock_eval_callback_instance
            
        final_metric = train_agent(log_to_file=False)
            
        MockEvalCallback.assert_called_once() 
        args, kwargs = MockEvalCallback.call_args
        assert kwargs['eval_freq'] > 0 
        assert kwargs['log_path'] is None 
        assert kwargs['best_model_save_path'] is None 
        assert final_metric == 123.45 

def test_train_agent_no_eval_callback_if_no_data(mock_config_dir, mock_sb3_models):
    """Test that EvalCallback is skipped if eval data is empty."""
    with patch('src.agents.train_agent.load_kline_data_for_range') as mock_load_kline, \
         patch('src.agents.train_agent.load_tick_data_for_range') as mock_load_tick:
        
        mock_kline_df_train = pd.DataFrame(
            {'Open': [100.0], 'High': [101.0], 'Low': [99.0], 'Close': [100.5], 'Volume': [10.0]},
            index=pd.to_datetime(['2024-01-01 00:00:00'], utc=True)
        )
        mock_tick_df_train = pd.DataFrame(
            {'Price': [100.0], 'Quantity': [1.0]},
            index=pd.to_datetime(['2024-01-01 00:00:00'], utc=True)
        )
        
        kline_call_count = 0
        def side_effect_kline(*args, **kwargs):
            nonlocal kline_call_count
            kline_call_count += 1
            if kline_call_count == 1: 
                return mock_kline_df_train
            return pd.DataFrame() 

        tick_call_count = 0
        def side_effect_tick(*args, **kwargs):
            nonlocal tick_call_count
            tick_call_count += 1
            if tick_call_count == 1: 
                return mock_tick_df_train
            return pd.DataFrame() 

        mock_load_kline.side_effect = side_effect_kline
        mock_load_tick.side_effect = side_effect_tick
        
        with patch('stable_baselines3.common.callbacks.EvalCallback') as MockEvalCallbackConstructor:
            final_metric = train_agent(log_to_file=False)
            MockEvalCallbackConstructor.assert_not_called() 
            assert final_metric == -np.inf