# tests/agents/test_train_agent.py
import pytest
import os
import shutil
import json
from unittest.mock import patch, MagicMock
import pandas as pd # <-- ADDED THIS IMPORT
import numpy as np # <-- ADDED THIS IMPORT, as numpy is also used

# Import the train_agent function from the new path
from src.agents.train_agent import train_agent, load_default_configs_for_training

# --- Fixtures ---
@pytest.fixture
def mock_config_dir(tmp_path):
    """Creates a temporary config directory with default configs."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    # Minimal default configs for testing train_agent setup
    (defaults_dir / "run_settings.yaml").write_text("run_settings:\n  log_dir_base: 'logs/'\n  model_name: 'test_agent'\n  log_level: 'none'\n  eval_freq_episodes: 1\n  n_evaluation_episodes: 1\n")
    # MODIFIED: Added tick_resample_interval_ms to environment.yaml mock
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 1\n  tick_feature_window_size: 1\n  kline_price_features: ['Close']\n  tick_features_to_use: ['Price']\n  tick_resample_interval_ms: 60000\n")
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.001\n  total_timesteps: 100\n  policy_kwargs: \"{'net_arch': [32]}\"\n")
    (defaults_dir / "sac_params.yaml").write_text("sac_params:\n  learning_rate: 0.0005\n  total_timesteps: 50\n")
    # Corrected binance_settings.yaml content with full datetime strings
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings:\n  default_symbol: 'BTCUSDT'\n  historical_interval: '1h'\n  historical_cache_dir: 'data_cache/'\n  start_date_kline_data: '2024-01-01 00:00:00'\n  end_date_kline_data: '2024-01-01 23:59:59'\n  start_date_tick_data: '2024-01-01 00:00:00'\n  end_date_tick_data: '2024-01-01 23:59:59'\n")
    # Corrected evaluation_data.yaml content with full datetime strings
    (defaults_dir / "evaluation_data.yaml").write_text("evaluation_data:\n  start_date_eval: '2024-01-02 00:00:00'\n  end_date_eval: '2024-01-02 23:59:59'\n")
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size', 'tick_resample_interval_ms']\n  agent_params:\n    PPO: ['learning_rate']\n    SAC: ['learning_rate']\n  binance_settings: ['default_symbol']\n")

    # Create dummy main config.yaml in the root tmp_path
    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n")
    
    return str(tmp_path)

@pytest.fixture
def mock_data_loader():
    """Mocks data loading functions."""
    with patch('src.data.utils.load_kline_data_for_range') as mock_load_kline, \
         patch('src.data.utils.load_tick_data_for_range') as mock_load_tick:
        
        # Mock minimal DataFrame for env initialization
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
        yield # Allows test to run, then unpatches

@pytest.fixture
def mock_sb3_models():
    """Mocks Stable Baselines3 model classes to prevent actual training."""
    with patch('stable_baselines3.PPO') as mock_ppo, \
         patch('stable_baselines3.SAC') as mock_sac, \
         patch('stable_baselines3.DDPG') as mock_ddpg, \
         patch('stable_baselines3.A2C') as mock_a2c, \
         patch('src.agents.train_agent.SB3_CONTRIB_AVAILABLE', True), \
         patch('sb3_contrib.RecurrentPPO') as mock_recurrent_ppo:
        
        mock_ppo.return_value = MagicMock()
        mock_sac.return_value = MagicMock()
        mock_ddpg.return_value = MagicMock()
        mock_a2c.return_value = MagicMock()
        mock_recurrent_ppo.return_value = MagicMock()
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
    # Use tmp_path as the base for the project root for testing
    project_root = tmp_path

    # Simulate presence of logs and data_cache in the mock root
    logs_dir = project_root / "logs"
    logs_dir.mkdir()
    (logs_dir / "training").mkdir()
    (logs_dir / "tensorboard_logs").mkdir()

    data_cache_dir = project_root / "data_cache"
    data_cache_dir.mkdir()

    # Monkeypatch os.getcwd to return tmp_path for the duration of the test
    monkeypatch.chdir(project_root)
    
    # Ensure DATA_CACHE_DIR in utils points to the mock cache
    import src.data.utils
    monkeypatch.setattr(src.data.utils, 'DATA_CACHE_DIR', str(data_cache_dir))

    yield

    # Teardown: not strictly needed with tmp_path, but good practice if not using it
    # shutil.rmtree(logs_dir)
    # shutil.rmtree(data_cache_dir)


# --- Tests for train_agent ---

def test_train_agent_ppo_setup(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that train_agent sets up PPO correctly."""
    # Simulate config.yaml for PPO (already done in fixture)
    
    final_metric = train_agent(log_to_file=False) # Run with log_to_file=False for cleaner test output
    
    assert final_metric == -np.inf # Expected when training is mocked and no eval reward returned

    mock_sb3_models["PPO"].assert_called_once()
    args, kwargs = mock_sb3_models["PPO"].call_args
    assert kwargs['learning_rate'] == 0.001
    assert kwargs['total_timesteps'] == 100 # Should be passed to .learn()
    mock_sb3_models["PPO"].return_value.learn.assert_called_once_with(
        total_timesteps=100,
        callback=mock_sb3_models["PPO"].return_value.learn.call_args.kwargs['callback'],
        progress_bar=True,
        tb_log_name=None
    )


def test_train_agent_sac_setup(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that train_agent sets up SAC correctly with an override."""
    # Override config.yaml for SAC
    with open(os.path.join(mock_config_dir, "config.yaml"), 'w') as f:
        f.write("agent_type: 'SAC'\n")
    
    # Ensure SAC params default file is copied to temp_path for load_default_configs_for_training
    sac_params_path = os.path.join(mock_config_dir, "configs", "defaults", "sac_params.yaml")
    
    # Manually ensure the default config loading paths are correct for the test fixture
    # This might require adjusting how load_default_configs_for_training finds its paths
    # or ensuring mock_config_dir includes SAC defaults too.
    # Simpler: pass override directly for agent_type and params
    override_config = {
        "agent_type": "SAC",
        "sac_params": {"learning_rate": 0.0005, "total_timesteps": 50}
    }

    final_metric = train_agent(config_override=override_config, log_to_file=False)
    
    assert final_metric == -np.inf

    mock_sb3_models["SAC"].assert_called_once()
    args, kwargs = mock_sb3_models["SAC"].call_args
    assert kwargs['learning_rate'] == 0.0005
    assert kwargs['total_timesteps'] == 50
    mock_sb3_models["SAC"].return_value.learn.assert_called_once_with(
        total_timesteps=50,
        callback=mock_sb3_models["SAC"].return_value.learn.call_args.kwargs['callback'],
        progress_bar=True,
        tb_log_name=None
    )


def test_train_agent_eval_callback_creation(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that EvalCallback is set up when eval data is present."""
    # Mock return values for load_kline_data_for_range and load_tick_data_for_range for eval data
    # to be non-empty
    with patch('src.data.utils.load_kline_data_for_range') as mock_load_kline, \
         patch('src.data.utils.load_tick_data_for_range') as mock_load_tick:
        
        mock_kline_df = pd.DataFrame(
            {'Open': [100.0], 'High': [101.0], 'Low': [99.0], 'Close': [100.5], 'Volume': [10.0]},
            index=pd.to_datetime(['2024-01-02 00:00:00'], utc=True)
        )
        mock_tick_df = pd.DataFrame(
            {'Price': [100.0], 'Quantity': [1.0]},
            index=pd.to_datetime(['2024-01-02 00:00:00'], utc=True)
        )
        mock_load_kline.side_effect = [
            mock_kline_df, # for training data
            mock_kline_df # for evaluation data
        ]
        mock_load_tick.side_effect = [
            mock_tick_df, # for training data
            mock_tick_df # for evaluation data
        ]

        # Mock EvalCallback constructor
        with patch('stable_baselines3.common.callbacks.EvalCallback', autospec=True) as MockEvalCallback:
            MockEvalCallback.return_value = MagicMock(best_mean_reward=100.0) # Mock its return value
            
            final_metric = train_agent(log_to_file=False)
            
            MockEvalCallback.assert_called_once()
            # Assert eval_freq is roughly correct
            args, kwargs = MockEvalCallback.call_args
            assert kwargs['eval_freq'] > 0
            assert final_metric == 100.0 # Should return the mocked best_mean_reward

def test_train_agent_no_eval_callback_if_no_data(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that EvalCallback is skipped if eval data is empty."""
    # Mock return values for load_kline_data_for_range and load_tick_data_for_range for eval data
    # to be empty
    with patch('src.data.utils.load_kline_data_for_range') as mock_load_kline, \
         patch('src.data.utils.load_tick_data_for_range') as mock_load_tick:
        
        # Training data is still valid
        mock_kline_df_train = pd.DataFrame(
            {'Open': [100.0], 'High': [101.0], 'Low': [99.0], 'Close': [100.5], 'Volume': [10.0]},
            index=pd.to_datetime(['2024-01-01 00:00:00'], utc=True)
        )
        mock_tick_df_train = pd.DataFrame(
            {'Price': [100.0], 'Quantity': [1.0]},
            index=pd.to_datetime(['2024-01-01 00:00:00'], utc=True)
        )
        mock_load_kline.side_effect = [
            mock_kline_df_train, # for training data
            pd.DataFrame() # for evaluation data (empty)
        ]
        mock_load_tick.side_effect = [
            mock_tick_df_train, # for training data
            pd.DataFrame() # for evaluation data (empty)
        ]
        
        with patch('stable_baselines3.common.callbacks.EvalCallback') as MockEvalCallback:
            final_metric = train_agent(log_to_file=False)
            MockEvalCallback.assert_not_called() # Should not be called
            assert final_metric == -np.inf # No best mean reward to return