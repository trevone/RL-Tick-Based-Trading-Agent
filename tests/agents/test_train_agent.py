# tests/agents/test_train_agent.py
import pytest
import pandas as pd
import traceback
from unittest.mock import patch, MagicMock

# Import the function to be tested
from src.agents.train_agent import train_agent

# This single, comprehensive fixture mocks ALL external dependencies.
@pytest.fixture(autouse=True)
def mock_all_dependencies():
    """A comprehensive fixture that mocks all slow I/O and computation."""
    
    mock_config = {
        "agent_type": "PPO",
        "run_settings": {
            "log_dir_base": "mock_logs/", "model_name": "test_agent", "log_level": "none",
            "eval_freq_episodes": 1, "n_evaluation_episodes": 1,
            "default_symbol": "DUMMYUSDT", "historical_interval": "1h",
            "historical_cache_dir": "mock_cache/", "start_date_train": "2024-01-01",
            "end_date_train": "2024-01-01", "start_date_eval": "2024-01-02",
            "end_date_eval": "2024-01-02", "device": "cpu",
            "continue_from_existing_model": False,
        },
        "environment": {
            "env_type": "simple", "tick_feature_window_size": 2, "kline_window_size": 2,
            "kline_price_features": [], "tick_features_to_use": ["Price", "Quantity"]
        },
        "ppo_params": {"n_steps": 5, "total_timesteps": 10}, "binance_settings": {}, "hash_config_keys": {},
    }
    
    mock_tick_df = pd.DataFrame({'Price': [100.0] * 10, 'Quantity': [1.0] * 10})
    
    patchers = {
        'load_configs': patch('src.agents.train_agent.load_default_configs_for_training', return_value=mock_config),
        'load_kline': patch('src.agents.train_agent.load_kline_data_for_range', return_value=pd.DataFrame({'Close': [100.0]*10})),
        'load_tick': patch('src.agents.train_agent.load_tick_data_for_range', return_value=mock_tick_df),
        'PPO': patch('stable_baselines3.PPO'),
        'SummaryWriter': patch('stable_baselines3.common.logger.SummaryWriter'),
        'os_makedirs': patch('os.makedirs'),
    }

    mocks = {name: p.start() for name, p in patchers.items()}
    
    model_instance_mock = MagicMock()
    mocks['PPO'].return_value = model_instance_mock
    
    yield mocks
    patch.stopall()

def test_train_agent_runs_without_error(mock_all_dependencies):
    """
    Tests that the main training logic executes without crashing.
    """
    try:
        train_agent(log_to_file=False)
        mock_all_dependencies["PPO"].assert_called_once()
        mock_all_dependencies["PPO"].return_value.learn.assert_called_once()
    except Exception as e:
        # The traceback module is now correctly imported for this call
        pytest.fail(f"train_agent raised an unexpected exception: {e}\n{traceback.format_exc()}")