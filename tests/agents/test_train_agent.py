import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the function to be tested
from src.agents.train_agent import train_agent

@pytest.fixture
def mock_data_loaders():
    """Mocks the data loading functions to prevent file access and network calls."""
    with patch('src.agents.train_agent.load_kline_data_for_range') as mock_load_kline, \
         patch('src.agents.train_agent.load_tick_data_for_range') as mock_load_tick:
        
        # Provide minimal, valid-looking DataFrames
        mock_kline_df = pd.DataFrame({
            'Open': [100.0] * 10, 'High': [101.0] * 10, 'Low': [99.0] * 10, 
            'Close': [100.5] * 10, 'Volume': [10.0] * 10
        }, index=pd.to_datetime(pd.date_range(start="2024-01-01", periods=10, freq='h', tz='UTC')))
        
        mock_tick_df = pd.DataFrame({
            'Price': [100.0] * 60, 'Quantity': [1.0] * 60
        }, index=pd.to_datetime(pd.date_range(start="2024-01-01", periods=60, freq='s', tz='UTC')))

        mock_load_kline.return_value = mock_kline_df
        mock_load_tick.return_value = mock_tick_df
        yield mock_load_kline, mock_load_tick

def test_train_agent_runs_without_error(mock_data_loaders):
    """
    A simple integration test to ensure train_agent can initialize and run 
    for a minimal number of timesteps without crashing.
    """
    # Override config for a very short test run
    test_config_override = {
        "run_settings": {
            "start_date_train": "2024-01-01 00:00:00",
            "end_date_train": "2024-01-01 23:59:59",
            "start_date_eval": "2024-01-01 00:00:00",
            "end_date_eval": "2024-01-01 23:59:59",
        },
        "ppo_params": {
            "total_timesteps": 100, # Very few steps
            "n_steps": 10
        },
        "environment": {
            "kline_window_size": 5,
            "tick_feature_window_size": 50,
        }
    }

    try:
        # We expect this to return a float (the reward), but the main goal
        # is to ensure no exceptions are raised.
        result = train_agent(config_override=test_config_override, log_to_file=False)
        assert isinstance(result, float), "train_agent should return a float metric."

    except Exception as e:
        pytest.fail(f"train_agent failed with an unexpected exception: {e}")