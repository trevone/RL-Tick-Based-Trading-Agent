# tests/agents/test_evaluation.py
import pytest
import os
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the main function from the new path
from src.agents.evaluate_agent import main as evaluate_agent_main
from src.agents.evaluate_agent import plot_performance # Also test plot_performance directly

# --- Fixtures ---
@pytest.fixture
def mock_config_dir(tmp_path):
    """Creates a temporary config directory with default configs."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    # Minimal default configs for testing evaluate_agent
    (defaults_dir / "run_settings.yaml").write_text("run_settings:\n  log_dir_base: 'logs/'\n  model_name: 'test_agent'\n  log_level: 'none'\n  eval_log_dir: 'logs/evaluation/'\n  model_path: 'logs/training/mock_hash_test_agent/best_model/best_model.zip'\n")
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 1\n  tick_feature_window_size: 1\n  kline_price_features: ['Close']\n  tick_features_to_use: ['Price']\n  initial_balance: 10000.0\n")
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings:\n  default_symbol: 'BTCUSDT'\n  historical_interval: '1h'\n  historical_cache_dir: 'data_cache/'\n")
    (defaults_dir / "evaluation_data.yaml").write_text("evaluation_data:\n  start_date_eval: '2024-01-01'\n  end_date_eval: '2024-01-01'\n  n_evaluation_episodes: 1\n")
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size']\n  agent_params:\n    PPO: ['learning_rate']\n  binance_settings: ['default_symbol']\n")
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.0003\n") # Minimal PPO params for hashing

    # Create dummy main config.yaml in the root tmp_path
    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n") # Set agent_type
    
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
            {'Price': [100.0, 100.1, 100.2, 100.3, 100.4], 'Quantity': [1.0, 1.1, 1.2, 1.3, 1.4]},
            index=pd.to_datetime(['2024-01-01 00:00:00.000', '2024-01-01 00:00:00.001', '2024-01-01 00:00:00.002', '2024-01-01 00:00:00.003', '2024-01-01 00:00:00.004'], utc=True)
        )

        mock_load_kline.return_value = mock_kline_df
        mock_load_tick.return_value = mock_tick_df
        yield # Allows test to run, then unpatches

@pytest.fixture
def mock_sb3_ppo_load():
    """Mocks PPO.load to prevent actual model loading."""
    with patch('stable_baselines3.PPO.load') as mock_ppo_load:
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = (np.array([0, 0.01]), None) # Mock predict to return (Hold, target)
        mock_ppo_load.return_value = mock_model_instance
        yield mock_ppo_load

@pytest.fixture
def mock_sb3_vecnormalize_load():
    """Mocks VecNormalize.load to prevent actual loading of stats."""
    with patch('stable_baselines3.common.vec_env.VecNormalize.load') as mock_vecnormalize_load:
        mock_vecnormalize_instance = MagicMock()
        mock_vecnormalize_load.return_value = mock_vecnormalize_instance
        yield mock_vecnormalize_load


@pytest.fixture(autouse=True)
def setup_teardown_dirs(tmp_path, monkeypatch):
    """Sets up temporary log/data_cache directories and cleans up."""
    project_root = tmp_path

    # Simulate presence of logs and data_cache in the mock root
    logs_dir = project_root / "logs"
    logs_dir.mkdir()
    (logs_dir / "training").mkdir()
    (logs_dir / "training" / "mock_hash_test_agent").mkdir()
    (logs_dir / "training" / "mock_hash_test_agent" / "best_model").mkdir()
    (logs_dir / "training" / "mock_hash_test_agent" / "best_model" / "best_model.zip").touch() # Mock model file
    (logs_dir / "training" / "mock_hash_test_agent" / "vec_normalize.pkl").touch() # Mock vec_normalize file
    (logs_dir / "evaluation").mkdir()

    data_cache_dir = project_root / "data_cache"
    data_cache_dir.mkdir()

    # Monkeypatch os.getcwd to return tmp_path for the duration of the test
    monkeypatch.chdir(project_root)
    
    # Ensure DATA_CACHE_DIR in utils points to the mock cache
    import src.data.utils
    monkeypatch.setattr(src.data.utils, 'DATA_CACHE_DIR', str(data_cache_dir))

    yield

    # No need for explicit cleanup with tmp_path


# --- Tests for evaluate_agent_main ---
def test_evaluate_agent_main_runs_successfully(
    mock_config_dir, mock_data_loader, mock_sb3_ppo_load, mock_sb3_vecnormalize_load
):
    """Test that the main evaluation function runs without critical errors."""
    try:
        evaluate_agent_main()
        # If no unhandled exceptions, test passes.
        # Check that model.load was called and data loaders were called.
        mock_sb3_ppo_load.assert_called_once()
        mock_data_loader.__enter__.return_value[0].assert_called() # kline loader
        mock_data_loader.__enter__.return_value[1].assert_called() # tick loader
    except SystemExit as e:
        pytest.fail(f"evaluate_agent_main exited unexpectedly with code {e.code}")
    except Exception as e:
        pytest.fail(f"evaluate_agent_main raised an unexpected exception: {e}")

def test_evaluate_agent_main_produces_logs_and_charts(
    mock_config_dir, mock_data_loader, mock_sb3_ppo_load, mock_sb3_vecnormalize_load
):
    """Test that evaluation logs and chart files are created."""
    with patch('src.agents.evaluate_agent.plot_performance') as mock_plot_performance, \
         patch('json.dump') as mock_json_dump: # To check if history is saved
        
        evaluate_agent_main()
        
        # Check for creation of eval log directory
        logs_path = os.path.join(mock_config_dir, "logs", "evaluation")
        eval_run_dirs = [d for d in os.listdir(logs_path) if d.startswith('eval_')]
        assert len(eval_run_dirs) > 0
        
        # Check for trade history JSON file
        mock_json_dump.assert_called()
        args, kwargs = mock_json_dump.call_args
        # Check if the output file path is correct
        assert "evaluation_" in args[1].name # Check filename part

        # Check that plot_performance was called
        mock_plot_performance.assert_called_once()
        args, kwargs = mock_plot_performance.call_args
        assert "performance_chart.png" in args[2] # Check filename of plot

def test_plot_performance_generates_file(tmp_path):
    """Test plot_performance independently."""
    trade_history = [
        {'time': '2024-01-01T00:00:00Z', 'type': 'initial_balance', 'equity': 10000.0, 'balance': 10000.0},
        {'time': '2024-01-01T00:00:01Z', 'type': 'buy', 'price': 100.0, 'equity': 9000.0, 'balance': 9000.0},
        {'time': '2024-01-01T00:00:02Z', 'type': 'sell', 'price': 101.0, 'equity': 10100.0, 'balance': 10100.0}
    ]
    price_data = pd.Series(
        [99.0, 100.0, 101.0, 102.0],
        index=pd.to_datetime(['2024-01-01T00:00:00Z', '2024-01-01T00:00:01Z', '2024-01-01T00:00:02Z', '2024-01-01T00:00:03Z'], utc=True),
        name="BTCUSDT"
    )
    log_dir = tmp_path / "plot_test_logs"
    log_dir.mkdir()
    
    plot_performance(trade_history, price_data, "test_plot_run", str(log_dir))
    
    plot_file = log_dir / "test_plot_run_performance_chart.png"
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0 # File should not be empty