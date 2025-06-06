# tests/agents/test_evaluation.py
import pytest
import os
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Fix for Matplotlib backend in tests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Import the main function from the new path
from src.agents.evaluate_agent import main as evaluate_agent_main
from src.agents.evaluate_agent import plot_performance

# Import necessary SB3 components for mocking
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, SAC, DDPG, A2C
from stable_baselines3.common.running_mean_std import RunningMeanStd

# Corrected imports after refactoring
from src.data.path_manager import DATA_CACHE_DIR


@pytest.fixture
def mock_config_dir(tmp_path):
    """Creates a temporary config directory with default configs."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    (defaults_dir / "run_settings.yaml").write_text("""
run_settings:
  log_dir_base: 'logs/'
  model_name: 'test_agent'
  log_level: 'none'
  eval_log_dir: 'logs/evaluation/'
  model_path: 'logs/training/mock_hash_test_agent/best_model/best_model.zip'
  default_symbol: 'BTCUSDT'
  historical_interval: '1h'
  historical_cache_dir: 'data_cache/'
  start_date_eval: '2024-01-01 00:00:00'
  end_date_eval: '2024-01-01 23:59:59'
  n_evaluation_episodes: 1
""")
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 1\n  tick_feature_window_size: 1\n  kline_price_features: ['Close']\n  tick_features_to_use: ['Price', 'IsBuyerMaker']\n  initial_balance: 10000.0\n  tick_resample_interval_ms: 60000\n")
    (defaults_dir / "evaluation_data.yaml").write_text("# This file is now empty\n")
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings: {}\n")
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size', 'tick_resample_interval_ms', 'tick_features_to_use']\n  agent_params:\n    PPO: ['learning_rate']\n  run_settings: ['default_symbol']\n")
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.0003\n")

    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n")

    return str(tmp_path)

@pytest.fixture
def mock_data_loader():
    """Mocks data loading functions by patching them where they are used in evaluate_agent."""
    with patch('src.agents.evaluate_agent.load_kline_data_for_range') as mock_load_kline, \
         patch('src.agents.evaluate_agent.load_tick_data_for_range') as mock_load_tick:

        mock_kline_df = pd.DataFrame(
            {'Open': [100.0], 'High': [101.0], 'Low': [99.0], 'Close': [100.5], 'Volume': [10.0]},
            index=pd.to_datetime(['2024-01-01 00:00:00'], utc=True)
        )
        mock_tick_df = pd.DataFrame(
            {'Price': [100.0, 100.1, 100.2, 100.3, 100.4],
             'Quantity': [1.0, 1.1, 1.2, 1.3, 1.4],
             'IsBuyerMaker': [False, True, False, True, False]},
            index=pd.to_datetime(['2024-01-01 00:00:00.000', '2024-01-01 00:00:00.001',
                                  '2024-01-01 00:00:00.002', '2024-01-01 00:00:00.003',
                                  '2024-01-01 00:00:00.004'], utc=True)
        )

        mock_load_kline.return_value = mock_kline_df
        mock_load_tick.return_value = mock_tick_df
        yield mock_load_kline, mock_load_tick

@pytest.fixture
def mock_sb3_ppo_load():
    """Mocks PPO.load to prevent actual model loading."""
    with patch('stable_baselines3.PPO.load') as mock_ppo_load:
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = (np.array([[0.0, 0.01]], dtype=np.float32), None)
        mock_ppo_load.return_value = mock_model_instance
        yield mock_ppo_load

@pytest.fixture
def mock_sb3_vecnormalize_load():
    """
    Mocks VecNormalize.load to simulate its behavior of modifying and returning
    the passed venv instance, without actual file loading.
    """
    def mock_load_method_side_effect(load_path_arg, venv_arg):
        venv_arg.training = False
        venv_arg.norm_reward = False
        return venv_arg

    with patch.object(VecNormalize, 'load', side_effect=mock_load_method_side_effect, autospec=True) as mock_load:
        yield mock_load


@pytest.fixture(autouse=True)
def setup_teardown_dirs(tmp_path, monkeypatch):
    """Sets up temporary log/data_cache directories and cleans up."""
    project_root = tmp_path

    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    training_logs_dir = logs_dir / "training"
    training_logs_dir.mkdir(exist_ok=True)
    mock_model_run_dir = training_logs_dir / "mock_hash_test_agent"
    mock_model_run_dir.mkdir(exist_ok=True)
    best_model_dir = mock_model_run_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True)
    (best_model_dir / "best_model.zip").touch()
    (mock_model_run_dir / "vec_normalize.pkl").touch()
    (logs_dir / "evaluation").mkdir(exist_ok=True)

    data_cache_dir = project_root / "data_cache"
    data_cache_dir.mkdir(exist_ok=True)

    monkeypatch.chdir(project_root)

    # --- THIS IS THE FIX ---
    # The problematic import of the deleted utils file is removed.
    # We now patch the DATA_CACHE_DIR constant in the new path_manager module.
    import src.data.path_manager
    monkeypatch.setattr(src.data.path_manager, 'DATA_CACHE_DIR', str(data_cache_dir))
    
    import src.agents.evaluate_agent
    if hasattr(src.agents.evaluate_agent, 'DATA_CACHE_DIR'):
         monkeypatch.setattr(src.agents.evaluate_agent, 'DATA_CACHE_DIR', str(data_cache_dir))
    # --- END FIX ---
    
    yield

def test_evaluate_agent_main_runs_successfully(
    mock_config_dir, mock_data_loader, mock_sb3_ppo_load, mock_sb3_vecnormalize_load
):
    """Test that the main evaluation function runs without critical errors."""
    try:
        mock_kline_loader, mock_tick_loader = mock_data_loader
        with patch('src.agents.evaluate_agent.load_default_configs_for_evaluation') as mock_load_defaults:
            dummy_model_path = os.path.join("logs", "training", "mock_hash_test_agent", "best_model", "best_model.zip")
            mock_load_defaults.return_value = {
                "agent_type": "PPO",
                "run_settings": {
                    "log_level": "none",
                    "eval_log_dir": "logs/evaluation/",
                    "model_path": dummy_model_path,
                    "default_symbol": "BTCUSDT",
                    "historical_interval": "1h",
                    "historical_cache_dir": "data_cache/",
                    "start_date_eval": '2024-01-01 00:00:00',
                    "end_date_eval": '2024-01-01 23:59:59',
                    "n_evaluation_episodes": 1,
                },
                "environment": {"kline_window_size": 1, "tick_feature_window_size": 1, "kline_price_features": ['Close'], "tick_features_to_use": ['Price', 'IsBuyerMaker'], "initial_balance":10000.0, "tick_resample_interval_ms": 60000},
                "binance_settings": {},
                "evaluation_data": {},
                "hash_config_keys": {}
            }
            evaluate_agent_main()

        mock_sb3_ppo_load.assert_called_once()
        mock_kline_loader.assert_called()
        mock_tick_loader.assert_called()
    except SystemExit as e:
        pytest.fail(f"evaluate_agent_main exited unexpectedly with code {e.code}")
    except Exception as e:
        pytest.fail(f"evaluate_agent_main raised an unexpected exception: {e}")

def test_evaluate_agent_main_produces_logs_and_charts(
    mock_config_dir, mock_data_loader, mock_sb3_ppo_load, mock_sb3_vecnormalize_load
):
    """Test that evaluation logs and chart files are created."""
    with patch('src.agents.evaluate_agent.plot_performance') as mock_plot_performance, \
         patch('json.dump') as mock_json_dump, \
         patch('src.agents.evaluate_agent.load_default_configs_for_evaluation') as mock_load_defaults:

        dummy_model_path = os.path.join("logs", "training", "mock_hash_test_agent", "best_model", "best_model.zip")
        eval_log_dir_relative = os.path.join("logs", "evaluation")

        mock_load_defaults.return_value = {
            "agent_type": "PPO",
            "run_settings": {
                "log_level": "none",
                "eval_log_dir": eval_log_dir_relative,
                "model_path": dummy_model_path,
                "default_symbol": "BTCUSDT",
                "historical_interval": "1h",
                "historical_cache_dir": "data_cache/",
                "start_date_eval": '2024-01-01 00:00:00',
                "end_date_eval": '2024-01-01 23:59:59',
                "n_evaluation_episodes": 1
            },
            "environment": {"kline_window_size": 1, "tick_feature_window_size": 1, "kline_price_features": ['Close'], "tick_features_to_use": ['Price', 'IsBuyerMaker'], "initial_balance":10000.0, "tick_resample_interval_ms": 60000},
            "binance_settings": {},
            "evaluation_data": {},
            "hash_config_keys": {}
        }

        mock_kline_loader, mock_tick_loader = mock_data_loader
        evaluate_agent_main()

        eval_log_dir_base_path = os.path.join(str(mock_config_dir), eval_log_dir_relative)
        assert os.path.isdir(eval_log_dir_base_path)
        eval_run_dirs = [d for d in os.listdir(eval_log_dir_base_path) if os.path.isdir(os.path.join(eval_log_dir_base_path, d)) and d.startswith('eval_')]
        assert len(eval_run_dirs) > 0

        mock_plot_performance.assert_called_once()
        args_plot, kwargs_plot = mock_plot_performance.call_args
        eval_run_id_from_plot_call = args_plot[2]
        eval_log_dir_for_this_run_from_plot_call = args_plot[3]
        abs_path_from_plot_call = os.path.abspath(eval_log_dir_for_this_run_from_plot_call)
        abs_expected_eval_log_dir_for_run = os.path.abspath(os.path.join(eval_log_dir_base_path, eval_run_id_from_plot_call))
        assert abs_path_from_plot_call == abs_expected_eval_log_dir_for_run

        mock_json_dump.assert_called()

def test_plot_performance_generates_file(tmp_path):
    """Test plot_performance independently."""
    trade_history = [
        {'time': '2024-01-01T00:00:00Z', 'type': 'initial_balance', 'equity': 10000.0, 'balance': 10000.0, 'price':0, 'volume':0, 'pnl':0, 'commission':0, 'profit_target':0},
        {'time': '2024-01-01T00:00:01Z', 'type': 'buy', 'price': 100.0, 'volume': 1.0, 'equity': 9000.0, 'balance': 9000.0, 'profit_target_set': 0.01, 'pnl': 0.0, 'commission': 0.0},
        {'time': '2024-01-01T00:00:02Z', 'type': 'sell', 'price': 101.0, 'volume': 1.0, 'equity': 10100.0, 'balance': 10100.0, 'profit_target_aimed': 0.01, 'pnl': 100.0, 'commission': 0.0, 'pnl_ratio_achieved': 0.01}
    ]
    price_data = pd.Series(
        [99.0, 100.0, 101.0, 102.0],
        index=pd.to_datetime(['2024-01-01T00:00:00Z', '2024-01-01T00:00:01Z', '2024-01-01T00:00:02Z', '2024-01-01T00:00:03Z'], utc=True),
        name="BTCUSDT"
    )
    log_dir = tmp_path / "plot_test_logs"
    log_dir.mkdir()

    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.show') as mock_show:
        plot_performance(trade_history, price_data, "test_plot_run", str(log_dir), log_level="none")

    expected_plot_file = os.path.join(str(log_dir), "test_plot_run_performance_chart.png")
    mock_savefig.assert_called_once_with(expected_plot_file)