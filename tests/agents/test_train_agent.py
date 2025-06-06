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

    # FIXED: Centralized all run-time and data settings into run_settings.yaml
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
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 1\n  tick_feature_window_size: 1\n  kline_price_features: ['Close']\n  tick_features_to_use: ['Price']\n  tick_resample_interval_ms: 1000\n")
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.001\n  total_timesteps: 100\n  policy_kwargs: \"{{'net_arch': [32]}}\"\n")
    (defaults_dir / "sac_params.yaml").write_text("sac_params:\n  learning_rate: 0.0005\n  total_timesteps: 50\n  buffer_size: 10000\n")
    # These files are now empty of the moved keys
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings: {}\n")
    (defaults_dir / "evaluation_data.yaml").write_text("evaluation_data: {}\n")
    (defaults_dir / "hash_keys.yaml").write_text("hash_config_keys:\n  environment: ['kline_window_size', 'tick_resample_interval_ms']\n  agent_params:\n    PPO: ['learning_rate']\n    SAC: ['learning_rate', 'buffer_size']\n  run_settings: ['default_symbol']\n")

    (tmp_path / "config.yaml").write_text("agent_type: 'PPO'\n")

    return str(tmp_path)

@pytest.fixture
def mock_data_loader():
    """Mocks data loading functions and yields the mock objects."""
    with patch('src.agents.train_agent.load_kline_data_for_range') as mock_load_kline, \
         patch('src.agents.train_agent.load_tick_data_for_range') as mock_load_tick:

        # Setup minimal valid DataFrames to allow train_agent to proceed
        mock_kline_df = pd.DataFrame(
            {'Open': [100.0]*2, 'High': [101.0]*2, 'Low': [99.0]*2, 'Close': [100.5]*2, 'Volume': [10.0]*2}, # Ensure enough data for small window
            index=pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 01:00:00'], utc=True)
        )
        mock_tick_df = pd.DataFrame(
            {'Price': [100.0]*2, 'Quantity': [1.0]*2, 'IsBuyerMaker': [False]*2}, # Ensure enough data
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
        # Mock the predict method if necessary for more complex interactions
        # model_instance_mock.predict.return_value = (env.action_space.sample(), None)

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

    # Ensure base log directory for training runs exists if log_to_file=True
    logs_dir_for_runs = project_root / "test_logs" / "training" # Matches mock_config_dir default
    logs_dir_for_runs.mkdir(parents=True, exist_ok=True)

    # Tensorboard logs base
    tensorboard_logs_base = project_root / "logs" / "tensorboard_logs"
    tensorboard_logs_base.mkdir(parents=True, exist_ok=True)

    # Default data cache used if not overridden by config
    default_data_cache_dir = project_root / "data_cache"
    default_data_cache_dir.mkdir(parents=True, exist_ok=True)

    # Test-specific data cache, if binance_settings.historical_cache_dir is set to this in mock_config_dir
    test_data_cache_from_config = project_root / "test_data_cache"
    test_data_cache_from_config.mkdir(parents=True, exist_ok=True)


    monkeypatch.chdir(project_root) # Run tests from tmp_path as project root

    # Patch the DATA_CACHE_DIR constant in src.data.utils if necessary,
    # though a well-behaved train_agent should rely on config.
    import src.data.utils
    monkeypatch.setattr(src.data.utils, 'DATA_CACHE_DIR', str(default_data_cache_dir))

    yield


# --- Tests for train_agent ---

def test_train_agent_ppo_setup(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that train_agent sets up PPO correctly."""
    final_metric = train_agent(log_to_file=False) # log_to_file=False simplifies test

    assert final_metric == -np.inf # Default if EvalCallback doesn't run or is mocked minimally

    mock_sb3_models["PPO"].assert_called_once()
    args, kwargs = mock_sb3_models["PPO"].call_args
    assert kwargs['learning_rate'] == 0.001

    mock_sb3_models["PPO"].return_value.learn.assert_called_once()
    learn_args, learn_kwargs = mock_sb3_models["PPO"].return_value.learn.call_args
    assert learn_kwargs['total_timesteps'] == 100
    assert learn_kwargs['tb_log_name'] is None # Due to log_to_file=False
    assert learn_kwargs['progress_bar'] is False # Due to log_level="none" when log_to_file=False
    assert learn_kwargs['callback'] is not None # Even if empty list or specific mock


def test_train_agent_sac_setup(mock_config_dir, mock_data_loader, mock_sb3_models):
    """Test that train_agent sets up SAC correctly with an override."""
    # Create a config.yaml in the mock_config_dir that specifies SAC
    with open(os.path.join(mock_config_dir, "config.yaml"), 'w') as f:
        f.write("agent_type: 'SAC'\n") # This will be loaded by load_default_configs_for_training

    # train_agent will load configs, including the one above.
    final_metric = train_agent(log_to_file=False)

    assert final_metric == -np.inf

    mock_sb3_models["SAC"].assert_called_once()
    args, kwargs = mock_sb3_models["SAC"].call_args
    # Check against defaults from sac_params.yaml in mock_config_dir
    assert kwargs['learning_rate'] == 0.0005
    assert kwargs['buffer_size'] == 10000

    mock_sb3_models["SAC"].return_value.learn.assert_called_once()
    learn_args, learn_kwargs = mock_sb3_models["SAC"].return_value.learn.call_args
    assert learn_kwargs['total_timesteps'] == 50


def test_train_agent_eval_callback_creation(mock_config_dir, mock_data_loader, mock_sb3_models, tmp_path):
    """Test that EvalCallback is set up and its result is returned by train_agent if log_to_file=True."""
    # Ensure log_dir_base points to tmp_path for this test when log_to_file=True
    log_base = tmp_path / "eval_callback_test_logs"
    config_override = {
        "run_settings": {"log_dir_base": str(log_base)}
    }

    with patch('src.agents.train_agent.EvalCallback', autospec=True) as MockEvalCallback:
        mock_eval_callback_instance = MagicMock()
        mock_eval_callback_instance.best_mean_reward = 123.45
        MockEvalCallback.return_value = mock_eval_callback_instance

        # log_to_file=True is needed for EvalCallback to be fully set up with paths
        final_metric = train_agent(config_override=config_override, log_to_file=True)

        MockEvalCallback.assert_called_once()
        args, kwargs = MockEvalCallback.call_args
        assert kwargs['eval_freq'] > 0
        assert kwargs['log_path'] is not None # Should point within tmp_path
        assert kwargs['best_model_save_path'] is not None # Should point within tmp_path
        assert final_metric == 123.45


def test_train_agent_no_eval_callback_if_no_data(mock_config_dir, mock_sb3_models, tmp_path):
    """Test that EvalCallback is skipped if eval data is empty, even if log_to_file=True."""
    log_base = tmp_path / "no_eval_data_test_logs"
    config_override = {
        "run_settings": {"log_dir_base": str(log_base)}
    }

    # Mock data loaders to return empty DataFrames for evaluation data
    with patch('src.agents.train_agent.load_kline_data_for_range') as mock_load_kline, \
         patch('src.agents.train_agent.load_tick_data_for_range') as mock_load_tick:

        # First calls are for training data (return valid DFs)
        mock_kline_df_train = pd.DataFrame(
            {'Open': [100.0]*2, 'High': [101.0]*2, 'Low': [99.0]*2, 'Close': [100.5]*2, 'Volume': [10.0]*2},
            index=pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 01:00:00'], utc=True)
        )
        mock_tick_df_train = pd.DataFrame(
            {'Price': [100.0]*2, 'Quantity': [1.0]*2, 'IsBuyerMaker': [False]*2},
            index=pd.to_datetime(['2024-01-01 00:00:00.000', '2024-01-01 00:00:00.001'], utc=True)
        )

        # Subsequent calls (for eval data) return empty DFs
        mock_load_kline.side_effect = [mock_kline_df_train, pd.DataFrame()] # Train, then Eval
        mock_load_tick.side_effect = [mock_tick_df_train, pd.DataFrame()]  # Train, then Eval

        with patch('src.agents.train_agent.EvalCallback') as MockEvalCallbackConstructor:
            # Call train_agent with log_to_file=True to attempt EvalCallback creation
            final_metric = train_agent(config_override=config_override, log_to_file=True)
            MockEvalCallbackConstructor.assert_not_called()
            assert final_metric == -np.inf # Default when no eval


# --- NEW TESTS FOR CACHE DIR AND LOG LEVEL PROPAGATION ---

def test_train_agent_uses_correct_cache_dir(mock_config_dir, mock_data_loader, mock_sb3_models, tmp_path):
    """
    Tests that train_agent passes the 'historical_cache_dir' from config
    to the data loading functions.
    """
    mock_load_kline, mock_load_tick = mock_data_loader

    custom_cache_dir = str(tmp_path / "my_unique_test_cache")

    # FIXED: Override historical_cache_dir within the correct run_settings key
    config_override = {
        "run_settings": {
            "historical_cache_dir": custom_cache_dir,
            "log_level": "none"
        },
        "ppo_params": {"total_timesteps": 10},
    }

    train_agent(config_override=config_override, log_to_file=False)

    # Assert that the mocked data loading functions were called with the correct cache_dir
    assert mock_load_kline.call_count > 0, "load_kline_data_for_range was not called"
    for k_call_args in mock_load_kline.call_args_list:
        _, kwargs = k_call_args
        assert kwargs.get("cache_dir") == custom_cache_dir, \
            f"Kline loader called with cache_dir {kwargs.get('cache_dir')}, expected {custom_cache_dir}"

    assert mock_load_tick.call_count > 0, "load_tick_data_for_range was not called"
    for t_call_args in mock_load_tick.call_args_list:
        _, kwargs = t_call_args
        assert kwargs.get("cache_dir") == custom_cache_dir, \
            f"Tick loader called with cache_dir {kwargs.get('cache_dir')}, expected {custom_cache_dir}"


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

    assert mock_load_kline.call_count > 0
    for k_call_args in mock_load_kline.call_args_list:
        _, kwargs = k_call_args
        assert kwargs.get("log_level") == log_level_setting, \
            f"Kline loader (log_to_file=True) called with log_level {kwargs.get('log_level')}, expected {log_level_setting}"

    assert mock_load_tick.call_count > 0
    for t_call_args in mock_load_tick.call_args_list:
        _, kwargs = t_call_args
        assert kwargs.get("log_level") == log_level_setting, \
            f"Tick loader (log_to_file=True) called with log_level {kwargs.get('log_level')}, expected {log_level_setting}"


    # Scenario 2: log_to_file = False
    mock_load_kline.reset_mock()
    mock_load_tick.reset_mock()

    train_agent(config_override=config_override, log_to_file=False)
    expected_propagated_log_level_for_no_file = "none"

    assert mock_load_kline.call_count >= 1
    for k_call_args in mock_load_kline.call_args_list:
        _, kwargs = k_call_args
        assert kwargs.get("log_level") == expected_propagated_log_level_for_no_file, \
             f"Kline loader (log_to_file=False) called with log_level {kwargs.get('log_level')}, expected {expected_propagated_log_level_for_no_file}"

    assert mock_load_tick.call_count >= 1
    for t_call_args in mock_load_tick.call_args_list:
        _, kwargs = t_call_args
        assert kwargs.get("log_level") == expected_propagated_log_level_for_no_file, \
            f"Tick loader (log_to_file=False) called with log_level {kwargs.get('log_level')}, expected {expected_propagated_log_level_for_no_file}"