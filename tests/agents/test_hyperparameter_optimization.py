# tests/agents/test_hyperparameter_optimization.py
import pytest
import optuna
import json
import os
from unittest.mock import patch, MagicMock, ANY, mock_open

from src.agents.hyperparameter_optimization import (
    load_default_configs_for_optimization,
    objective,
    main as hpo_main
)
from src.data.utils import convert_to_native_types


@pytest.fixture
def mock_hpo_config_dir(tmp_path, monkeypatch):
    config_root = tmp_path / "hpo_test_project_root"
    config_root.mkdir()
    
    defaults_dir = config_root / "configs" / "defaults"
    defaults_dir.mkdir(parents=True)

    (defaults_dir / "run_settings.yaml").write_text("run_settings:\n  log_level: 'none'\n")
    (defaults_dir / "binance_settings.yaml").write_text("binance_settings:\n  historical_cache_dir: 'test_cache/'\n")
    (defaults_dir / "environment.yaml").write_text("environment:\n  kline_window_size: 20\n  tick_feature_window_size: 50\n")
    (defaults_dir / "ppo_params.yaml").write_text("ppo_params:\n  learning_rate: 0.0003\n  n_steps: 2048\n  total_timesteps: 10000\n  policy_kwargs: \"{'net_arch': [64, 64]}\"\n")
    (defaults_dir / "sac_params.yaml").write_text("sac_params:\n  learning_rate: 0.0003\n  buffer_size: 100000\n  total_timesteps: 10000\n  policy_kwargs: \"{'net_arch': [256, 256]}\"\n  gamma: 0.99\n  tau: 0.005\n  gradient_steps: 1\n")
    (defaults_dir / "hyperparameter_optimization.yaml").write_text(
        """
hyperparameter_optimization:
  study_name: "test_hpo_study"
  db_file: "test_hpo_optuna.db"
  load_if_exists: false
  direction: "maximize"
  n_trials: 3
  timeout_seconds: null
  sampler_type: "RandomSampler"
  seed: 42
  use_pruner: false
  trial_total_timesteps: 500

  ppo_optim_params:
    learning_rate: {low: 0.00001, high: 0.001, log: true}
    n_steps: {choices: [128, 256]}
    gamma: {low: 0.9, high: 0.99}
    policy_kwargs_net_arch: {choices: ["[32, 32]", "[64, 32]"]}

  sac_optim_params:
    learning_rate: {low: 0.0001, high: 0.0005, log: true}
    buffer_size: {choices: [10000, 20000]}
    gamma: {low: 0.9, high: 0.99}
    tau: {low: 0.001, high: 0.01}
    gradient_steps: {choices: [1, 2]}
    policy_kwargs_net_arch: {choices: ["(64, 64)", "(128, 128)"]}

  env_optim_params:
    kline_window_size: {choices: [5, 10]}
    tick_feature_window_size: {choices: [15, 25]}
"""
    )
    for fname in ["hash_keys.yaml", "ddpg_params.yaml", "a2c_params.yaml",
                  "recurrent_ppo_params.yaml", "evaluation_data.yaml", "live_trader_settings.yaml"]:
        (defaults_dir / fname).write_text("{}\n")

    (config_root / "config.yaml").write_text("agent_type: 'PPO'\n")

    original_cwd = os.getcwd()
    monkeypatch.chdir(config_root)
    yield config_root
    monkeypatch.chdir(original_cwd)


def test_load_default_configs_for_optimization(mock_hpo_config_dir):
    """Test that configurations are loaded and merged correctly for HPO."""
    loaded_config = load_default_configs_for_optimization()
    assert "run_settings" in loaded_config
    assert "hyperparameter_optimization" in loaded_config
    assert loaded_config["agent_type"] == "PPO"
    assert loaded_config["hyperparameter_optimization"]["study_name"] == "test_hpo_study"
    assert loaded_config["ppo_params"]["n_steps"] == 2048


@patch('src.agents.hyperparameter_optimization.train_agent')
def test_objective_ppo_suggestions(mock_train_agent, mock_hpo_config_dir):
    """Test PPO hyperparameter suggestions and passing to train_agent."""
    mock_train_agent.return_value = 0.75
    base_config = load_default_configs_for_optimization()
    base_config["agent_type"] = "PPO"
    base_config["hyperparameter_optimization"]["trial_total_timesteps"] = 555

    mock_trial = MagicMock(spec=optuna.trial.Trial)
    mock_trial.number = 1
    expected_lr, expected_n_steps, expected_gamma = 1.5e-4, 128, 0.92
    expected_net_arch_str, expected_kline_ws, expected_tick_ws = "[32, 32]", 10, 25

    def mock_suggest(name, *args, **kwargs):
        if name == "learning_rate": return expected_lr
        if name == "n_steps": return expected_n_steps
        if name == "gamma": return expected_gamma
        if name == "policy_kwargs_net_arch": return expected_net_arch_str
        if name == "kline_window_size": return expected_kline_ws
        if name == "tick_feature_window_size": return expected_tick_ws
        pytest.fail(f"Unexpected suggest call for name: {name}")

    mock_trial.suggest_float.side_effect = mock_suggest
    mock_trial.suggest_categorical.side_effect = mock_suggest
    mock_trial.should_prune.return_value = False
    mock_trial.params = {
        'learning_rate': expected_lr, 'n_steps': expected_n_steps, 'gamma': expected_gamma,
        'policy_kwargs_net_arch': expected_net_arch_str,
        'kline_window_size': expected_kline_ws, 'tick_feature_window_size': expected_tick_ws
    }

    returned_metric = objective(mock_trial, base_config)
    assert returned_metric == 0.75
    mock_train_agent.assert_called_once()
    call_args = mock_train_agent.call_args
    called_config_override = call_args[1]['config_override']
    assert called_config_override["ppo_params"]["learning_rate"] == expected_lr
    assert called_config_override["ppo_params"]["n_steps"] == expected_n_steps
    assert called_config_override["ppo_params"]["gamma"] == expected_gamma
    assert called_config_override["ppo_params"]["policy_kwargs"] == {"net_arch": eval(expected_net_arch_str)}
    assert called_config_override["ppo_params"]["total_timesteps"] == 555
    assert called_config_override["environment"]["kline_window_size"] == expected_kline_ws
    assert called_config_override["environment"]["tick_feature_window_size"] == expected_tick_ws
    assert call_args[1]['log_to_file'] is False


@patch('src.agents.hyperparameter_optimization.train_agent')
def test_objective_sac_suggestions(mock_train_agent, mock_hpo_config_dir):
    """Test SAC hyperparameter suggestions."""
    mock_train_agent.return_value = 0.65
    base_config = load_default_configs_for_optimization()
    base_config["agent_type"] = "SAC"
    base_config["hyperparameter_optimization"]["trial_total_timesteps"] = 666

    mock_trial = MagicMock(spec=optuna.trial.Trial)
    mock_trial.number = 2
    expected_lr, expected_buffer_size, expected_sac_gamma = 2e-4, 10000, 0.98
    expected_sac_tau, expected_sac_grad_steps, expected_sac_net_arch_str = 0.008, 2, "(128, 128)"

    def mock_suggest_sac(name, *args, **kwargs):
        if name == "learning_rate": return expected_lr
        if name == "buffer_size": return expected_buffer_size
        if name == "gamma": return expected_sac_gamma
        if name == "tau": return expected_sac_tau
        if name == "gradient_steps": return expected_sac_grad_steps
        if name == "sac_policy_kwargs_net_arch": return expected_sac_net_arch_str
        if name == "kline_window_size": return 5
        if name == "tick_feature_window_size": return 15
        pytest.fail(f"Unexpected SAC suggest call for name: {name}")

    mock_trial.suggest_float.side_effect = mock_suggest_sac
    mock_trial.suggest_categorical.side_effect = mock_suggest_sac
    mock_trial.should_prune.return_value = False
    mock_trial.params = {
        'learning_rate': expected_lr, 'buffer_size': expected_buffer_size,
        'gamma': expected_sac_gamma, 'tau': expected_sac_tau,
        'gradient_steps': expected_sac_grad_steps,
        'sac_policy_kwargs_net_arch': expected_sac_net_arch_str,
        'kline_window_size': 5, 'tick_feature_window_size': 15
    }

    objective(mock_trial, base_config)
    mock_train_agent.assert_called_once()
    called_config_override = mock_train_agent.call_args[1]['config_override']

    assert called_config_override["sac_params"]["learning_rate"] == expected_lr
    assert called_config_override["sac_params"]["buffer_size"] == expected_buffer_size
    assert called_config_override["sac_params"]["gamma"] == expected_sac_gamma
    assert called_config_override["sac_params"]["tau"] == expected_sac_tau
    assert called_config_override["sac_params"]["gradient_steps"] == expected_sac_grad_steps
    assert called_config_override["sac_params"]["policy_kwargs"] == {"net_arch": eval(expected_sac_net_arch_str)}
    assert called_config_override["sac_params"]["total_timesteps"] == 666


@patch('src.agents.hyperparameter_optimization.train_agent')
def test_objective_pruning_raises_exception(mock_train_agent, mock_hpo_config_dir):
    """Test that TrialPruned is raised if trial.should_prune() is True."""
    mock_train_agent.return_value = 0.1
    base_config = load_default_configs_for_optimization()
    base_config["agent_type"] = "PPO"

    mock_trial = MagicMock(spec=optuna.trial.Trial)
    mock_trial.number = 3
    mock_trial.suggest_float.return_value = 1e-4
    mock_trial.suggest_categorical.side_effect = [128, "[32]", 5, 15]
    mock_trial.should_prune.return_value = True
    mock_trial.params = {
        'learning_rate': 1e-4, 'n_steps': 128, 'policy_kwargs_net_arch': "[32]",
        'kline_window_size': 5, 'tick_feature_window_size': 15, 'gamma': 0.9
    }

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(mock_trial, base_config)
    mock_train_agent.assert_called_once()


@patch('src.agents.hyperparameter_optimization.load_default_configs_for_optimization')
@patch('optuna.create_study')
@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
def test_hpo_main_calls(mock_json_dump, mock_builtin_open_file, mock_create_study, mock_load_configs, mock_hpo_config_dir, tmp_path):
    """Test the main HPO orchestration logic."""
    test_study_name = "test_hpo_study_main_direct"
    test_db_file = "test_main_optuna_direct.db"
    n_trials_from_config = 2

    mock_loaded_config = {
        "agent_type": "PPO",
        "hyperparameter_optimization": {
            "study_name": test_study_name, "db_file": test_db_file, "load_if_exists": False,
            "direction": "maximize", "n_trials": n_trials_from_config, "timeout_seconds": None,
            "sampler_type": "TPESampler", "seed": 123, "use_pruner": True,
            "pruner_n_startup_trials": 1, "pruner_n_warmup_steps": 0, "pruner_interval_steps": 1,
        },
        "run_settings": {"log_level": "normal"}
    }
    mock_load_configs.return_value = mock_loaded_config

    mock_study_instance = MagicMock(spec=optuna.study.Study)
    mock_study_instance.best_trial = MagicMock()
    mock_study_instance.best_trial.value = 0.9
    mock_study_instance.best_trial.params = {"lr_mock": 1e-4, "n_steps_mock": 256}
    mock_study_instance.best_trial.number = 0
    mock_study_instance.trials = [mock_study_instance.best_trial]
    mock_create_study.return_value = mock_study_instance

    with patch('src.agents.hyperparameter_optimization.objective') as mock_hpo_objective_fn:
        mock_hpo_objective_fn.return_value = 0.88
        hpo_main()

        mock_create_study.assert_called_once()
        create_study_call_kwargs = mock_create_study.call_args.kwargs
        assert create_study_call_kwargs['study_name'] == test_study_name

        expected_relative_db_path_for_arg = os.path.join("optuna_studies", test_db_file)
        expected_storage_url_arg_string = f"sqlite:///{expected_relative_db_path_for_arg}"
        assert create_study_call_kwargs['storage'] == expected_storage_url_arg_string
        
        assert create_study_call_kwargs['load_if_exists'] is False
        assert create_study_call_kwargs['direction'] == "maximize"
        assert isinstance(create_study_call_kwargs['sampler'], optuna.samplers.TPESampler)
        assert isinstance(create_study_call_kwargs['pruner'], optuna.pruners.MedianPruner)

        mock_study_instance.optimize.assert_called_once()
        optimize_call_kwargs = mock_study_instance.optimize.call_args.kwargs
        assert optimize_call_kwargs['n_trials'] == n_trials_from_config
        assert optimize_call_kwargs['timeout'] is None
        assert optimize_call_kwargs['callbacks'] is None

        # CORRECTED: Assert the relative path string for the saved JSON file.
        expected_save_filename = f"best_hyperparameters_{test_study_name}.json"
        expected_relative_save_path = os.path.join("optuna_studies", expected_save_filename)
        
        mock_builtin_open_file.assert_any_call(expected_relative_save_path, 'w')
        
        mock_json_dump.assert_called_once_with(
            convert_to_native_types(mock_study_instance.best_trial.params),
            mock_builtin_open_file(),
            indent=4
        )