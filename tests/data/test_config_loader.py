# tests/data/test_config_loader.py
import pytest
import pandas as pd
import numpy as np
import os
import yaml
from unittest.mock import patch

from src.data.config_loader import (
    _load_single_yaml_config,
    load_config,
    merge_configs,
    generate_config_hash,
    convert_to_native_types,
    get_relevant_config_for_hash
)
from src.utils import resolve_model_path

@pytest.fixture
def mock_configs_dir(tmp_path):
    """Creates a temporary config directory with default configs for testing."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    defaults_dir = cfg_dir / "defaults"
    defaults_dir.mkdir()

    (defaults_dir / "run_settings.yaml").write_text(
        "run_settings:\n  log_dir_base: 'logs/'\n  model_name: 'test_agent'\n  default_symbol: 'BTCUSDT'\n"
    )
    (defaults_dir / "environment.yaml").write_text(
        "environment:\n  kline_window_size: 20\n  tick_resample_interval_ms: 1000\n  kline_price_features: ['Close', 'Volume']\n  initial_balance: 10000\n"
    )
    (defaults_dir / "ppo_params.yaml").write_text(
        "ppo_params:\n  learning_rate: 0.0003\n  n_steps: 2048\n  policy_kwargs: \"{'net_arch': [64, 64]}\"\n"
    )
    (defaults_dir / "sac_params.yaml").write_text(
        "sac_params:\n  learning_rate: 0.001\n  buffer_size: 100000\n"
    )
    (defaults_dir / "binance_settings.yaml").write_text(
        "binance_settings:\n  historical_interval: '1h'\n"
    )
    (defaults_dir / "hash_keys.yaml").write_text(
        "hash_config_keys:\n"
        "  run_settings:\n"
        "    - default_symbol\n"
        "  environment:\n"
        "    - kline_window_size\n"
        "    - tick_resample_interval_ms\n"
        "    - initial_balance\n"
        "  agent_params:\n"
        "    PPO:\n"
        "      - learning_rate\n"
        "      - n_steps\n"
        "      - policy_kwargs\n"
        "    SAC:\n"
        "      - learning_rate\n"
    )
    (tmp_path / "config.yaml").write_text(
        "agent_type: 'PPO'\n"
        "environment:\n  initial_balance: 20000\n"
    )
    return str(tmp_path)


class TestConfigAndModelUtils:
    def test_convert_to_native_types(self):
        data = {
            "int": np.int64(5), "float": np.float64(3.14), "bool": np.bool_(True),
            "list_of_np": [np.int64(1), np.float64(2.2)],
            "array": np.array([1,2,3]),
            "timestamp": pd.Timestamp("2023-01-01T12:00:00Z")
        }
        native_data = convert_to_native_types(data)
        assert isinstance(native_data["int"], int)
        assert isinstance(native_data["float"], float)
        assert isinstance(native_data["bool"], bool)
        assert isinstance(native_data["list_of_np"][0], int)
        assert isinstance(native_data["list_of_np"][1], float)
        assert isinstance(native_data["array"], list)
        assert native_data["array"] == [1,2,3]
        assert native_data["timestamp"] == "2023-01-01T12:00:00+00:00"

    def test_generate_config_hash_consistency_and_difference(self):
        config1 = {"lr": 0.001, "arch": [64, 64], "gamma": 0.99}
        config2 = {"lr": 0.001, "arch": [64, 64], "gamma": 0.99}
        config3 = {"lr": 0.0001, "arch": [64, 64], "gamma": 0.99}
        config4 = {"gamma": 0.99, "arch": [64, 64], "lr": 0.001}

        hash1 = generate_config_hash(config1, length=8)
        hash2 = generate_config_hash(config2, length=8)
        hash3 = generate_config_hash(config3, length=8)
        hash4 = generate_config_hash(config4, length=8)

        assert isinstance(hash1, str)
        assert len(hash1) == 8
        assert hash1 == hash2
        assert hash1 != hash3
        assert hash1 == hash4

    def test_get_relevant_config_for_hash(self, mock_configs_dir, monkeypatch):
        monkeypatch.chdir(mock_configs_dir)
        default_paths=[
            "configs/defaults/run_settings.yaml",
            "configs/defaults/environment.yaml",
            "configs/defaults/ppo_params.yaml",
            "configs/defaults/binance_settings.yaml",
            "configs/defaults/hash_keys.yaml",
        ]
        effective_config = load_config(main_config_path="config.yaml", default_config_paths=default_paths)
        relevant_config = get_relevant_config_for_hash(effective_config)

        assert "run_settings" in relevant_config
        assert "environment" in relevant_config
        assert "ppo_params" in relevant_config
        assert relevant_config["run_settings"]["default_symbol"] == "BTCUSDT"
        assert relevant_config["environment"]["initial_balance"] == 20000
        assert "kline_price_features" not in relevant_config["environment"]
        assert relevant_config["ppo_params"]["learning_rate"] == 0.0003
        assert "gamma" not in relevant_config["ppo_params"]

    @patch('src.utils.os.path.exists')
    @patch('src.utils.get_relevant_config_for_hash')
    @patch('src.utils.generate_config_hash')
    def test_resolve_model_path_explicit_path_found(self, mock_gen_hash, mock_get_rel_conf, mock_exists, tmp_path):
        model_zip = tmp_path / "explicit_model.zip"
        model_zip.touch()
        effective_config = {"run_settings": {"model_path": str(model_zip)}}
        mock_exists.return_value = True

        model_path, alt_path = resolve_model_path(effective_config)
        assert model_path == str(model_zip)
        mock_exists.assert_called_once_with(str(model_zip))
        mock_get_rel_conf.assert_not_called()
        mock_gen_hash.assert_not_called()

    @patch('src.utils.os.path.exists')
    @patch('src.utils.get_relevant_config_for_hash')
    @patch('src.utils.generate_config_hash')
    def test_resolve_model_path_reconstruction_finds_best_model(self, mock_gen_hash, mock_get_rel_conf, mock_exists, tmp_path):
        log_dir_base = tmp_path / "logs_resolve"
        run_settings = {"model_path": None, "model_name": "test_agent_recon", "log_dir_base": str(log_dir_base)}
        
        effective_config = {
            "run_settings": run_settings,
            "agent_type": "PPO",
            "hash_config_keys": { "run_settings": ["model_name"] }
        }
        
        mock_get_rel_conf.return_value = {"run_settings": {"model_name": "test_agent_recon"}}
        mock_gen_hash.return_value = "abcdef"
        
        expected_run_dir = log_dir_base / "training" / "abcdef_test_agent_recon"
        expected_best_model_path = expected_run_dir / "best_model" / "best_model.zip"
        mock_exists.side_effect = lambda path: str(path) == str(expected_best_model_path)

        model_path, alt_path = resolve_model_path(effective_config)
        
        assert model_path == str(expected_best_model_path)
        mock_get_rel_conf.assert_called_once_with(effective_config)
        mock_gen_hash.assert_called_once_with({"run_settings": {"model_name": "test_agent_recon"}})
        mock_exists.assert_any_call(str(expected_best_model_path))


class TestConfigLoading:
    def test_load_single_yaml_config_valid(self, tmp_path):
        yaml_content = "key: value\nnested:\n  n_key: 123"
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        config = _load_single_yaml_config(str(p))
        assert config == {"key": "value", "nested": {"n_key": 123}}

    def test_load_single_yaml_config_missing(self):
        config = _load_single_yaml_config("non_existent_file.yaml")
        assert config == {}

    def test_merge_configs(self):
        default = {"a": 1, "b": {"x": 10, "y": 20}, "d": 100}
        loaded = {"b": {"y": 25, "z": 30}, "c": 3}
        merged = merge_configs(default, loaded)
        expected = {"a": 1, "b": {"x": 10, "y": 25, "z": 30}, "c": 3, "d": 100}
        assert merged == expected

    def test_load_config_integration(self, mock_configs_dir, monkeypatch):
        monkeypatch.chdir(mock_configs_dir)
        default_paths = [
            "configs/defaults/run_settings.yaml",
            "configs/defaults/environment.yaml",
            "configs/defaults/ppo_params.yaml",
            "configs/defaults/binance_settings.yaml",
            "configs/defaults/hash_keys.yaml"
        ]
        effective_config = load_config(main_config_path="config.yaml", default_config_paths=default_paths)
        assert effective_config["agent_type"] == "PPO"
        assert effective_config["environment"]["initial_balance"] == 20000
        assert effective_config["ppo_params"]["learning_rate"] == 0.0003