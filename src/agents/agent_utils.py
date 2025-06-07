# src/agents/agent_utils.py
import os
from ..data.config_loader import load_config

def load_default_configs_for_training(config_dir="configs/defaults") -> dict:
    """Loads all default configurations for a training run."""
    default_config_paths = [
        os.path.join(config_dir, "run_settings.yaml"),
        os.path.join(config_dir, "environment.yaml"),
        os.path.join(config_dir, "ppo_params.yaml"),
        os.path.join(config_dir, "sac_params.yaml"),
        os.path.join(config_dir, "ddpg_params.yaml"),
        os.path.join(config_dir, "a2c_params.yaml"),
        os.path.join(config_dir, "recurrent_ppo_params.yaml"),
        os.path.join(config_dir, "binance_settings.yaml"),
        os.path.join(config_dir, "evaluation_data.yaml"),
        os.path.join(config_dir, "hash_keys.yaml"),
    ]
    return load_config(main_config_path="config.yaml", default_config_paths=default_config_paths)

def load_default_configs_for_evaluation(config_dir="configs/defaults") -> dict:
    """Loads all default configurations for an evaluation run."""
    # This function is often the same as for training, but kept separate for future flexibility.
    return load_default_configs_for_training(config_dir)