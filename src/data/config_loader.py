# src/data/config_loader.py
import yaml
import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict

def _load_single_yaml_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        print(f"Error loading YAML configuration from {config_path}: {e}")
        return {}

def load_config(main_config_path: str = "config.yaml",
                default_config_paths: List[str] = None) -> Dict:
    if default_config_paths is None:
        default_config_paths = []
    merged_config = {}
    for path in default_config_paths:
        default_cfg = _load_single_yaml_config(path)
        merged_config = merge_configs(merged_config, default_cfg)
    main_cfg = _load_single_yaml_config(main_config_path)
    merged_config = merge_configs(merged_config, main_cfg)
    return merged_config

def merge_configs(default_config: Dict, loaded_config: Dict) -> Dict:
    merged = default_config.copy()
    if loaded_config:
        for key, value in loaded_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged

def convert_to_native_types(data):
    if isinstance(data, list): return [convert_to_native_types(item) for item in data]
    if isinstance(data, dict): return {key: convert_to_native_types(value) for key, value in data.items()}
    if isinstance(data, np.integer): return int(data)
    if isinstance(data, np.floating): return float(data)
    if isinstance(data, np.ndarray): return data.tolist()
    if isinstance(data, (np.bool_, bool)): return bool(data)
    if isinstance(data, pd.Timestamp): return data.isoformat()
    return data

def generate_config_hash(config_dict: Dict, length: int = 7) -> str:
    config_string = json.dumps(convert_to_native_types(config_dict), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()[:length]

def get_relevant_config_for_hash(effective_config: Dict) -> Dict:
    relevant_config_for_hash = {}
    hash_keys_structure = effective_config.get("hash_config_keys", {})

    if "run_settings" in hash_keys_structure and isinstance(hash_keys_structure["run_settings"], list):
        relevant_config_for_hash["run_settings"] = {
            k: effective_config["run_settings"].get(k) for k in hash_keys_structure["run_settings"] if k in effective_config.get("run_settings",{})
        }

    if "environment" in hash_keys_structure and isinstance(hash_keys_structure["environment"], list):
        relevant_config_for_hash["environment"] = {
            k: effective_config["environment"].get(k) for k in hash_keys_structure["environment"] if k in effective_config.get("environment",{})
        }
    agent_type = effective_config.get("agent_type")
    if agent_type and "agent_params" in hash_keys_structure and isinstance(hash_keys_structure["agent_params"], dict):
        if agent_type in hash_keys_structure["agent_params"] and isinstance(hash_keys_structure["agent_params"][agent_type], list):
            agent_keys_to_hash = hash_keys_structure["agent_params"][agent_type]
            algo_params_section_name = f"{agent_type.lower()}_params"
            algo_params_section = effective_config.get(algo_params_section_name, {})
            relevant_agent_params = {}
            for k in agent_keys_to_hash:
                if k in algo_params_section:
                    value = algo_params_section.get(k)
                    if k == "policy_kwargs" and isinstance(value, str):
                        try: value = eval(value)
                        except: pass
                    relevant_agent_params[k] = value
            if relevant_agent_params:
                 relevant_config_for_hash[algo_params_section_name] = relevant_agent_params

    if "binance_settings" in hash_keys_structure and isinstance(hash_keys_structure["binance_settings"], list):
        relevant_config_for_hash["binance_settings"] = {
            k: effective_config["binance_settings"].get(k) for k in hash_keys_structure["binance_settings"] if k in effective_config.get("binance_settings",{})
        }
    return {k: v for k, v in relevant_config_for_hash.items() if v}