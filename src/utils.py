# src/utils.py
import os
import json
from src.data.config_loader import get_relevant_config_for_hash, generate_config_hash, convert_to_native_types

def resolve_model_path(effective_config: dict, log_level: str = "normal") -> tuple:
    run_settings = effective_config.get("run_settings", {})
    model_load_path = run_settings.get("model_path")
    alt_model_load_path = run_settings.get("alt_model_path")

    if model_load_path and os.path.exists(model_load_path) and model_load_path.endswith(".zip"):
        if log_level in ["normal", "detailed"]: print(f"Using explicit model_path: {model_load_path}")
        return model_load_path, alt_model_load_path
    elif model_load_path and log_level != "none":
         print(f"Warning: Explicit model_path '{model_load_path}' not found or invalid. Attempting reconstruction.")

    if log_level in ["normal", "detailed"]: print("Attempting to reconstruct model path from training config hash...")
    relevant_config_for_hash = get_relevant_config_for_hash(effective_config)
    train_log_dir_base = run_settings.get("log_dir_base", "logs/")

    train_base_model_name = run_settings.get("model_name")
    if train_log_dir_base and train_base_model_name:
        if not relevant_config_for_hash and log_level != "none":
            print("Warning: Cannot auto-find model, relevant config for hash is empty.")
        else:
            config_hash = generate_config_hash(relevant_config_for_hash)
            final_model_name_with_hash = f"{config_hash}_{train_base_model_name}"

            if "training" not in train_log_dir_base.lower().replace("\\", "/").split("/"):
                 expected_run_dir_base_for_training = os.path.join(train_log_dir_base, "training")
            else:
                 expected_run_dir_base_for_training = train_log_dir_base

            expected_run_dir = os.path.join(expected_run_dir_base_for_training, final_model_name_with_hash)

            if log_level == "detailed":
                print(f"Expected run directory for model (based on current config for hash): {expected_run_dir}")
                print(f"  Relevant parts for hash were: {json.dumps(convert_to_native_types(relevant_config_for_hash), indent=2, sort_keys=True)}")

            path_best_model = os.path.join(expected_run_dir, "best_model", "best_model.zip")
            path_final_model = os.path.join(expected_run_dir, "trained_model_final.zip")

            if os.path.exists(path_best_model):
                if log_level in ["normal", "detailed"]: print(f"Found best model: {path_best_model}")
                return path_best_model, path_final_model if os.path.exists(path_final_model) else alt_model_load_path
            elif os.path.exists(path_final_model):
                if log_level in ["normal", "detailed"]: print(f"Found final model: {path_final_model}")
                return path_final_model, alt_model_load_path
            elif log_level != "none": print(f"No standard models found in {expected_run_dir} (check hash and paths).")
    elif log_level != "none": print("Cannot reconstruct: training log_dir_base or model_name missing in current effective config.")

    if log_level != "none": print("No valid model path found through explicit path or reconstruction.")
    return None, alt_model_load_path