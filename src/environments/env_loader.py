# src/environments/env_loader.py
import os
import importlib
import inspect
from pathlib import Path

# Import the base class to check for subclasses
from src.environments.base_env import SimpleTradingEnv

def load_environments(experiments_dir="src/environments/experiments"):
    """
    Dynamically discovers and loads all valid environment classes from a directory.

    A valid environment is a .py file containing a class that inherits from SimpleTradingEnv.

    Returns:
        A dictionary mapping the environment's name (from its filename) to its class.
        e.g., {'experimental_env': <class '...ExperimentalTradingEnv'>}
    """
    available_envs = {}
    experiments_path = Path(experiments_dir)
    
    # Add the base environment by default
    available_envs['simple'] = SimpleTradingEnv

    if not experiments_path.is_dir():
        print(f"Warning: Experiments directory not found at '{experiments_dir}'. Only 'simple' env is available.")
        return available_envs

    for file_path in experiments_path.iterdir():
        # Process only Python files, excluding __init__.py
        if file_path.is_file() and file_path.name.endswith(".py") and file_path.name != "__init__.py":
            module_name = file_path.stem  # e.g., "experimental_env"
            
            # Create the full import path, e.g., src.environments.experiments.experimental_env
            import_path = f"{experiments_path.parts[0]}.{experiments_path.parts[1]}.{experiments_path.parts[2]}.{module_name}"

            try:
                # Dynamically import the module
                module = importlib.import_module(import_path)

                # Inspect the module for classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if the class is a subclass of SimpleTradingEnv but not SimpleTradingEnv itself
                    if issubclass(obj, SimpleTradingEnv) and obj is not SimpleTradingEnv:
                        # Use the filename as the key for the environment
                        available_envs[module_name] = obj
                        print(f"Discovered experimental environment: '{module_name}' -> {obj.__name__}")
                        # Found our class, no need to check others in this file
                        break 
            except ImportError as e:
                print(f"Error importing experimental environment '{module_name}': {e}")

    return available_envs