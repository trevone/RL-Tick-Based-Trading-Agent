# src/data/path_manager.py
import os
import re
import json
import hashlib
from src.data.config_loader import convert_to_native_types # Assuming this function exists and is relevant for config hashing

DATA_CACHE_DIR = "data_cache/"
RANGE_CACHE_SUBDIR = "range_cache"

def get_data_path_for_day(date_str: str, symbol: str, data_type: str = "agg_trades",
                          interval: str = None, cache_dir: str = DATA_CACHE_DIR,
                          resample_interval_ms: int = None,
                          kline_config_hash: str = None) -> str:
    """
    Constructs the file path for a single day's cached data,
    incorporating a hash for K-line configurations to support TA caching.

    Args:
        date_str (str): The date string (e.g., "YYYY-MM-DD").
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        data_type (str): The type of data ("agg_trades" or "kline").
        interval (str, optional): K-line interval (e.g., "1m", "1h"). Required for 'kline' data_type.
        cache_dir (str, optional): Base directory for data cache. Defaults to DATA_CACHE_DIR.
        resample_interval_ms (int, optional): Resampling interval in milliseconds for agg_trades.
        kline_config_hash (str, optional): A hash representing the K-line feature configuration.
                                           Used to differentiate cached K-line files with different TAs.

    Returns:
        str: The full path to the cached data file for the specified day.

    Raises:
        ValueError: If interval is not provided for kline data_type or data_type is unsupported.
    """
    # Base directory includes symbol and data_type
    base_dir = os.path.join(cache_dir, symbol, data_type)

    filename_prefix = ""
    safe_filename = ""

    if data_type == "agg_trades":
        filename_prefix = "bn_agg_trades"
        resample_suffix = f"_R{resample_interval_ms}ms" if resample_interval_ms else ""
        
        # If resampled, add resample interval as a subdirectory for better organization
        if resample_suffix:
            # Remove leading underscore from suffix for directory name
            subdirectory = resample_suffix[1:] 
            full_data_path = os.path.join(base_dir, subdirectory)
        else:
            full_data_path = base_dir

        safe_filename = f"{filename_prefix}_{symbol}_{date_str}{resample_suffix}.parquet"
    elif data_type == "kline":
        if not interval:
            raise ValueError("Interval must be provided for kline data type.")
        
        # Use config hash for filename suffix to distinguish feature sets
        config_hash_suffix = f"_{kline_config_hash}" if kline_config_hash else ""
        
        filename_prefix = "bn_klines"
        # The filename now includes the kline_config_hash to uniquely identify the feature set
        safe_filename = f"{filename_prefix}_{symbol}_{interval}_{date_str}{config_hash_suffix}.parquet"
        full_data_path = base_dir # kline data remains in data_type/symbol/kline folder
    else:
        raise ValueError(f"Unsupported data_type: {data_type}. Must be 'agg_trades' or 'kline'.")
    
    # Ensure the directory exists
    os.makedirs(full_data_path, exist_ok=True)
    return os.path.join(full_data_path, safe_filename)

def generate_config_hash(config_dict: dict, length: int = 10) -> str:
    """
    Generates a consistent MD5 hash for any dictionary configuration.
    This is used for generating unique filenames for cached data based on
    the configuration parameters, including technical indicators.

    Args:
        config_dict (dict): The dictionary configuration to hash.
        length (int): The desired length of the hash string.

    Returns:
        str: The generated hash string.
    """
    # Ensure consistent order by sorting keys and convert to native Python types
    # This is crucial for consistent hashing across different runs/environments
    processed_config = convert_to_native_types(config_dict)
    config_string = json.dumps(processed_config, sort_keys=True, ensure_ascii=False)
    
    # Hash the string representation of the configuration
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()[:length]

def _get_range_cache_path(symbol: str, start_date_str: str, end_date_str: str, data_type_suffix: str,
                          cache_config_params: dict, cache_dir: str = DATA_CACHE_DIR) -> str:
    """
    Constructs the file path for cached data spanning a date range.

    Args:
        symbol (str): The trading symbol.
        start_date_str (str): The start date string of the range.
        end_date_str (str): The end date string of the range.
        data_type_suffix (str): Suffix identifying the type of data (e.g., "klines", "agg_trades").
        cache_config_params (dict): A dictionary of configuration parameters that define the content
                                    of the cached range data (e.g., kline interval, TA features hash).
        cache_dir (str, optional): Base directory for data cache. Defaults to DATA_CACHE_DIR.

    Returns:
        str: The full path to the cached range data file.
    """
    symbol_cache_dir = os.path.join(cache_dir, symbol)
    range_cache_dir = os.path.join(symbol_cache_dir, RANGE_CACHE_SUBDIR)
    
    # Generate a hash for the cache configuration parameters
    config_hash = generate_config_hash(cache_config_params)
    
    # Create safe filenames by replacing problematic characters
    safe_start = re.sub(r'[^\w\s-]', '', start_date_str).replace(" ", "_")
    safe_end = re.sub(r'[^\w\s-]', '', end_date_str).replace(" ", "_")
    
    filename = f"{config_hash}_{symbol}_{data_type_suffix}_{safe_start}_to_{safe_end}.parquet"
    
    # Ensure the directory exists
    os.makedirs(range_cache_dir, exist_ok=True)
    return os.path.join(range_cache_dir, filename)

