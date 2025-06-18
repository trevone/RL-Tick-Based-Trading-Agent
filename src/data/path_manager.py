# src/data/path_manager.py
import os
import re
import json
import hashlib
from src.data.config_loader import convert_to_native_types

DATA_CACHE_DIR = "data_cache/"
RANGE_CACHE_SUBDIR = "range_cache"

def get_data_path_for_day(date_str: str, symbol: str, data_type: str = "agg_trades",
                          interval: str = None, price_features_to_add: list = None,
                          cache_dir: str = DATA_CACHE_DIR, resample_interval_ms: int = None) -> str:
    
    # Base directory includes symbol and data_type
    base_dir = os.path.join(cache_dir, symbol, data_type)

    filename_prefix = ""
    safe_filename = ""

    if data_type == "agg_trades":
        filename_prefix = "bn_agg_trades" # Changed: Prefix updated from 'bn_aggtrades' to 'bn_agg_trades'
        resample_suffix = f"_R{resample_interval_ms}ms" if resample_interval_ms else ""
        
        # Changed: If resampled, add resample interval as a subdirectory
        if resample_suffix:
            # Remove leading underscore from suffix for directory name
            subdirectory = resample_suffix[1:] 
            full_data_path = os.path.join(base_dir, subdirectory)
        else:
            full_data_path = base_dir

        safe_filename = f"{filename_prefix}_{symbol}_{date_str}{resample_suffix}.parquet"
    elif data_type == "kline":
        if not interval: raise ValueError("Interval must be provided for kline data type.")
        if price_features_to_add:
            ta_hash = _generate_data_config_hash_key({"price_features": price_features_to_add})
        filename_prefix = "bn_klines"
        safe_filename = f"{filename_prefix}_{symbol}_{interval}_{date_str}_{ta_hash}.parquet"
        full_data_path = base_dir # kline data remains in data_type folder
    else:
        raise ValueError(f"Unsupported data_type: {data_type}. Must be 'agg_trades' or 'kline'.")
        
    return os.path.join(full_data_path, safe_filename)

def _generate_data_config_hash_key(params: dict, length: int = 10) -> str:
    if 'price_features' in params and isinstance(params['price_features'], list):
        params['price_features'] = sorted(params['price_features'])
    config_string = json.dumps(convert_to_native_types(params), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()[:length]

def _get_range_cache_path(symbol: str, start_date_str: str, end_date_str: str, data_type_suffix: str,
                          cache_config_params: dict, cache_dir: str = DATA_CACHE_DIR) -> str:
    symbol_cache_dir = os.path.join(cache_dir, symbol)
    range_cache_dir = os.path.join(symbol_cache_dir, RANGE_CACHE_SUBDIR)
    config_hash = _generate_data_config_hash_key(cache_config_params)
    safe_start = start_date_str.replace(" ", "_").replace(":", "")
    safe_end = end_date_str.replace(" ", "_").replace(":", "")
    filename = f"{config_hash}_{symbol}_{data_type_suffix}_{safe_start}_to_{safe_end}.parquet"
    return os.path.join(range_cache_dir, filename)