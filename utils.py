# utils.py
import yaml
import os
import pandas as pd
import numpy as np
import traceback
import json
import hashlib
from datetime import datetime, timezone, timedelta
import time
from tqdm.auto import tqdm # Import tqdm for general progress bars
import talib 

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_CLIENT_AVAILABLE = True
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False
    print("CRITICAL ERROR: python-binance library not found. This project now exclusively uses Binance for data. "
          "Please install with 'pip install python-binance' for the scripts to function.")

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at {config_path}. Using script fallbacks if available.")
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        print(f"Error loading YAML configuration from {config_path}: {e}")
        return {}

def _calculate_technical_indicators(df: pd.DataFrame, price_features_to_add: list) -> pd.DataFrame:
    if df.empty:
        return df
    df_processed = df.copy()

    # Ensure required columns are present and numeric
    required_cols_for_ta = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols_for_ta:
        if col not in df_processed.columns:
            print(f"ERROR: Missing required column '{col}' for TA calculation. Cannot calculate TAs.")
            df_processed[col] = np.nan # Use NaN first to allow TA-Lib to handle it
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # Ensure numeric

    # Drop rows with NaN in critical columns at the start to prevent TA-Lib errors
    df_processed.dropna(subset=['High', 'Low', 'Close'], inplace=True)
    if df_processed.empty:
        print("WARNING: DataFrame became empty after dropping NaNs for TA calculation.")
        return df_processed

    # Prepare numpy arrays from pandas Series for TA-Lib
    high_np = df_processed['High'].values
    low_np = df_processed['Low'].values
    close_np = df_processed['Close'].values
    open_np = df_processed['Open'].values
    volume_np = df_processed['Volume'].values

    # --- Dynamic TA-Lib Calculations for Indicators and Candlestick Patterns ---
    for feature_name in price_features_to_add:
        try:
            # Moving Averages
            if feature_name.startswith('SMA_'):
                timeperiod = int(feature_name.split('_')[1])
                if not df_processed['Close'].empty:
                    df_processed[feature_name] = talib.SMA(close_np, timeperiod=timeperiod)
                else: df_processed[feature_name] = np.nan

            elif feature_name.startswith('EMA_'):
                timeperiod = int(feature_name.split('_')[1])
                if not df_processed['Close'].empty:
                    df_processed[feature_name] = talib.EMA(close_np, timeperiod=timeperiod)
                else: df_processed[feature_name] = np.nan

            # Oscillators
            elif feature_name.startswith('RSI_'):
                timeperiod = int(feature_name.split('_')[1])
                if not df_processed['Close'].empty:
                    df_processed[feature_name] = talib.RSI(close_np, timeperiod=timeperiod)
                else: df_processed[feature_name] = np.nan

            elif feature_name == 'MACD':
                if not df_processed['Close'].empty:
                    macd, macdsignal, macdhist = talib.MACD(close_np, fastperiod=12, slowperiod=26, signalperiod=9)
                    df_processed['MACD'] = macd
                else: df_processed[feature_name] = np.nan

            elif feature_name == 'ADX':
                if not (df_processed['High'].empty or df_processed['Low'].empty or df_processed['Close'].empty):
                    df_processed['ADX'] = talib.ADX(high_np, low_np, close_np, timeperiod=14)
                else: df_processed[feature_name] = np.nan
            
            elif feature_name == 'STOCH_K':
                if not (df_processed['High'].empty or df_processed['Low'].empty or df_processed['Close'].empty):
                    stoch_k, stoch_d = talib.STOCH(high_np, low_np, close_np,
                                                  fastk_period=5, slowk_period=3, slowd_period=3)
                    df_processed['STOCH_K'] = stoch_k
                else: df_processed[feature_name] = np.nan
            
            # Volatility Indicators
            elif feature_name == 'ATR':
                if not (df_processed['High'].empty or df_processed['Low'].empty or df_processed['Close'].empty):
                    df_processed['ATR'] = talib.ATR(high_np, low_np, close_np, timeperiod=14)
                else: df_processed[feature_name] = np.nan
            
            elif feature_name == 'BBANDS_Upper':
                if not df_processed['Close'].empty:
                    upper, middle, lower = talib.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                    df_processed['BBANDS_Upper'] = upper
                else: df_processed[feature_name] = np.nan

            # Volume Indicators
            elif feature_name == 'AD':
                if not (df_processed['High'].empty or df_processed['Low'].empty or df_processed['Close'].empty or df_processed['Volume'].empty):
                    df_processed['AD'] = talib.AD(high_np, low_np, close_np, volume_np)
                else: df_processed[feature_name] = np.nan

            elif feature_name == 'OBV':
                if not (df_processed['Close'].empty or df_processed['Volume'].empty):
                    df_processed['OBV'] = talib.OBV(close_np, volume_np)
                else: df_processed[feature_name] = np.nan
            
            # Candlestick Patterns
            elif feature_name.startswith('CDL'):
                # All CDL patterns require OHLC.
                # Special handling for patterns returning multiple outputs or needing extra params (like penetration for stars)
                if not (df_processed['Open'].empty or df_processed['High'].empty or df_processed['Low'].empty or df_processed['Close'].empty):
                    if feature_name == 'CDLMORNINGSTAR':
                        df_processed[feature_name] = talib.CDLMORNINGSTAR(open_np, high_np, low_np, close_np, penetration=0)
                    elif feature_name == 'CDLEVENINGSTAR':
                        df_processed[feature_name] = talib.CDLEVENINGSTAR(open_np, high_np, low_np, close_np, penetration=0)
                    else:
                        # Use getattr to call the function dynamically by its string name
                        if hasattr(talib, feature_name):
                            pattern_func = getattr(talib, feature_name)
                            # Most CDL functions just take OHLC
                            df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np)
                        else:
                            print(f"Warning: TA-Lib function '{feature_name}' not found for candlestick pattern.")
                            df_processed[feature_name] = np.nan
                else:
                    df_processed[feature_name] = np.nan
            
            # Default case for base price features (OHLCV) that might not be processed by TA-Lib
            elif feature_name in required_cols_for_ta:
                pass # Already handled at the top, just ensure it's not flagged as uncalculated later

            else:
                # If a feature name is requested but not in the defined logic above
                print(f"Warning: Technical indicator or pattern '{feature_name}' requested but no specific calculation logic defined. Assigning NaN.")
                df_processed[feature_name] = np.nan # Assign NaN, will be filled below

        except Exception as e:
            print(f"Error calculating TA '{feature_name}': {e}. Assigning NaN.")
            traceback.print_exc()
            df_processed[feature_name] = np.nan # Assign NaN if error during calculation

    # After TA-Lib calculations, fill any leading NaNs generated by the TA-Lib functions.
    df_processed.bfill(inplace=True) # Backward fill
    df_processed.ffill(inplace=True) # Forward fill
    df_processed.fillna(0, inplace=True) # Final fallback for any remaining NaNs (e.g., if series was all NaNs)
    
    # Final check to ensure all requested features are present, filling with 0 if somehow still missing
    for feature in price_features_to_add:
        if feature not in df_processed.columns:
            print(f"Warning: Requested feature '{feature}' is still missing after all calculations and fills. Setting to 0.0.")
            df_processed[feature] = 0.0

    return df_processed

def fetch_and_cache_kline_data(
    symbol: str, interval: str, start_date_str: str, end_date_str: str,
    cache_dir: str, 
    price_features_to_add: list = None,
    api_key: str = None, api_secret: str = None, testnet: bool = False,
    cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2
) -> pd.DataFrame:
    if not BINANCE_CLIENT_AVAILABLE:
        print("CRITICAL ERROR in fetch_and_cache_kline_data: python-binance library not found.")
        return pd.DataFrame()

    price_features_to_add = price_features_to_add if price_features_to_add is not None else []
    os.makedirs(cache_dir, exist_ok=True)

    # Removed sorted_features_str from cache_specifier for concise filenames
    cache_specifier = f"bn_klines_{symbol}_{interval}_{start_date_str}_to_{end_date_str}" 
    
    safe_filename = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in cache_specifier])
    cache_file = os.path.join(cache_dir, f"{safe_filename}.{cache_file_type}")

    if os.path.exists(cache_file):
        if log_level in ["normal", "detailed"]: print(f"Loading K-line data from cache: {cache_file}")
        try:
            df = pd.read_parquet(cache_file) if cache_file_type == "parquet" else pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.index.tz is None and not df.empty: df.index = df.index.tz_localize('UTC')
            # Check if columns are consistent, if not, refetch (e.g., if TA features changed)
            # This check is still necessary to ensure cached data matches requested features
            if not all(f in df.columns for f in price_features_to_add if f not in ['Open', 'High', 'Low', 'Close', 'Volume']):
                print(f"Warning: Cached K-line data missing some requested TA features. Refetching: {cache_file}")
                os.remove(cache_file) # Delete incomplete cache
                return pd.DataFrame() # Signal to re-fetch
            return df
        except Exception as e:
            print(f"Error loading K-line data from cache {cache_file}: {e}. Refetching.")
            if os.path.exists(cache_file): os.remove(cache_file) # Delete corrupt cache
            return pd.DataFrame()

    if log_level in ["normal", "detailed"]:
        print(f"Fetching K-line data for {symbol}, Interval: {interval}, From: {start_date_str}, To: {end_date_str}")
    
    client = Client(api_key or os.environ.get('BINANCE_API_KEY'), 
                    api_secret or os.environ.get('BINANCE_API_SECRET'), 
                    testnet=testnet)
    if testnet: client.API_URL = 'https://testnet.binance.vision/api'

    try:
        # Removed BarSpinner as it caused ModuleNotFoundError; basic print is sufficient for single call
        klines_raw = client.get_historical_klines(symbol, interval, start_date_str, end_str=end_date_str)

        if not klines_raw:
            if log_level != "none": print(f"No k-lines returned for {symbol} {start_date_str}-{end_date_str}.")
            return pd.DataFrame()

        kline_cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 
                        'NumberofTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore']
        df_fetched = pd.DataFrame(klines_raw, columns=kline_cols)
        
        df_fetched['OpenTimeDate'] = pd.to_datetime(df_fetched['OpenTime'], unit='ms', utc=True)
        df_fetched.set_index('OpenTimeDate', inplace=True) # Index by k-line open time

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df_fetched.columns: df_fetched[col] = pd.to_numeric(df_fetched[col], errors='coerce')
        
        df_to_process = df_fetched[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_to_process = df_to_process.astype(float)

        df_with_ta = _calculate_technical_indicators(df_to_process, price_features_to_add)
        
        if not df_with_ta.empty:
            try:
                if cache_file_type == "parquet": df_with_ta.to_parquet(cache_file)
                else: df_with_ta.to_csv(cache_file)
                if log_level in ["normal", "detailed"]: print(f"K-line data with TAs saved to cache: {cache_file}")
            except Exception as e: print(f"Error saving K-line data to cache {cache_file}: {e}")
        
        return df_with_ta

    except BinanceAPIException as bae:
        print(f"Binance API Exception during K-line fetch: Code={bae.code}, Message='{bae.message}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error during K-line fetch/processing: {e}")
        traceback.print_exc(); return pd.DataFrame()

def fetch_continuous_aggregate_trades(
    symbol: str, start_date_str: str, end_date_str: str, 
    cache_dir: str, api_key: str = None, api_secret: str = None, 
    testnet: bool = False, cache_file_type: str = "parquet", log_level: str = "normal",
    api_request_delay_seconds: float = 0.2
) -> pd.DataFrame:
    if not BINANCE_CLIENT_AVAILABLE:
        print("CRITICAL ERROR in fetch_continuous_aggregate_trades: python-binance library not found.")
        return pd.DataFrame()

    os.makedirs(cache_dir, exist_ok=True)
    
    # Updated cache specifier to include full timestamps to ensure unique naming for precise ranges
    start_dt_for_filename = pd.to_datetime(start_date_str, utc=True)
    end_dt_for_filename = pd.to_datetime(end_date_str, utc=True)
    
    start_filename_str = start_dt_for_filename.strftime("%Y-%m-%d_%H-%M-%S")
    end_filename_str = end_dt_for_filename.strftime("%Y-%m-%d_%H-%M-%S")

    cache_specifier = f"bn_aggtrades_{symbol}_{start_filename_str}_to_{end_filename_str}"
    safe_filename = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in cache_specifier])
    cache_file = os.path.join(cache_dir, f"{safe_filename}.{cache_file_type}")

    if os.path.exists(cache_file):
        if log_level in ["normal", "detailed"]: print(f"Loading aggregate trades from cache: {cache_file}")
        try:
            df = pd.read_parquet(cache_file) if cache_file_type == "parquet" else pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.index.tz is None and not df.empty: df.index = df.index.tz_localize('UTC')
            return df
        except Exception as e:
            print(f"Error loading aggregate trades from cache {cache_file}: {e}. Refetching.")
            if os.path.exists(cache_file): os.remove(cache_file) # Delete corrupt cache
            return pd.DataFrame()

    if log_level in ["normal", "detailed"]:
        print(f"Fetching continuous aggregate trades for {symbol} from {start_date_str} to {end_date_str}.")

    client = Client(api_key or os.environ.get('BINANCE_API_KEY'), 
                    api_secret or os.environ.get('BINANCE_API_SECRET'), 
                    testnet=testnet)
    if testnet: client.API_URL = 'https://testnet.binance.vision/api'

    all_trades = []
    start_dt = pd.to_datetime(start_date_str, utc=True)
    end_dt = pd.to_datetime(end_date_str, utc=True)
    
    initial_start_ms = int(start_dt.timestamp() * 1000)
    current_start_ms = initial_start_ms
    end_ms = int(end_dt.timestamp() * 1000)

    total_duration_ms = end_ms - initial_start_ms
    if total_duration_ms <= 0:
        if log_level != "none": print(f"Invalid or zero duration date range: {start_date_str} to {end_date_str}")
        return pd.DataFrame()

    pbar = None
    if log_level in ["normal", "detailed"]:
        # Total is the full duration of the fetch period in milliseconds
        pbar = tqdm(total=total_duration_ms, desc=f"Fetching {symbol} trades", unit="ms", unit_scale=True, disable=(log_level == "none"), leave=False)


    while True:
        try:
            if log_level == "detailed":
                print(f"  Fetching agg trades. Current start_ms: {current_start_ms}")

            chunk_end_ms = min(current_start_ms + (60*60*1000 -1) , end_ms)

            if log_level == "detailed": print(f"    Fetching chunk: Start {current_start_ms}, End {chunk_end_ms}")

            trades_chunk = client.get_aggregate_trades(
                symbol=symbol, 
                startTime=current_start_ms,
                endTime=chunk_end_ms,
                limit=1000
            )
            
            if trades_chunk:
                all_trades.extend(trades_chunk)
                current_start_ms = trades_chunk[-1]['T'] + 1 
                if log_level == "detailed": print(f"    Fetched {len(trades_chunk)} trades in chunk. New start_ms: {current_start_ms}")
            else:
                if current_start_ms < end_ms:
                     current_start_ms = chunk_end_ms + 1
                     if log_level == "detailed": print(f"    No trades in chunk, advancing window. New start_ms: {current_start_ms}")
                else:
                    break

            if pbar:
                # Calculate current progress in milliseconds
                current_pbar_pos = current_start_ms - initial_start_ms
                pbar.n = min(current_pbar_pos, total_duration_ms) # Ensure current position doesn't exceed total
                pbar.set_postfix_str(f"Trades: {len(all_trades)}")
                pbar.refresh()

            if current_start_ms >= end_ms:
                if pbar: pbar.close() # Close progress bar when done
                break
            
            time.sleep(api_request_delay_seconds)

        except BinanceAPIException as bae:
            if pbar: pbar.close() # Close pbar on error
            print(f"Binance API Exception during aggregate trade fetch: {bae.code} - {bae.message}. Retrying or stopping...")
            time.sleep(api_request_delay_seconds * 5)
            break 
        except Exception as e:
            if pbar: pbar.close() # Close pbar on error
            print(f"Error fetching aggregate trades: {e}")
            traceback.print_exc()
            break

    if not all_trades:
        if log_level != "none": print(f"No aggregate trades returned for {symbol} {start_date_str}-{end_date_str}.")
        return pd.DataFrame()

    df_trades = pd.DataFrame(all_trades)
    df_trades.rename(columns={'T': 'Timestamp', 'p': 'Price', 'q': 'Quantity', 'm': 'IsBuyerMaker', 'a': 'TradeId'}, inplace=True)
    df_trades['Timestamp'] = pd.to_datetime(df_trades['Timestamp'], unit='ms', utc=True)
    df_trades.set_index('Timestamp', inplace=True)
    df_trades[['Price', 'Quantity']] = df_trades[['Price', 'Quantity']].astype(float)
    
    df_trades_final = df_trades[['Price', 'Quantity', 'IsBuyerMaker']].copy()

    if not df_trades_final.empty:
        try:
            if cache_file_type == "parquet": df_trades_final.to_parquet(cache_file)
            else: df_trades_final.to_csv(cache_file)
            if log_level in ["normal", "detailed"]: print(f"Aggregate trades saved to cache: {cache_file}")
        except Exception as e: print(f"Error saving aggregate trades to cache {cache_file}: {e}")
            
    return df_trades_final


def merge_configs(default_config: dict, loaded_config: dict, section_name: str = "") -> dict:
    merged = default_config.copy()
    if loaded_config:
        for key, value in loaded_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value, f"{section_name}.{key if section_name else key}")
            else:
                merged[key] = value
    return merged

def generate_config_hash(config_dict: dict, length: int = 7) -> str:
    config_string = json.dumps(convert_to_native_types(config_dict), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()[:length]

def convert_to_native_types(data):
    if isinstance(data, list): return [convert_to_native_types(item) for item in data]
    if isinstance(data, dict): return {key: convert_to_native_types(value) for key, value in data.items()}
    if isinstance(data, np.integer): return int(data)
    if isinstance(data, np.floating): return float(data)
    if isinstance(data, np.ndarray): return data.tolist() 
    if isinstance(data, (np.bool_, bool)): return bool(data)
    if isinstance(data, pd.Timestamp): return data.isoformat()
    return data

def get_relevant_config_for_hash(
        full_yaml_config: dict,
        train_script_fallback_config: dict, 
        env_script_fallback_config: dict 
    ) -> dict:
    # Use the full config (after merging with defaults) for getting effective values
    effective_train_cfg = train_script_fallback_config 
    effective_env_cfg = effective_train_cfg.get("environment", env_script_fallback_config)
    effective_bn_cfg = effective_train_cfg.get("binance_settings", {})
    effective_ppo_cfg = effective_train_cfg.get("ppo_params", {}) 

    # Use the 'hash_config_keys' from the *full_yaml_config* (which comes from config.yaml)
    # as this defines what the user *wants* to be hashed.
    hash_keys_structure = full_yaml_config.get("hash_config_keys", {}) 

    relevant_config_for_hash = {}

    if "environment" in hash_keys_structure and isinstance(hash_keys_structure["environment"], list):
        env_keys_to_hash = hash_keys_structure["environment"]
        relevant_config_for_hash["environment"] = {
            k: effective_env_cfg.get(k) for k in env_keys_to_hash if k in effective_env_cfg
        }

    if "ppo_params" in hash_keys_structure and isinstance(hash_keys_structure["ppo_params"], list):
        ppo_keys_to_hash = hash_keys_structure["ppo_params"]
        relevant_config_for_hash["ppo_params"] = {
            k: effective_ppo_cfg.get(k) for k in ppo_keys_to_hash if k in effective_ppo_cfg
        }
        # Special handling for policy_kwargs which can be a string in YAML
        if "policy_kwargs" in relevant_config_for_hash.get("ppo_params", {}) and \
           isinstance(relevant_config_for_hash["ppo_params"]["policy_kwargs"], str):
            try:
                # Safely convert string to dict for hashing consistency
                relevant_config_for_hash["ppo_params"]["policy_kwargs"] = eval(relevant_config_for_hash["ppo_params"]["policy_kwargs"])
            except Exception:
                pass # Keep as string if eval fails or not string

    if "binance_settings" in hash_keys_structure and isinstance(hash_keys_structure["binance_settings"], list):
        bn_keys_to_hash = hash_keys_structure["binance_settings"]
        relevant_config_for_hash["binance_settings"] = {
            k: effective_bn_cfg.get(k) for k in bn_keys_to_hash if k in effective_bn_cfg
        }
    return {k: v for k, v in relevant_config_for_hash.items() if v}


def resolve_model_path(eval_specific_config: dict, full_yaml_config: dict, train_script_fallback_config: dict, env_script_fallback_config: dict, log_level: str = "normal") -> tuple:
    model_load_path = eval_specific_config.get("model_path") or eval_specific_config.get("model_load_path")
    alt_model_load_path = eval_specific_config.get("alt_model_path")

    if model_load_path and os.path.exists(model_load_path) and model_load_path.endswith(".zip"):
        if log_level in ["normal", "detailed"]: print(f"Using explicit model_path: {model_load_path}")
        return model_load_path, alt_model_load_path
    elif model_load_path: 
         if log_level != "none": print(f"Warning: Explicit model_path '{model_load_path}' not found or invalid. Attempting reconstruction.")

    if log_level in ["normal", "detailed"]: print("Attempting to reconstruct model path from training config hash...")
    
    # Reconstruct the effective_config as it would have been in training for hashing
    # IMPORTANT: The `train_script_fallback_config` here should be the `DEFAULT_TRAIN_CONFIG`
    # from your `train_simple_agent.py` to ensure hashing consistency.
    temp_effective_config_for_hash = merge_configs(train_script_fallback_config, full_yaml_config) 
    relevant_config_for_hash = get_relevant_config_for_hash(full_yaml_config, temp_effective_config_for_hash, env_script_fallback_config)

    train_log_dir_base = temp_effective_config_for_hash.get("run_settings", {}).get("log_dir_base")
    train_base_model_name = temp_effective_config_for_hash.get("run_settings", {}).get("model_name")

    if train_log_dir_base and train_base_model_name:
        if not relevant_config_for_hash and log_level != "none": print("Warning: Cannot auto-find model, relevant config for hash is empty.")
        else:
            config_hash = generate_config_hash(relevant_config_for_hash)
            final_model_name_with_hash = f"{config_hash}_{train_base_model_name}"
            expected_run_dir = os.path.join(train_log_dir_base, final_model_name_with_hash)
            
            if log_level == "detailed": print(f"Expected run directory for model (based on current config for hash): {expected_run_dir}")
            if log_level == "detailed": print(f"  Relevant parts for hash were: {json.dumps(convert_to_native_types(relevant_config_for_hash), indent=2, sort_keys=True)}")


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


if __name__ == '__main__':
    print("--- Testing utils.py (New Structure with Tick and K-line Fetchers) ---")

    cfg_test = load_config()
    # Dummy DEFAULT_TRAIN_CONFIG for utils test's `get_relevant_config_for_hash`
    # In a real setup, this would be imported from `train_simple_agent.py`
    temp_default_train_config_for_test = {
        "run_settings": {"log_level": "normal", "log_dir_base": "./logs/ppo_trading_test/", "model_name": "test_model"},
        "environment": {
            "kline_price_features": ["Open", "High", "Low", "Close", "Volume"] # Minimal default for features
        },
        "ppo_params": {"total_timesteps": 100}, # Minimal default
        "binance_settings": {
            "default_symbol": "BTCUSDT",
            "historical_interval": "1h",
            "historical_cache_dir": "./binance_data_cache_test_utils/",
            "api_key": os.environ.get("BINANCE_API_KEY"),
            "api_secret": os.environ.get("BINANCE_API_SECRET"),
            "testnet": True,
            "api_request_delay_seconds": 0.1
        },
        "hash_config_keys": { # Dummy hash keys for this test only, should align with config.yaml
            "environment": ["kline_price_features"],
            "binance_settings": ["default_symbol", "historical_interval"]
        }
    }

    # Merge with a dummy default to ensure all expected keys are present for testing.
    test_config_merged = merge_configs(temp_default_train_config_for_test, cfg_test)
    test_config_merged["environment"] = merge_configs(test_config_merged["environment"], cfg_test.get("environment", {}))


    current_utils_log_level = test_config_merged.get("run_settings", {}).get("log_level")
    bn_settings_test = test_config_merged.get("binance_settings", {})
    env_settings_test_features = test_config_merged.get("environment", {}).get("kline_price_features", [])

    if BINANCE_CLIENT_AVAILABLE:
        print("\n--- Testing fetch_and_cache_kline_data ---")
        kline_df = fetch_and_cache_kline_data(
            symbol=bn_settings_test.get("default_symbol"),
            interval=bn_settings_test.get("historical_interval"),
            start_date_str=test_config_merged.get("start_date_kline_test", "2024-04-01"),
            end_date_str=test_config_merged.get("end_date_kline_test", "2024-04-02"),
            cache_dir=bn_settings_test.get("historical_cache_dir"),
            price_features_to_add=env_settings_test_features,
            api_key=bn_settings_test.get("api_key"),
            api_secret=bn_settings_test.get("api_secret"),
            testnet=bn_settings_test.get("testnet"),
            log_level=current_utils_log_level,
            api_request_delay_seconds=bn_settings_test.get("api_request_delay_seconds")
        )
        if not kline_df.empty:
            print(f"K-line fetch successful. Shape: {kline_df.shape}, Columns: {kline_df.columns.tolist()}")
            print("K-line Head:\n", kline_df.head())
        else:
            print("K-line fetch FAILED or returned empty DataFrame.")

        print("\n--- Testing fetch_continuous_aggregate_trades ---")
        tick_start = test_config_merged.get("start_date_tick_test", "2024-04-01 10:00:00")
        tick_end = test_config_merged.get("end_date_tick_test", "2024-04-01 10:05:00")
        
        print(f"Fetching aggregate trades for {bn_settings_test.get('default_symbol')} from {tick_start} to {tick_end}")

        agg_trades_df = fetch_continuous_aggregate_trades(
            symbol=bn_settings_test.get("default_symbol"),
            start_date_str=tick_start,
            end_date_str=tick_end,
            cache_dir=bn_settings_test.get("historical_cache_dir"),
            api_key=bn_settings_test.get("api_key"),
            api_secret=bn_settings_test.get("api_secret"),
            testnet=bn_settings_test.get("testnet"),
            log_level=current_utils_log_level,
            api_request_delay_seconds=bn_settings_test.get("api_request_delay_seconds")
        )
        if not agg_trades_df.empty:
            print(f"Aggregate trades fetch successful. Shape: {agg_trades_df.shape}, Columns: {agg_trades_df.columns.tolist()}")
            print("Agg Trades Head:\n", agg_trades_df.head())
        else:
            print("Agg Trades fetch FAILED or returned empty DataFrame.")
            print("Note: For aggregate trades, ensure the test period has trading activity on Binance Testnet (if used) or Mainnet.")

    else:
        print("Skipped Binance data fetching tests as python-binance library is not available.")
    
    print("\n--- Utils.py testing finished ---")