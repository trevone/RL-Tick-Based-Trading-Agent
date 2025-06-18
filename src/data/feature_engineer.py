# src/data/feature_engineer.py
import pandas as pd
import numpy as np
import re
from typing import Dict

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not found. Technical indicators will not be calculated. "
          "Please install with 'pip install TA-Lib' for full functionality.")

def get_indicator_calculators(high_np, low_np, close_np, open_np, volume_np):
    """
    Returns a dictionary mapping indicator names to their calculation functions.
    """
    if not TALIB_AVAILABLE:
        return {}
    return {
        'SMA': lambda **params: talib.SMA(close_np, **params),
        'EMA': lambda **params: talib.EMA(close_np, **params),
        'RSI': lambda **params: talib.RSI(close_np, **params),
        'MACD': lambda **params: talib.MACD(close_np, **params),
        'ADX': lambda **params: talib.ADX(high_np, low_np, close_np, **params),
        'STOCH': lambda **params: talib.STOCH(high_np, low_np, close_np, **params),
        'ATR': lambda **params: talib.ATR(high_np, low_np, close_np, **params),
        'BBANDS': lambda **params: talib.BBANDS(close_np, **params),
        'AD': lambda **params: talib.AD(high_np, low_np, close_np, volume_np),
        'OBV': lambda **params: talib.OBV(close_np, volume_np),
    }

def calculate_technical_indicators(df: pd.DataFrame, price_features_to_add: Dict) -> pd.DataFrame:
    """
    Calculates technical indicators on a DataFrame, enforcing the new dictionary format.
    """
    df_processed = df.copy()
    required_cols_for_ta = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if not isinstance(price_features_to_add, dict):
        print(f"ERROR: price_features_to_add must be a dictionary. Received {type(price_features_to_add)}. No features calculated.")
        return df_processed[[col for col in required_cols_for_ta if col in df_processed.columns]]

    if not TALIB_AVAILABLE:
        print("TA-Lib not available, skipping all technical indicator calculations.")
        return df_processed[[col for col in required_cols_for_ta if col in df_processed.columns]]

    for col in required_cols_for_ta:
        if col not in df_processed.columns:
            print(f"ERROR: Missing required column '{col}' for TA calculation. Returning original DF.")
            return df
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed.dropna(subset=required_cols_for_ta, inplace=True)
    if df_processed.empty:
        return df_processed

    high_np = df_processed['High'].values.astype(float)
    low_np = df_processed['Low'].values.astype(float)
    close_np = df_processed['Close'].values.astype(float)
    open_np = df_processed['Open'].values.astype(float)
    volume_np = df_processed['Volume'].values.astype(float)

    indicator_calculators = get_indicator_calculators(high_np, low_np, close_np, open_np, volume_np)

    for feature_name, feature_config in price_features_to_add.items():
        if feature_name in required_cols_for_ta:
            continue
        
        # 1. Parse the base indicator name (e.g., "RSI" from "RSI_7")
        # indicator_name_match = re.match(r"([A-Z]+)", feature_name)
        # if not indicator_name_match:
        #     print(f"Warning: Could not determine base indicator for '{feature_name}'. Skipping.")
        #     continue
        # indicator_name = indicator_name_match.group(1)
        
        try:
            params = feature_config.get('params', {}).copy()
            indicator_name = feature_config.get('function', '')

            # 2. Intelligently add 'timeperiod' from name if not specified in params
            # timeperiod_indicators = ['SMA', 'EMA', 'RSI', 'ADX', 'ATR']
            # if indicator_name in timeperiod_indicators and 'timeperiod' not in params:
            #     period_match = re.search(r'_(\d+)', feature_name)
            #     if period_match:
            #         params['timeperiod'] = int(period_match.group(1))

            calculator_func = indicator_calculators.get(indicator_name)

            if calculator_func:
                result = calculator_func(**params)
                if isinstance(result, tuple):
                    output_field = feature_config.get('output_field', 0)
                    df_processed[feature_name] = result[output_field]
                else:
                    df_processed[feature_name] = result
            elif indicator_name.startswith('CDL'):
                if hasattr(talib, indicator_name):
                    pattern_func = getattr(talib, indicator_name)
                    df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np)
                else:
                    print(f"Warning: Candlestick pattern '{indicator_name}' not found. Skipping.")
            else:
                print(f"Warning: Indicator '{indicator_name}' is not defined in the calculator. Skipping.")

        except Exception as e:
            print(f"Error calculating '{feature_name}' (Indicator: '{indicator_name}', Params: {params}): {e}. Skipping.")
            df_processed[feature_name] = np.nan

    # --- FIXED ---
    # Chained inplace operations return None. They must be on separate lines.
    df_processed.bfill(inplace=True)
    df_processed.ffill(inplace=True)
    df_processed.fillna(0, inplace=True)

    final_columns = list(price_features_to_add.keys())
    for col in final_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0.0

    return df_processed[final_columns]
