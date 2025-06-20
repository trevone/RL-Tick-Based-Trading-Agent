# src/data/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict

from .custom_indicators import get_custom_indicator_calculators

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not found. Technical indicators will not be calculated. "
          "Please install with 'pip install TA-Lib' for full functionality.")

def calculate_technical_indicators(df: pd.DataFrame, price_features_to_add: Dict) -> pd.DataFrame:
    """
    Calculates technical indicators on a DataFrame, enforcing the new dictionary format.
    This version dynamically selects the data source (High, Low, Close, etc.) based on the config.
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

    data_sources = {
        'High': df_processed['High'].values.astype(float),
        'Low': df_processed['Low'].values.astype(float),
        'Close': df_processed['Close'].values.astype(float),
        'Open': df_processed['Open'].values.astype(float),
        'Volume': df_processed['Volume'].values.astype(float),
    }
    
    custom_calculators = get_custom_indicator_calculators()

    for feature_name, feature_config in price_features_to_add.items():
        if feature_name in required_cols_for_ta:
            continue
        
        try:
            params = feature_config.get('params', {}).copy()
            indicator_name = feature_config.get('function', '')

            data_source_key = feature_config.get('data_source', 'Close')
            input_data = data_sources.get(data_source_key)

            if input_data is None:
                print(f"Warning: Data source '{data_source_key}' not found for feature '{feature_name}'. Skipping.")
                continue

            result = None
            
            if indicator_name in custom_calculators:
                calculator_func = custom_calculators[indicator_name]
                result = calculator_func(input_data, **params)
            elif indicator_name.startswith('CDL'):
                if hasattr(talib, indicator_name):
                    pattern_func = getattr(talib, indicator_name)
                    result = pattern_func(data_sources['Open'], data_sources['High'], data_sources['Low'], data_sources['Close'])
                else:
                    print(f"Warning: Candlestick pattern '{indicator_name}' not found. Skipping.")
            elif hasattr(talib, indicator_name):
                calculator_func = getattr(talib, indicator_name)

                if indicator_name in ['AD', 'ADOSC']:
                    result = calculator_func(data_sources['High'], data_sources['Low'], data_sources['Close'], data_sources['Volume'], **params)
                elif indicator_name == 'OBV':
                    result = calculator_func(data_sources['Close'], data_sources['Volume'], **params)
                else:
                    result = calculator_func(input_data, **params)

            elif indicator_name:
                print(f"Warning: Indicator '{indicator_name}' is not defined. Skipping.")

            if result is not None:
                if isinstance(result, tuple):
                    output_field = feature_config.get('output_field', 0)
                    df_processed[feature_name] = result[output_field]
                else:
                    df_processed[feature_name] = result

        except Exception as e:
            print(f"Error calculating '{feature_name}' (Indicator: '{indicator_name}', Params: {params}): {e}. Skipping.")
            df_processed[feature_name] = np.nan

    df_processed.bfill(inplace=True)
    df_processed.ffill(inplace=True)
    df_processed.fillna(0, inplace=True)

    final_columns = list(price_features_to_add.keys())
    for col in final_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0.0

    return df_processed[final_columns]