# src/data/feature_engineer.py
import pandas as pd
import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("CRITICAL WARNING: TA-Lib not found. Technical indicators will NOT be calculated. "
          "Please install with 'pip install TA-Lib' (or via conda) for full functionality. "
          "Refer to TA-Lib installation instructions for your OS.")

def get_indicator_calculators(high_np, low_np, close_np, open_np, volume_np):
    """
    Returns a dictionary mapping indicator names to their calculation functions.
    These functions accept parameters as keyword arguments (**params),
    making them highly flexible for configuration-driven feature engineering.
    """
    if not TALIB_AVAILABLE:
        print("INFO: TA-Lib is not available. Skipping creation of indicator calculators.")
        return {} # Return an empty dictionary if TA-Lib is not available

    calculators = {
        # Moving Averages
        'SMA': lambda **params: talib.SMA(close_np, **params),
        'EMA': lambda **params: talib.EMA(close_np, **params),
        # Momentum Indicators
        'RSI': lambda **params: talib.RSI(close_np, **params),
        'MACD': lambda **params: talib.MACD(close_np, **params),
        'ADX': lambda **params: talib.ADX(high_np, low_np, close_np, **params),
        'STOCH': lambda **params: talib.STOCH(high_np, low_np, close_np, **params),
        # Volatility Indicators
        'ATR': lambda **params: talib.ATR(high_np, low_np, close_np, **params),
        'BBANDS': lambda **params: talib.BBANDS(close_np, **params),
        # Volume Indicators
        'AD': lambda **params: talib.AD(high_np, low_np, close_np, volume_np, **params),
        'OBV': lambda **params: talib.OBV(close_np, volume_np, **params),
    }

    # Verify that each function actually exists in TA-Lib
    # This loop specifically checks the functions that are *expected* to be in TA-Lib
    for name, func in list(calculators.items()): # Use list() to allow modification during iteration
        # For lambda functions, we can't directly check 'hasattr(talib, name)' easily.
        # Instead, we try a dummy call or ensure the core TA-Lib functions are present.
        # A simpler check is just whether the 'talib' module itself is loaded.
        # The primary check is TALIB_AVAILABLE at the top of the file.
        # If it's not available, then the functions won't be in 'talib' anyway.
        pass # The outer TALIB_AVAILABLE check is sufficient.

    return calculators

def calculate_technical_indicators(df: pd.DataFrame, technical_indicators_config: dict) -> pd.DataFrame:
    """
    Calculates technical indicators based on the provided configuration and adds them
    as new columns to the DataFrame. Handles indicators returning multiple outputs
    and candlestick patterns.

    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV data.
        technical_indicators_config (dict): A dictionary defining the technical
                                            indicators to calculate (from technical_indicators.yaml).

    Returns:
        pd.DataFrame: A new DataFrame with original OHLCV data and added technical indicator columns.
    """
    df_processed = df.copy()
    required_cols_for_ta = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Ensure required columns are numeric and handle potential NaNs before TA calculation
    for col in required_cols_for_ta:
        if col not in df_processed.columns:
            print(f"ERROR: Missing required column '{col}' for TA calculation. Cannot calculate TAs.")
            df_processed[col] = 0.0 # Assign 0.0 to missing required cols
            return df_processed.assign(**{k: np.nan for k in technical_indicators_config.keys()}) # Return with NaN TAs if critical base cols are missing
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # Drop rows where critical OHLCV data is missing, as TA-Lib can't operate on them
    # Only drop if the DataFrame is not empty, otherwise, it leads to issues.
    original_rows = len(df_processed)
    df_processed.dropna(subset=['High', 'Low', 'Close'], inplace=True)
    if df_processed.empty:
        print(f"WARNING: DataFrame became empty after dropping NaNs for TA calculation (from {original_rows} rows). Returning empty DataFrame for TAs.")
        # Create an empty DataFrame with expected TA columns filled with NaN or 0
        ta_cols = list(technical_indicators_config.keys())
        return pd.DataFrame(columns=required_cols_for_ta + ta_cols)


    if not TALIB_AVAILABLE:
        print("TA-Lib not available, skipping technical indicator calculation for all features.")
        # Return original OHLCV, with TA columns added and filled with 0s/NaNs as placeholders
        ta_cols = list(technical_indicators_config.keys())
        for col in ta_cols:
            if col not in df_processed.columns: # Only add if not already there (shouldn't be for TAs)
                df_processed[col] = 0.0 # Or np.nan if you prefer NaNs
        return df_processed[required_cols_for_ta + ta_cols].bfill().ffill().fillna(0) # Ensure no NaNs from OHLCV if this path is taken

    # Convert columns to numpy arrays for TA-Lib performance
    high_np = df_processed['High'].values.astype(float)
    low_np = df_processed['Low'].values.astype(float)
    close_np = df_processed['Close'].values.astype(float)
    open_np = df_processed['Open'].values.astype(float)
    volume_np = df_processed['Volume'].values.astype(float)

    indicator_calculators = get_indicator_calculators(high_np, low_np, close_np, open_np, volume_np)

    for feature_name, feature_config in technical_indicators_config.items(): # Changed from price_features_to_add.items()
        indicator_name = feature_config.get('indicator', feature_name)
        
        # Skip base features (Open, High, Low, Close, Volume) which are not calculated TAs
        if indicator_name in required_cols_for_ta:
            # We don't need to explicitly assign these as they are already in df_processed.
            # We just need to make sure they are included in the final column selection if desired.
            continue
            
        try:
            params = feature_config.get('params', {})
            calculator_func = indicator_calculators.get(indicator_name)

            if calculator_func:
                result = calculator_func(**params)
                
                # Handle indicators that return multiple outputs (e.g., MACD, BBANDS)
                if isinstance(result, tuple):
                    output_field = feature_config.get('output_field', 0)
                    if output_field < len(result):
                        df_processed[feature_name] = result[output_field]
                    else:
                        print(f"Warning: Output field {output_field} out of bounds for {indicator_name}. Assigning NaN.")
                        df_processed[feature_name] = np.nan
                else:
                    df_processed[feature_name] = result
            
            # Handle candlestick patterns which have a different function signature
            elif indicator_name.startswith('CDL'):
                if hasattr(talib, indicator_name):
                    pattern_func = getattr(talib, indicator_name)
                    df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np, **params)
                else:
                    # This specific warning should now ideally not be hit if TALIB_AVAILABLE is False already
                    print(f"Warning: TA-Lib function '{indicator_name}' not found for pattern. Assigning NaN.")
                    df_processed[feature_name] = np.nan
            else:
                # This branch should now only be hit if an indicator_name in config
                # is not a standard TA-Lib function AND not a CDL pattern.
                print(f"Warning: TA '{indicator_name}' not defined for calculation (or not found in TA-Lib functions). Assigning NaN.")
                df_processed[feature_name] = np.nan
        except Exception as e:
            print(f"Error calculating TA '{feature_name}' with indicator '{indicator_name}': {e}. Assigning NaN.")
            # If an error occurs during calculation, assign NaN to that feature
            df_processed[feature_name] = np.nan

    # After calculating all features, handle any remaining NaNs
    # This is crucial for environments that don't handle NaNs
    # Use bfill then ffill to propagate non-NaN values, then fill remaining with 0
    df_processed.bfill(inplace=True)
    df_processed.ffill(inplace=True)
    df_processed.fillna(0, inplace=True) # Fill any remaining NaNs (e.g., at very start of series) with 0

    # Ensure final columns order and inclusion.
    # The 'required_cols_for_ta' ensures OHLCV are always first, then TAs.
    final_columns = required_cols_for_ta + list(technical_indicators_config.keys())
    for col in final_columns:
        if col not in df_processed.columns:
            # This should ideally not happen if features are consistently added or original columns exist
            df_processed[col] = 0.0 # Default value if a feature somehow entirely failed or wasn't added

    # Select and reorder columns
    return df_processed[final_columns]
