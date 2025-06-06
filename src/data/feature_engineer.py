# src/data/feature_engineer.py
import pandas as pd
import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: TA-Lib not found. Technical indicators will not be calculated. "
          "Please install with 'pip install TA-Lib' for full functionality.")

def calculate_technical_indicators(df: pd.DataFrame, price_features_to_add: list) -> pd.DataFrame:
    df_processed = df.copy()
    required_cols_for_ta = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols_for_ta:
        if col not in df_processed.columns:
            print(f"ERROR: Missing required column '{col}' for TA calculation. Cannot calculate TAs.")
            df_processed[col] = np.nan
            df_processed.fillna(0, inplace=True)
            return df_processed
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed.dropna(subset=['High', 'Low', 'Close'], inplace=True)
    if df_processed.empty:
        print("WARNING: DataFrame became empty after dropping NaNs for TA calculation.")
        return df_processed

    if not TALIB_AVAILABLE:
        print("TA-Lib not available, skipping technical indicator calculation.")
        final_df = df_processed[[col for col in required_cols_for_ta if col in df_processed.columns]].copy()
        return final_df.bfill().ffill().fillna(0)

    high_np = df_processed['High'].values.astype(float)
    low_np = df_processed['Low'].values.astype(float)
    close_np = df_processed['Close'].values.astype(float)
    open_np = df_processed['Open'].values.astype(float)
    volume_np = df_processed['Volume'].values.astype(float)

    for feature_name in price_features_to_add:
        if feature_name in required_cols_for_ta:
            continue
        try:
            if feature_name.startswith('SMA_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.SMA(close_np, timeperiod=timeperiod)
            elif feature_name.startswith('EMA_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.EMA(close_np, timeperiod=timeperiod)
            elif feature_name.startswith('RSI_'):
                timeperiod = int(feature_name.split('_')[1])
                df_processed[feature_name] = talib.RSI(close_np, timeperiod=timeperiod)
            elif feature_name == 'MACD':
                macd, macdsignal, macdhist = talib.MACD(close_np, fastperiod=12, slowperiod=26, signalperiod=9)
                df_processed['MACD'] = macd
            elif feature_name == 'ADX':
                df_processed['ADX'] = talib.ADX(high_np, low_np, close_np, timeperiod=14)
            elif feature_name == 'STOCH_K':
                stoch_k, stoch_d = talib.STOCH(high_np, low_np, close_np, fastk_period=5, slowk_period=3, slowd_period=3)
                df_processed['STOCH_K'] = stoch_k
            elif feature_name == 'ATR':
                df_processed['ATR'] = talib.ATR(high_np, low_np, close_np, timeperiod=14)
            elif feature_name == 'BBANDS_Upper':
                upper, middle, lower = talib.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                df_processed['BBANDS_Upper'] = upper
            elif feature_name == 'AD':
                df_processed['AD'] = talib.AD(high_np, low_np, close_np, volume_np)
            elif feature_name == 'OBV':
                df_processed['OBV'] = talib.OBV(close_np, volume_np)
            elif feature_name.startswith('CDL'):
                if hasattr(talib, feature_name):
                    pattern_func = getattr(talib, feature_name)
                    if feature_name in ['CDLMORNINGSTAR', 'CDLEVENINGSTAR']:
                         df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np, penetration=0)
                    else:
                         df_processed[feature_name] = pattern_func(open_np, high_np, low_np, close_np)
                else:
                    print(f"Warning: TA-Lib function '{feature_name}' not found. Assigning NaN.")
                    df_processed[feature_name] = np.nan
            else:
                print(f"Warning: TA '{feature_name}' not defined for calculation. Assigning NaN.")
                df_processed[feature_name] = np.nan
        except Exception as e:
            print(f"Error calculating TA '{feature_name}': {e}. Assigning NaN.")
            df_processed[feature_name] = np.nan

    df_processed.bfill(inplace=True)
    df_processed.ffill(inplace=True)
    df_processed.fillna(0, inplace=True)

    final_columns = [col for col in required_cols_for_ta if col in df_processed.columns]
    for feature in price_features_to_add:
        if feature not in final_columns:
            if feature not in df_processed.columns:
                df_processed[feature] = 0.0
            final_columns.append(feature)

    return df_processed[final_columns]