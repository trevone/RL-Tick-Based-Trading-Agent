# src/data/custom_indicators.py
import numpy as np
from typing import Dict, Callable, Tuple

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

def calculate_envelopes(data: np.ndarray, timeperiod: int = 20, percentage: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Upper and Lower Envelopes for a given data series.
    """
    if not TALIB_AVAILABLE:
        nan_array = np.full(data.shape, np.nan)
        return nan_array, nan_array
        
    sma = talib.SMA(data, timeperiod=timeperiod)
    upper_envelope = sma + (sma * percentage)
    lower_envelope = sma - (sma * percentage)
    return upper_envelope, lower_envelope

def get_custom_indicator_calculators() -> Dict[str, Callable]:
    """
    Returns a dictionary mapping custom indicator names to their calculation functions.
    The main feature engineer is responsible for passing the correct data source (e.g., close, high).
    """
    return {
        'ENVELOPES': calculate_envelopes,
        # 'MY_CUSTOM_INDICATOR': calculate_my_indicator,
    }