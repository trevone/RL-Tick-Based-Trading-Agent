# live_trader.py
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timezone, timedelta
import os
import sys
import traceback
import math # For math.floor and ceil

# --- Import Binance API Client for actual trading ---
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_CLIENT_AVAILABLE = True
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False
    print("CRITICAL ERROR: python-binance library not found. Install with 'pip install python-binance' for live trading.")
    sys.exit(1)

# Assuming utils.py, base_env.py, custom_wrappers.py are in the same directory or accessible via PYTHONPATH
try:
    from utils import load_config, merge_configs, convert_to_native_types, resolve_model_path
    from utils import _calculate_technical_indicators # Import the TA calculation function directly
    from base_env import SimpleTradingEnv, DEFAULT_ENV_CONFIG
    # For agent loading, if using StableBaselines3:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    # Import DEFAULT_TRAIN_CONFIG from train_simple_agent to ensure consistent hashing logic
    from train_simple_agent import DEFAULT_TRAIN_CONFIG as TRAIN_SCRIPT_FULL_DEFAULT_CONFIG
    from custom_wrappers import FlattenAction

except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import necessary modules. Ensure utils.py, base_env.py, stable-baselines3, and custom_wrappers.py are accessible: {e}")
    print("Exiting live trader script.")
    sys.exit(1)

# --- Configuration & Defaults ---
DEFAULT_LIVE_TRADER_CONFIG = {
    "run_settings": {
        "log_level": "normal",
        "live_log_dir": "./logs/live_trading/",
        "model_path": None, # Explicit path to a trained model (.zip file). If None, script tries to derive.
        "refresh_interval_sec": 1, # How often to check for new observation and take action (dummy: 1 sec, adjust for tick)
        "data_processing_interval_ms": 100 # Process buffered ticks every X ms for environment step
    },
    "binance_websocket": {
        "uri": "wss://stream.binance.com:9443/ws", # Binance Spot market stream
        "symbol": "btcusdt@aggTrade",             # Aggregate Trade Stream for BTCUSDT
        "kline_symbol_stream_suffix": "@kline_", # Suffix for K-line stream (e.g., @kline_1h)
        "testnet_uri": "wss://testnet.binance.vision/ws", # Testnet URI
        "testnet_symbol": "btcusdt@aggTrade", # Testnet symbol for aggTrade
        "testnet_kline_symbol_stream_suffix": "@kline_", # Testnet suffix for K-line stream
    },
    "binance_api_client": { # New section for REST API settings
        "timeout_seconds": 10,
        "recv_window_ms": 5000
    },
    "binance_settings": { # Re-include basic binance_settings for testnet flag and api_keys
        "api_key": None,
        "api_secret": None,
        "testnet": False,
        "default_symbol": "BTCUSDT", # Needed for client init
        "historical_interval": "1h" # Needed for live kline stream subscription
    },
    "environment": DEFAULT_ENV_CONFIG, # Inherit default environment configuration
    # No hash_config_keys here as it's for training models, not defining the live script
}

# --- Global Variables for Live Operation ---
live_tick_buffer = deque()
historical_tick_data = pd.DataFrame(columns=['Price', 'Quantity', 'IsBuyerMaker', 'Timestamp'])
historical_tick_data.set_index('Timestamp', inplace=True)

# New global for real-time K-line data
# This DataFrame will be continuously updated by the K-line WebSocket listener
# It will store the most recent kline_window_size candles, including TAs
realtime_kline_data_with_ta = pd.DataFrame()

# Binance REST API Client (will be initialized in main)
binance_client = None
# Global to store symbol info for quantization
symbol_info = {}

# --- Helper Functions for Binance API Interaction ---
def get_symbol_info(client: Client, symbol: str, log_level: str) -> dict:
    """Fetches and returns symbol exchange information for quantization rules."""
    global symbol_info
    if symbol in symbol_info:
        return symbol_info[symbol]

    try:
        info = client.get_symbol_info(symbol)
        if info:
            symbol_info[symbol] = info
            if log_level != "none":
                print(f"Fetched symbol info for {symbol}.")
            return info
        else:
            if log_level != "none":
                print(f"Could not get symbol info for {symbol}.")
            return {}
    except BinanceAPIException as bae:
        print(f"Binance API Exception getting symbol info: {bae.code} - {bae.message}")
        return {}
    except Exception as e:
        print(f"Unexpected error getting symbol info: {e}")
        traceback.print_exc()
        return {}

def quantize_quantity(quantity: float, symbol_info: dict) -> float:
    """Quantizes a quantity according to Binance exchange rules."""
    if not symbol_info:
        return quantity

    for f in symbol_info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            step_size = float(f['stepSize'])
            min_qty = float(f['minQty'])
            max_qty = float(f['maxQty'])

            precision = int(round(-math.log10(step_size)))
            
            quantized_qty = math.floor(quantity / step_size) * step_size
            quantized_qty = max(min_qty, min(quantized_qty, max_qty))
            
            return float(f'{quantized_qty:.{precision}f}')
    return quantity

def quantize_price(price: float, symbol_info: dict) -> float:
    """Quantizes a price according to Binance exchange rules."""
    if not symbol_info:
        return price

    for f in symbol_info['filters']:
        if f['filterType'] == 'PRICE_FILTER':
            tick_size = float(f['tickSize'])
            precision = int(round(-math.log10(tick_size)))
            
            quantized_price = round(price / tick_size) * tick_size
            
            return float(f'{quantized_price:.{precision}f}')
    return price

# --- WebSocket Listener Functions ---

async def connect_and_listen_aggtrade_websocket(uri: str, symbol: str, log_level: str):
    """
    Connects to the Binance Aggregate Trade WebSocket and listens for trades.
    Puts received ticks into the live_tick_buffer.
    """
    full_uri = f"{uri}/{symbol}"
    if log_level != "none":
        print(f"Connecting to AggTrade WebSocket: {full_uri}")
    try:
        async with websockets.connect(full_uri) as ws:
            if log_level != "none":
                print("AggTrade WebSocket connection established.")
            while True:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    tick = {
                        'Price': float(data['p']),
                        'Quantity': float(data['q']),
                        'Timestamp': pd.to_datetime(data['T'], unit='ms', utc=True),
                        'IsBuyerMaker': data['m']
                    }
                    live_tick_buffer.append(tick)
                    if log_level == "detailed":
                        print(f"Received aggTrade: Price={tick['Price']:.4f}, Qty={tick['Quantity']:.4f}, Time={tick['Timestamp']}")

                except websockets.exceptions.ConnectionClosedOK:
                    print("AggTrade WebSocket connection closed cleanly.")
                    break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"AggTrade WebSocket connection closed with error: {e}. Attempting to reconnect in 5s.")
                    await asyncio.sleep(5)
                    break
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON message from AggTrade: {e}, Message: {message}")
                except Exception as e:
                    print(f"Unexpected error in AggTrade WebSocket receiver: {e}")
                    traceback.print_exc()
                    await asyncio.sleep(1)
    except Exception as e:
        print(f"Error connecting to AggTrade WebSocket: {e}. Retrying connection in 10s.")
        traceback.print_exc()
        await asyncio.sleep(10)

async def connect_and_listen_kline_websocket(uri: str, symbol: str, interval: str, kline_price_features: list, kline_window_size: int, log_level: str):
    """
    Connects to the Binance K-line WebSocket and builds/updates real-time K-line data with TAs.
    """
    global realtime_kline_data_with_ta
    full_uri = f"{uri}/{symbol}{interval}"
    if log_level != "none":
        print(f"Connecting to K-line WebSocket: {full_uri}")
    
    # Define columns for raw K-line data based on Binance API
    raw_kline_cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime']
    # Filter only relevant columns for TA calculation
    ta_input_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Initial fetch of historical K-lines to populate the window
    # This requires `binance_client` to be initialized.
    # We will assume `binance_client` is ready before this task starts.
    if binance_client:
        try:
            # Fetch a few more candles than window_size to ensure smooth start of TA calculation
            initial_klines_raw = binance_client.get_historical_klines(symbol, interval, limit=kline_window_size + 10)
            initial_df = pd.DataFrame(initial_klines_raw, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberofTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
            
            initial_df['OpenTime'] = pd.to_datetime(initial_df['OpenTime'], unit='ms', utc=True)
            initial_df.set_index('OpenTime', inplace=True)
            initial_df[ta_input_cols] = initial_df[ta_input_cols].astype(float)
            
            # Calculate TAs on initial historical data
            realtime_kline_data_with_ta = _calculate_technical_indicators(initial_df[ta_input_cols].copy(), kline_price_features)
            
            if log_level != "none":
                print(f"Initial K-line historical data (via REST) loaded for real-time window: {realtime_kline_data_with_ta.shape}")
            realtime_kline_data_with_ta = realtime_kline_data_with_ta.iloc[-kline_window_size:].copy()
            if log_level != "none":
                print(f"Real-time K-line window initialized with {len(realtime_kline_data_with_ta)} candles.")
        except Exception as e:
            print(f"ERROR: Could not fetch initial K-line historical data for live window: {e}. K-line observation might be delayed/incomplete.")
            traceback.print_exc()
            # If initial fetch fails, initialize with empty DataFrame to prevent errors
            realtime_kline_data_with_ta = pd.DataFrame(columns=kline_price_features)
            realtime_kline_data_with_ta.index.name = 'OpenTime' # Ensure index name for consistency
            if log_level != "none":
                print("Real-time K-line data initialized as empty.")

    try:
        async with websockets.connect(full_uri) as ws:
            if log_level != "none":
                print("K-line WebSocket connection established.")
            while True:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    # Binance K-line stream sends a 'k' object for the candle data
                    candle = data['k']
                    
                    # Only process if the candle is closed or if it's the current candle updating
                    # For live trading, you usually want to act on the *closed* candle for final values
                    # or the *most recent update* of the current open candle.
                    # Here, we'll process each update of the *current* (open) candle and add it as a new row
                    # or update the last row if the timestamp matches.
                    
                    open_time = pd.to_datetime(candle['t'], unit='ms', utc=True)
                    is_candle_closed = candle['x'] # True if candle is closed

                    # Create a temporary DataFrame for the new candle
                    new_candle_df = pd.DataFrame([{
                        'OpenTime': open_time,
                        'Open': float(candle['o']),
                        'High': float(candle['h']),
                        'Low': float(candle['l']),
                        'Close': float(candle['c']),
                        'Volume': float(candle['v']),
                        'CloseTime': pd.to_datetime(candle['T'], unit='ms', utc=True)
                    }]).set_index('OpenTime')
                    new_candle_df = new_candle_df[ta_input_cols] # Keep only TA input cols
                    
                    # Update global K-line data: If last candle is same time, update; else append
                    if not realtime_kline_data_with_ta.empty and realtime_kline_data_with_ta.index[-1] == open_time:
                        # Update the last candle (it's still forming)
                        temp_combined_df = pd.concat([realtime_kline_data_with_ta.iloc[:-1], new_candle_df])
                    else:
                        # Append new candle
                        temp_combined_df = pd.concat([realtime_kline_data_with_ta, new_candle_df])
                    
                    # Ensure only a reasonable amount of data is kept for TA calculation performance
                    temp_combined_df = temp_combined_df.iloc[-(kline_window_size + 20):].copy() # Keep enough history for TAs
                    
                    # Recalculate TAs on the updated window
                    # Pass the correct `kline_price_features` from environment config
                    global_kline_features_from_env_config = kline_price_features # Using the kline_price_features from main()
                    
                    df_with_new_tas = _calculate_technical_indicators(temp_combined_df, global_kline_features_from_env_config)
                    
                    # Keep only the window size for the observation
                    realtime_kline_data_with_ta = df_with_new_tas.iloc[-kline_window_size:].copy()

                    if log_level == "detailed":
                        status = "CLOSED" if is_candle_closed else "OPEN"
                        print(f"Received K-line ({interval}, {status}): {open_time} - C:{candle['c']:.4f}, H:{candle['h']:.4f}, L:{candle['l']:.4f}")
                        if not realtime_kline_data_with_ta.empty:
                            print(f"  Real-time K-line window end: {realtime_kline_data_with_ta.index[-1]}, SMA_10: {realtime_kline_data_with_ta['SMA_10'].iloc[-1]:.4f} (last)")


                except websockets.exceptions.ConnectionClosedOK:
                    print("K-line WebSocket connection closed cleanly.")
                    break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"K-line WebSocket connection closed with error: {e}. Attempting to reconnect in 5s.")
                    await asyncio.sleep(5)
                    break
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON message from K-line: {e}, Message: {message}")
                except Exception as e:
                    print(f"Unexpected error in K-line WebSocket receiver: {e}")
                    traceback.print_exc()
                    await asyncio.sleep(1)
    except Exception as e:
        print(f"Error connecting to K-line WebSocket: {e}. Retrying connection in 10s.")
        traceback.print_exc()
        await asyncio.sleep(10)


async def process_and_act(env_config: dict, model, run_config: dict, binance_api_client: Client):
    """
    Periodically processes buffered ticks, updates the environment's observation,
    gets an action from the agent, and attempts to execute trades.
    """
    global historical_tick_data, realtime_kline_data_with_ta, symbol_info # Add realtime_kline_data_with_ta to globals
    log_level = run_config["run_settings"]["log_level"]
    data_processing_interval_ms = run_config["run_settings"]["data_processing_interval_ms"] / 1000.0
    env_tick_window_size = env_config["tick_feature_window_size"]
    trade_symbol = run_config["binance_settings"]["default_symbol"] # e.g., "BTCUSDT"

    print("\n--- Starting Agent Action Loop ---")

    # Initialize environment with dummy data for initial observation space creation
    # The tick_df will be updated, kline_df_with_ta comes from the global realtime_kline_data_with_ta
    base_live_env_instance = SimpleTradingEnv( # Create base env first
        tick_df=historical_tick_data.copy(), # Initially empty or small
        kline_df_with_ta=realtime_kline_data_with_ta.copy(), # Initial K-line data for env structure
        config=env_config
    )
    # Apply the FlattenAction wrapper to the base environment
    live_env_instance_wrapped = FlattenAction(base_live_env_instance)
    # Wrap the result with Monitor
    live_env_instance = Monitor(live_env_instance_wrapped)
    
    # Do an initial reset to set up the environment's internal state (balance, position, etc.)
    current_observation, info = live_env_instance.reset()


    # Get symbol info once on startup for quantization rules
    if binance_api_client and not symbol_info.get(trade_symbol):
        symbol_info[trade_symbol] = get_symbol_info(binance_api_client, trade_symbol, log_level)
        if not symbol_info[trade_symbol]:
            print(f"WARNING: Could not retrieve symbol info for {trade_symbol}. Quantity/price quantization may be inaccurate.")

    while True:
        await asyncio.sleep(data_processing_interval_ms)

        # 1. Update historical_tick_data from live_tick_buffer
        new_ticks_list = []
        while live_tick_buffer:
            new_ticks_list.append(live_tick_buffer.popleft())
        
        if new_ticks_list:
            new_ticks_df = pd.DataFrame(new_ticks_list).set_index('Timestamp').sort_index()
            historical_tick_data = pd.concat([historical_tick_data, new_ticks_df])
            
            historical_tick_data = historical_tick_data.iloc[-(env_tick_window_size + 100):]
            historical_tick_data = historical_tick_data[~historical_tick_data.index.duplicated(keep='last')]
            historical_tick_data.sort_index(inplace=True)

            if log_level == "detailed":
                print(f"Processed {len(new_ticks_list)} new ticks. Historical tick data size: {len(historical_tick_data)}")
                if not historical_tick_data.empty:
                    print(f"  Historical tick data range: {historical_tick_data.index.min()} to {historical_tick_data.index.max()}")
        else:
            if log_level == "detailed":
                print(f"No new ticks to process. Tick buffer empty.")
            continue


        # 2. Generate current observation for the agent
        # Ensure sufficient tick data for observation window
        if len(historical_tick_data) < live_env_instance.env.tick_feature_window_size:
            if log_level != "none":
                print("WARNING: Not enough historical tick data for full observation window. Padding will occur. Skipping action.")
            continue

        # Ensure sufficient K-line data for observation window
        if len(realtime_kline_data_with_ta) < live_env_instance.env.kline_window_size:
            if log_level != "none":
                print("WARNING: Not enough real-time K-line data for full observation window. Padding will occur. Skipping action.")
            continue


        # Create a temporary env instance to generate the observation with the latest data.
        # This temp env will use the *real-time updated* historical_tick_data and realtime_kline_data_with_ta.
        temp_base_env_for_obs = SimpleTradingEnv(
            tick_df=historical_tick_data.copy(), # Pass updated tick data
            kline_df_with_ta=realtime_kline_data_with_ta.copy(), # Pass updated k-line data
            config=env_config
        )
        temp_env_for_obs = FlattenAction(temp_base_env_for_obs)
        
        # Set its current_step to the last available tick index for observation generation
        temp_env_for_obs.env.current_step = temp_env_for_obs.env.start_step + len(historical_tick_data) - 1
        if temp_env_for_obs.env.current_step < temp_env_for_obs.env.start_step:
            temp_env_for_obs.env.current_step = temp_env_for_obs.env.start_step

        current_observation, info = temp_env_for_obs._get_observation(), temp_env_for_obs.env._get_info()

        # --- Agent Action ---
        if model is not None and current_observation is not None and info["current_tick_price"] > 0:
            obs_for_model = np.array(current_observation).reshape(1, -1)
            
            action_array, _states = model.predict(obs_for_model, deterministic=True)
            
            discrete_action = int(np.round(action_array[0]))
            profit_target_param = action_array[1]
            
            action_desc = live_env_instance.env.ACTION_MAP.get(discrete_action, "UNKNOWN")
            print(f"\n--- Agent Decision ({datetime.now(timezone.utc).isoformat()}) ---")
            print(f"  Current Tick Price: {info['current_tick_price']:.4f}")
            print(f"  Portfolio Equity: {info['equity']:.2f}, Balance: {info['current_balance']:.2f}")
            print(f"  Position Open: {info['position_open']}, Entry: {info['entry_price']:.4f}, Volume: {info['position_volume']:.4f}")
            print(f"  Agent Action: {action_desc}, Profit Target: {profit_target_param:.4f}")

            current_price = info['current_tick_price']
            
            # --- REAL TRADE EXECUTION LOGIC (BINANCE TESTNET/MAINNET) ---
            
            current_symbol_info = symbol_info.get(trade_symbol, {})

            base_trade_amount_usd = live_env_instance.env.initial_balance * live_env_instance.env.base_trade_amount_ratio
            desired_quantity = base_trade_amount_usd / (current_price + 1e-9)

            quantity_to_trade_buy = quantize_quantity(desired_quantity, current_symbol_info)
            
            if discrete_action == 1: # BUY
                if not info['position_open']:
                    if binance_api_client and quantity_to_trade_buy > 0:
                        print(f"  ATTEMPTING REAL BUY ORDER on Binance: {quantity_to_trade_buy:.6f} {trade_symbol[:-4]} @ MARKET price")
                        try:
                            # --- ACTUAL BINANCE API CALL (UNCOMMENT TO ENABLE REAL TRADES) ---
                            order = binance_api_client.order_market_buy(symbol=trade_symbol, quantity=quantity_to_trade_buy)
                            print(f"  Binance BUY Order Sent: ID={order.get('orderId')}, Status={order.get('status')}")
                            # --- END ACTUAL API CALL ---

                            # Simulated fill logic, update base env's internal state
                            simulated_cost = quantity_to_trade_buy * current_price * (1 + live_env_instance.env.commission_pct)
                            if live_env_instance.env.current_balance >= simulated_cost:
                                live_env_instance.env.current_balance -= simulated_cost
                                live_env_instance.env.position_open = True
                                live_env_instance.env.entry_price = current_price
                                live_env_instance.env.position_volume = quantity_to_trade_buy
                                live_env_instance.env.current_desired_profit_target = profit_target_param
                                print(f"  REAL BUY (Simulated Fill): {quantity_to_trade_buy:.6f} @ {current_price:.4f}. Balance: {live_env_instance.env.current_balance:.2f}")
                            else:
                                print("  REAL BUY (Simulated Fill Fail): Insufficient internal balance for trade.")
                        except BinanceAPIException as bae:
                            print(f"  ERROR sending BUY order: {bae.code} - {bae.message}")
                        except Exception as e:
                            print(f"  UNEXPECTED ERROR sending BUY order: {e}")
                            traceback.print_exc()
                    else:
                        print("  Binance API client not initialized or quantity <= 0. Skipping real BUY trade.")
                else:
                    print("  Cannot BUY: Position already open.")

            elif discrete_action == 2: # SELL
                if info['position_open']:
                    quantity_to_trade_sell = live_env_instance.env.position_volume
                    if binance_api_client and quantity_to_trade_sell > 0:
                        print(f"  ATTEMPTING REAL SELL ORDER on Binance: {quantity_to_trade_sell:.6f} {trade_symbol[:-4]} @ MARKET price")
                        try:
                            # --- ACTUAL BINANCE API CALL (UNCOMMENT TO ENABLE REAL TRADES) ---
                            order = binance_api_client.order_market_sell(symbol=trade_symbol, quantity=quantity_to_trade_sell)
                            print(f"  Binance SELL Order Sent: ID={order.get('orderId')}, Status={order.get('status')}")
                            # --- END ACTUAL API CALL ---

                            # Simulated fill logic, update base env's internal state
                            simulated_revenue = quantity_to_trade_sell * current_price * (1 - live_env_instance.env.commission_pct)
                            simulated_cost_basis = live_env_instance.env.entry_price * quantity_to_trade_sell
                            simulated_pnl = simulated_revenue - simulated_cost_basis

                            live_env_instance.env.current_balance += simulated_revenue
                            live_env_instance.env.position_open = False; live_env_instance.env.entry_price = 0.0; live_env_instance.env.position_volume = 0.0; live_env_instance.env.current_desired_profit_target = 0.0
                            print(f"  REAL SELL (Simulated Fill): {quantity_to_trade_sell:.6f} @ {current_price:.4f}. PnL: {simulated_pnl:.2f}. Bal: {live_env_instance.env.current_balance:.2f}")
                        except BinanceAPIException as bae:
                            print(f"  ERROR sending SELL order: {bae.code} - {bae.message}")
                        except Exception as e:
                            print(f"  UNEXPECTED ERROR sending SELL order: {e}")
                            traceback.print_exc()
                    else:
                        print("  Binance API client not initialized or quantity <= 0. Skipping real SELL trade.")
                else:
                    print("  Cannot SELL: No position to sell.")
            
            else: # HOLD
                if log_level == "detailed":
                    print("  HOLD.")

            current_equity_live = live_env_instance.env.current_balance + (live_env_instance.env.position_volume * current_price if live_env_instance.env.position_open else 0)
            if current_equity_live < live_env_instance.env.catastrophic_loss_limit:
                print(f"  CRITICAL: Catastrophic loss detected! Equity dropped to {current_equity_live:.2f}")
                if live_env_instance.env.position_open:
                    if binance_api_client:
                        print(f"  ATTEMPTING EMERGENCY LIQUIDATION on Binance for {live_env_instance.env.position_volume:.6f} {trade_symbol[:-4]}...")
                        try:
                            # --- ACTUAL BINANCE API CALL (UNCOMMENT TO ENABLE REAL TRADES) ---
                            order = binance_api_client.order_market_sell(symbol=trade_symbol, quantity=live_env_instance.env.position_volume)
                            print("  Emergency liquidation order sent.")
                            # --- END ACTUAL API CALL ---
                        except Exception as e:
                            print(f"  ERROR during emergency liquidation: {e}")
                
                live_env_instance.env.current_balance = live_env_instance.env.initial_balance
                live_env_instance.env.position_open = False
                break

        else:
            if model is None: print("  Agent model not loaded. Skipping action.")
            if current_observation is None: print("  Observation not available. Skipping action.")
            if info["current_tick_price"] == 0.0: print("  Current tick price is 0.0. Skipping action.")


# --- Main Execution Block ---
async def main():
    global binance_client, initial_dummy_kline_data_for_env, realtime_kline_data_with_ta # Add realtime_kline_data_with_ta to globals

    try:
        loaded_config = load_config("config.yaml")
        config = merge_configs(DEFAULT_LIVE_TRADER_CONFIG, loaded_config)
    except Exception as e:
        print(f"Error loading or merging configuration: {e}. Using default live trader config.")
        config = DEFAULT_LIVE_TRADER_CONFIG.copy()

    run_settings = config["run_settings"]
    binance_ws_config = config["binance_websocket"]
    binance_api_config = config["binance_api_client"]
    binance_settings = config["binance_settings"]
    env_config = config["environment"]
    log_level = run_settings.get("log_level", "normal")

    is_testnet = binance_settings.get("testnet", False)
    # Determine WebSocket URI and symbol for AGGREGATE TRADES
    websocket_uri_aggtrade = binance_ws_config["testnet_uri"] if is_testnet else binance_ws_config["uri"]
    websocket_symbol_aggtrade = binance_ws_config["testnet_symbol"] if is_testnet else binance_ws_config["symbol"]
    
    # Determine WebSocket URI and symbol for K-LINES
    websocket_uri_kline = binance_ws_config["testnet_uri"] if is_testnet else binance_ws_config["uri"]
    kline_stream_suffix = binance_ws_config["testnet_kline_symbol_stream_suffix"] if is_testnet else binance_ws_config["kline_symbol_stream_suffix"]
    websocket_symbol_kline = f"{binance_settings['default_symbol'].lower()}{kline_stream_suffix}{binance_settings['historical_interval']}" # e.g., btcusdt@kline_1h


    api_key = binance_settings.get("api_key")
    api_secret = binance_settings.get("api_secret")

    print(f"--- Starting Live Trading Script (Log Level: {log_level}) ---")
    print(f"Operating on Binance {'TESTNET' if is_testnet else 'MAINNET'}")
    print(f"Targeting AggTrade WebSocket: {websocket_uri_aggtrade} for {websocket_symbol_aggtrade}")
    print(f"Targeting K-line WebSocket: {websocket_uri_kline} for {websocket_symbol_kline}")


    # --- Initialize Binance REST API Client ---
    if BINANCE_CLIENT_AVAILABLE and api_key and api_secret:
        try:
            binance_client = Client(
                api_key, api_secret,
                testnet=is_testnet,
                tld='us' if 'USDT' in binance_settings.get('default_symbol') else 'com', # Simple heuristic for TLD, adjust if needed
                requests_params={'timeout': binance_api_config.get("timeout_seconds", 10)}
            )
            if is_testnet:
                 binance_client.API_URL = 'https://testnet.binance.vision/api'
            else:
                 binance_client.API_URL = 'https://api.binance.com/api' # Spot API URL
            
            if binance_api_config.get("recv_window_ms"):
                binance_client.options['recvWindow'] = binance_api_config["recv_window_ms"]
            print("Binance REST API client initialized.")
            
            try:
                account_info = binance_client.get_account()
                print(f"Connected account type: {account_info.get('accountType', 'N/A')}")
            except Exception as e:
                print(f"WARNING: Could not fetch account info with API client: {e}. Check API keys/permissions/network.")

        except Exception as e:
            print(f"ERROR: Failed to initialize Binance REST API client: {e}. Live trading will be simulated only.")
            traceback.print_exc()
            binance_client = None
    else:
        print("Binance client not available (library missing or API keys not provided). Live trading will be simulated only.")
        binance_client = None

    # --- Prepare initial dummy K-line data for env observation space ---
    # This will be overridden by real-time data but sets up structure.
    kline_window_size = env_config.get("kline_window_size", DEFAULT_ENV_CONFIG["kline_window_size"])
    kline_price_features_for_env = env_config.get("kline_price_features", DEFAULT_ENV_CONFIG["kline_price_features"])
    
    # Initialize the global realtime_kline_data_with_ta with an empty DataFrame with correct columns
    # This will be populated by connect_and_listen_kline_websocket.
    realtime_kline_data_with_ta = pd.DataFrame(columns=kline_price_features_for_env)
    realtime_kline_data_with_ta.index.name = 'OpenTime' # Ensure index name for consistency
    
    # NOTE: The initial_dummy_kline_data_for_env here is no longer the live source.
    # It is just used to create the dummy env for model loading if needed.
    dummy_kline_rows_for_model_load = max(kline_window_size + 5, 20)
    dummy_kline_index_for_model_load = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=dummy_kline_rows_for_model_load, freq='1H')
    initial_dummy_kline_data_for_env = pd.DataFrame(np.random.rand(dummy_kline_rows_for_model_load, len(kline_price_features_for_env)) * 100 + 10000,
                                            index=dummy_kline_index_for_model_load,
                                            columns=kline_price_features_for_env)
    initial_dummy_kline_data_for_env.fillna(method='bfill', inplace=True)
    initial_dummy_kline_data_for_env.fillna(method='ffill', inplace=True)
    initial_dummy_kline_data_for_env.fillna(0, inplace=True)
    print(f"Initial dummy K-line data prepared for model loading env: {initial_dummy_kline_data_for_env.shape}")


    # --- Load Agent Model ---
    model = None
    model_path_from_config = run_settings.get("model_path")
    
    resolved_model_path = resolve_model_path(
        eval_specific_config=run_settings,
        full_yaml_config=loaded_config,
        train_script_fallback_config=TRAIN_SCRIPT_FULL_DEFAULT_CONFIG,
        env_script_fallback_config=DEFAULT_ENV_CONFIG,
        log_level=log_level
    )

    if resolved_model_path:
        try:
            # Create a dummy environment for model loading (StableBaselines3 needs it for obs/action space validation)
            dummy_base_env_for_model_load = SimpleTradingEnv(
                tick_df=pd.DataFrame(columns=['Price', 'Quantity', 'IsBuyerMaker'], index=pd.to_datetime([], utc=True)),
                kline_df_with_ta=initial_dummy_kline_data_for_env.copy(), # Use the dummy kline data for structure
                config=env_config
            )
            dummy_env_for_model_load = FlattenAction(dummy_base_env_for_model_load)
            
            model = PPO.load(resolved_model_path, env=dummy_env_for_model_load)
            print(f"Agent model loaded successfully from: {resolved_model_path}")
        except Exception as e:
            print(f"ERROR: Could not load agent model from {resolved_model_path}: {e}")
            traceback.print_exc()
            model = None
    else:
        print("WARNING: Agent model could not be resolved or found. Live trading will operate without agent actions.")
        print("Ensure 'model_path' in config.yaml points to a valid trained model .zip file, or check auto-resolution logic.")
        
    # --- Start Async Tasks ---
    # WebSocket Listener for Aggregate Trades (ticks)
    aggtrade_ws_task = asyncio.create_task(connect_and_listen_aggtrade_websocket(websocket_uri_aggtrade, websocket_symbol_aggtrade, log_level))

    # WebSocket Listener for K-lines
    # Pass kline_price_features for TA calculation, and kline_window_size for history management
    kline_ws_task = asyncio.create_task(connect_and_listen_kline_websocket(
        websocket_uri_kline,
        binance_settings['default_symbol'],
        binance_settings['historical_interval'],
        env_config['kline_price_features'], # Pass the configured features for TA calculation
        env_config['kline_window_size'],
        log_level
    ))

    # Agent Processing and Action task
    # This task now uses the globally updated realtime_kline_data_with_ta
    action_task = asyncio.create_task(process_and_act(env_config, model, config, binance_client))

    await asyncio.gather(aggtrade_ws_task, kline_ws_task, action_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nLive trading script interrupted by user. Shutting down...")
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        traceback.print_exc()