
# RL Tick-Based Trading Agent

This project implements a Reinforcement Learning (RL) agent for algorithmic trading, designed to operate on real-time tick and K-line market data using the Binance exchange API. The agent is trained using the PPO (Proximal Policy Optimization) algorithm from Stable Baselines3 within a custom Gymnasium environment.

## Table of Contents

1.  [Features](https://www.google.com/search?q=%23features)
2.  [Project Structure](https://www.google.com/search?q=%23project-structure)
3.  [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
      * [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      * [Environment Setup](https://www.google.com/search?q=%23environment-setup)
      * [Binance API Keys](https://www.google.com/search?q=%23binance-api-keys)
      * [TA-Lib Installation](https://www.google.com/search?q=%23ta-lib-installation)
4.  [Configuration (`config.yaml`)](https://www.google.com/search?q=%23configuration-configyaml)
5.  [Usage](https://www.google.com/search?q=%23usage)
      * [1. Data Acquisition and Caching](https://www.google.com/search?q=%231-data-acquisition-and-caching)
      * [2. Agent Training](https://www.google.com/search?q=%232-agent-training)
      * [3. Hyperparameter Optimization](https://www.google.com/search?q=%233-hyperparameter-optimization)
      * [4. Model Evaluation](https://www.google.com/search?q=%234-model-evaluation)
      * [5. Live Trading (Testnet/Mainnet)](https://www.google.com/search?q=%235-live-trading-testnetmainnet)
6.  [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)
7.  [Contributing](https://www.google.com/search?q=%23contributing)
8.  [License](https://www.google.com/search?q=%23license)

## Features

  * **Custom Gymnasium Trading Environment (`base_env.py`)**: A tick-level trading environment simulating Binance market dynamics, commissions, and portfolio management.
  * **Observation Space**: Integrates granular tick data (Price, Quantity), comprehensive K-line data (OHLCV, SMAs, RSIs, ATR, MACD), and candlestick patterns (Doji, Hammer, Engulfing, etc.) for a rich market observation.
  * **Action Space**: Supports both discrete trading actions (Hold, Buy, Sell) and a continuous profit target parameter for strategic exits.
  * **Stable Baselines3 Integration**: Uses the PPO algorithm for agent training, leveraging its robust implementations and utilities (e.g., `EvalCallback`, `Monitor`).
  * **Action Space Wrapper (`custom_wrappers.py`)**: Includes a custom `FlattenAction` wrapper to convert the environment's `Tuple` action space into a `Box` space, making it compatible with standard Stable Baselines3 algorithms like PPO.
  * **Binance Data Fetching (`utils.py`)**: Fetches historical K-line and aggregate trade (tick) data directly from Binance (Mainnet or Testnet) using `python-binance`.
  * **Data Caching**: Implements smart caching of fetched data to local Parquet files, significantly speeding up subsequent runs and reducing API calls.
  * **Technical Analysis (TA-Lib)**: Calculates various technical indicators and detects candlestick patterns using the optimized `TA-Lib` library. Supports dynamic period configuration for indicators like SMA, EMA, and RSI.
  * **Comprehensive Configuration (`config.yaml`)**: All environment, agent, data, and run-specific parameters are managed via a central YAML configuration file for easy customization and reproducibility.
  * **Configuration Hashing**: Training runs are uniquely identified by a hash of their core configuration parameters, ensuring organized logging and easy model retrieval.
  * **Model Evaluation Script (`evaluate_agent.py`)**: A dedicated script to evaluate a trained agent on unseen historical data, providing performance summaries, trade history, and interactive plots of price, equity, and trade signals.
  * **Live Trading Script (`live_trader.py`)**: Connects to Binance WebSockets for real-time tick data, feeds observations to the agent, and can be configured to execute actual market trades (on Testnet or Mainnet) with quantity quantization.
  * **Hyperparameter Optimization (`optimize_hyperparameters.py`)**: Integrates Optuna for automated hyperparameter tuning, maximizing agent performance by exploring different parameter combinations.

## Project Structure

```
.
├── config.yaml                     # Main configuration file for all scripts
├── base_env.py                     # Custom Gymnasium trading environment definition
├── utils.py                        # Utility functions for data fetching, TA calculation, config handling, etc.
├── custom_wrappers.py              # Custom Gymnasium environment wrappers (e.g., FlattenAction)
├── train_simple_agent.py           # Script for training the RL agent
├── evaluate_agent.py               # Script for evaluating a trained agent with plots
├── live_trader.py                  # Script for live (or simulated live) trading
├── optimize_hyperparameters.py     # Script for hyperparameter optimization using Optuna
├── README.md                       # This file
├── logs/                           # Directory for training, evaluation, and live trading logs
│   ├── ppo_trading/                # Training run logs (hashed subdirectories)
│   ├── evaluation_runs/            # Evaluation run logs (timestamped subdirectories)
│   └── live_trading/               # Live trading session logs
└── binance_data_cache/             # Cached historical K-line and tick data
    ├── bn_klines_...parquet
    └── bn_aggtrades_...parquet
# Additional files might include:
# ├── optuna_study.db               # SQLite database for Optuna study persistence
# └── check_tick_cache.py           # (Optional) Script to verify integrity of cached tick files
```

## Setup and Installation

### Prerequisites

  * Python 3.8+ (recommended using Anaconda/Miniconda)
  * Access to Binance API keys

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n rl_trading_env python=3.9 # Or your preferred Python version
    conda activate rl_trading_env
    ```
3.  **Install core dependencies:**
    ```bash
    pip install pandas numpy gymnasium stable-baselines3[extra] python-binance PyYAML matplotlib optuna
    # For plotting optimization results (optional)
    pip install plotly
    ```

### Binance API Keys

You need API keys from Binance to fetch historical data and (optionally) execute live trades.

1.  **Generate API Keys:**

      * **Mainnet:** Log in to your Binance account, go to "API Management" (or similar section) to create new API keys. Grant appropriate permissions (e.g., "Read Info", "Spot Trading").
      * **Testnet:** For testing live trading safely, create separate API keys on the Binance Testnet: `https://testnet.binance.vision/` (URL may vary, search for "Binance Testnet"). These keys are distinct from Mainnet keys.

2.  **Configure `config.yaml`:**
    Open `config.yaml` and update the `binance_settings` section with your keys:

    ```yaml
    binance_settings:
      api_key: YOUR_API_KEY_HERE
      api_secret: YOUR_API_SECRET_HERE
      testnet: true # Set to 'true' for Testnet, 'false' for Mainnet
    ```

    It's **highly recommended** to set these as environment variables (e.g., `BINANCE_API_KEY`, `BINANCE_API_SECRET`) and then set `api_key: null` and `api_secret: null` in `config.yaml`. The scripts are configured to read from environment variables if the config values are `None`.

      * **Example (Windows Command Prompt):**
        ```cmd
        set BINANCE_API_KEY="YOUR_KEY"
        set BINANCE_API_SECRET="YOUR_SECRET"
        ```
      * **Example (Linux/macOS Bash):**
        ```bash
        export BINANCE_API_KEY="YOUR_KEY"
        export BINANCE_API_SECRET="YOUR_SECRET"
        ```

### TA-Lib Installation

`TA-Lib` is a high-performance library for technical analysis. Its installation requires a C library.

**For Windows:**

1.  Download `ta-lib-0.4.0-msvc.zip` from [https://ta-lib.org/](https://ta-lib.org/) (Download section).
2.  Unzip the `ta-lib` folder directly to `C:\` (so you have `C:\ta-lib`).
3.  Activate your Conda environment: `conda activate rl_trading_env`
4.  Install the Python wrapper: `pip install TA-Lib`
      * If `pip install TA-Lib` fails, download the `.whl` file matching your Python version from [https://www.lfd.uci.edu/\~gohlke/pythonlibs/\#ta-lib](https://www.google.com/search?q=https://www.lfd.uci.edu/~gohlke/pythonlibs/%23ta-lib) and install via `pip install path\to\your\downloaded_file.whl`.

**For macOS/Linux:**

1.  **macOS (using Homebrew):**
    ```bash
    brew install ta-lib
    ```
2.  **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt-get update
    sudo apt-get install build-essential python3-dev
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure
    make
    sudo make install
    ```
3.  Activate your environment and install Python wrapper:
    ```bash
    conda activate rl_trading_env # or source venv/bin/activate
    pip install TA-Lib
    ```

## Configuration (`config.yaml`)

The `config.yaml` file is the central point for all project settings. Review and adjust it according to your needs:

  * **`environment`**: Defines the features included in the observation space (`kline_price_features`, `tick_features_to_use`, `kline_window_size`, `tick_feature_window_size`), trading rules (`initial_balance`, `commission_pct`), and reward structure. Enable desired TA indicators and candlestick patterns by uncommenting them in `kline_price_features`.
  * **`ppo_params`**: Core hyperparameters for the PPO RL agent (learning rate, `n_steps`, `gamma`, etc.). These are often targeted for optimization.
  * **`binance_settings`**: Specifies the default symbol, historical data date ranges (`start_date_kline_data`, `end_date_tick_data`), cache directory, and API keys/testnet flag.
  * **`binance_websocket`**: Defines URIs and symbols for real-time market data streams (`btcusdt@aggTrade`).
  * **`binance_api_client`**: Contains settings for direct REST API calls for order placement (`timeout_seconds`, `recv_window_ms`).
  * **`evaluation_data`**: Sets specific date ranges for historical data used during model evaluation.
  * **`hash_config_keys`**: **Crucial for reproducibility and model organization**. This section defines which parameters, when changed, will result in a new unique hash for a training run. Ensure any features you modify for an experiment are listed here.

## Usage

### 1\. Data Acquisition and Caching

The `train_simple_agent.py` script will automatically fetch and cache historical K-line and tick data based on the dates specified in `config.yaml` (`binance_settings` section).

  * **Initial Run**: The first time you run a script that requires data for a new period or with new features, it will download the data, which can take significant time for tick data.
  * **Subsequent Runs**: After the initial download, the data will be loaded directly from the local cache (`./binance_data_cache/`), making subsequent runs much faster.

### 2\. Agent Training

To train your RL agent:

1.  **Configure `config.yaml`**: Adjust parameters in `environment` and `ppo_params` as desired. Set historical data ranges in `binance_settings`.
2.  **Run the training script**:
    ```bash
    python train_simple_agent.py
    ```
      * Training logs, TensorBoard files, and the trained model (`trained_model_final.zip` and `best_model/best_model.zip`) will be saved in a unique subdirectory under `logs/ppo_trading/` named after the configuration hash and model name.
      * You can monitor training progress using TensorBoard: `tensorboard --logdir logs/ppo_trading/`

### 3\. Hyperparameter Optimization

To find optimal hyperparameters using Optuna:

1.  **Configure `optimize_hyperparameters.py`**:
      * Adjust `OPTIMIZATION_CONFIG` for number of trials, sampler, pruner, and the SQLite database URL.
      * Define the `HYPERPARAMETER_SEARCH_SPACE` with ranges for the hyperparameters you want to optimize (e.g., `learning_rate`, `n_steps`, `gamma`).
      * Note that `total_timesteps` is typically reduced for faster optimization trials.
2.  **Run the optimization script**:
    ```bash
    python optimize_hyperparameters.py
    ```
      * Optuna will run multiple training trials, saving progress to `optuna_study.db`.
      * Upon completion, it will print the best found hyperparameters and their corresponding metric value. It will also save these to `best_hyperparameters.json`.
      * If `plotly` is installed, it can show interactive plots of the optimization history.

### 4\. Model Evaluation

To evaluate a trained model on unseen data:

1.  **Configure `config.yaml`**:
      * In the `evaluation_data` section, specify date ranges for evaluation data that are *different* from your training data.
      * In `run_settings`, set `model_path` to the path of your trained model (`.zip` file) you wish to evaluate (e.g., `logs/ppo_trading/<YOUR_RUN_ID>/best_model/best_model.zip`).
      * Adjust `n_evaluation_episodes` as needed.
2.  **Run the evaluation script**:
    ```bash
    python evaluate_agent.py
    ```
      * The script will fetch evaluation data, load the specified model, run evaluation episodes, and print a summary of performance metrics (total reward, final equity, profit/loss percentage).
      * It will generate and save a performance chart (`<eval_run_id>_performance_chart.png`) in `logs/evaluation_runs/`, visualizing price, account equity, and trade entry/exit points.

### 5\. Live Trading (Testnet/Mainnet)

The `live_trader.py` script allows real-time market data consumption and agent-driven trade execution.

**WARNING: Live trading involves real money and significant financial risk. Always test thoroughly on Testnet before deploying to Mainnet.**

1.  **Configure `config.yaml`**:
      * Set `binance_settings.testnet: true` for Testnet trading, or `false` for Mainnet.
      * **Crucially, provide your corresponding Binance API keys** in `binance_settings.api_key` and `api_secret`. (Remember: Testnet keys are different from Mainnet keys).
      * Set `run_settings.model_path` to the `.zip` file of the trained model you want to use.
2.  **Enable Actual Trading (Modify `live_trader.py`)**:
    By default, trade execution is simulated. To enable real trades:
      * Open `live_trader.py`.
      * Locate the `--- ACTUAL BINANCE API CALL (UNCOMMENT TO ENABLE REAL TRADES) ---` sections within the `process_and_act` function (for `order_market_buy`, `order_market_sell`, and emergency liquidation).
      * **Uncomment** the `order = binance_api_client.order_market_buy(...)` and `order = binance_api_client.order_market_sell(...)` lines.
      * Ensure the `quantity` parameter passed to these calls is correctly quantized based on Binance's rules (the script includes `quantize_quantity` helpers).
3.  **Run the live trader**:
    ```bash
    python live_trader.py
    ```
      * The script will connect to the Binance WebSocket, start fetching real-time ticks, and when the agent makes a decision, it will attempt to send real orders to your configured Binance account (Testnet or Mainnet).
      * Console output will show received ticks, agent decisions, and the status of order attempts.

## Troubleshooting

  * **`ModuleNotFoundError`**: Ensure all `.py` files (`base_env.py`, `utils.py`, `custom_wrappers.py`, etc.) are in the same directory as the script you're running, or that your `PYTHONPATH` is correctly configured. Double-check file names (e.g., `utils.py` vs `uitls.py`).
  * **`TypeError: __init__() got an unexpected keyword argument 'total_timesteps'`**: This means `total_timesteps` was incorrectly passed to the `PPO` constructor. Ensure you've applied the fix in `train_simple_agent.py` where `total_timesteps` is `pop()`'d from `ppo_params` and passed to `model.learn()` instead.
  * **`AssertionError: The algorithm only supports ... action spaces but Tuple(...) was provided`**: This indicates `Stable Baselines3` doesn't directly support your environment's `Tuple` action space. Ensure you've correctly implemented and applied the `FlattenAction` wrapper from `custom_wrappers.py` to your environment instances before passing them to the PPO model.
  * **"No k-lines returned" / Data Fetching Errors**: Verify your `config.yaml` date ranges are valid and have data on Binance. Ensure your Binance API keys are correct and have necessary permissions. Check your internet connection.
  * **"Error loading K-line data from cache missing some requested TA features"**: If you change `kline_price_features` in `config.yaml`, the old cache file might not have the new columns. The script will automatically delete and re-fetch, but you can manually delete `binance_data_cache/` contents to force a full re-download.
  * **`BinanceAPIException` (during trade execution)**: Check your API keys and secrets. Ensure they are for the correct network (Testnet vs. Mainnet) and have sufficient permissions. Verify trading pair symbols and quantities adhere to Binance's exchange rules (e.g., minimum trade amount, decimal precision). Adjust `recv_window_ms` in `config.yaml` if you get timestamp errors.
  * **`TA-Lib` Installation Issues**: Refer to the "TA-Lib Installation" section in this README for platform-specific instructions, especially for the C library.
