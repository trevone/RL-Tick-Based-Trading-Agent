
# RL Tick-Based Trading Agent

This project implements a Reinforcement Learning (RL) agent for algorithmic trading, designed to operate on real-time tick and K-line market data using the Binance exchange API. The agent is trained using various RL algorithms from Stable Baselines3 within a custom Gymnasium environment.

## Table of Contents

1.  [Features](#features)
2.  [Project Structure](#project-structure)
3.  [Setup and Installation](#setup-and-installation)
      * [Prerequisites](#prerequisites)
      * [Environment Setup](#environment-setup)
      * [Binance API Keys](#binance-api-keys)
      * [TA-Lib Installation](#ta-lib-installation)
4.  [Configuration](#configuration)
5.  [Usage](#usage)
      * [1. Data Acquisition and Caching](#1-data-acquisition-and-caching)
      * [2. Data Verification](#2-data-verification)
      * [3. Agent Training](#3-agent-training)
      * [4. Hyperparameter Optimization](#4-hyperparameter-optimization)
      * [5. Model Evaluation](#5-model-evaluation)
      * [6. Live Trading (Testnet/Mainnet)](#6-live-trading-testnetmainnet)
6.  [Troubleshooting](#troubleshooting)
7.  [Contributing](#contributing)
8.  [License](#license)

## Features

  * **Custom Gymnasium Trading Environment (`src/environments/base_env.py`)**: A tick-level trading environment simulating Binance market dynamics, commissions, and portfolio management.
  * **Observation Space**: Integrates granular tick data (Price, Quantity), comprehensive K-line data (OHLCV, SMAs, RSIs, ATR, MACD), and candlestick patterns (Doji, Hammer, Engulfing, etc.) for a rich market observation.
  * **Action Space**: Supports both discrete trading actions (Hold, Buy, Sell) and a continuous profit target parameter for strategic exits.
  * **Multi-Algorithm Support**: Now supports various Stable Baselines3 algorithms, including **PPO**, **SAC**, **DDPG**, **A2C**, and **RecurrentPPO** (from `sb3-contrib`), configurable via `config.yaml`.
  * **Unified Training & Evaluation Scripts**: A single `src/agents/train_agent.py` script handles training for all supported algorithms, and `src/agents/evaluate_agent.py` can evaluate any trained model.
  * **Action Space Wrapper (`src/environments/custom_wrappers.py`)**: Includes a custom `FlattenAction` wrapper to convert the environment's `Tuple` action space into a `Box` space, making it compatible with standard Stable Baselines3 algorithms.
  * **Binance Data Fetching (`src/data/utils.py`)**: Fetches historical K-line and aggregate trade (tick) data directly from Binance (Mainnet or Testnet) using `python-binance`.
  * **Data Caching**: Implements smart caching of fetched data to local Parquet files in `data_cache/`, significantly speeding up subsequent runs and reducing API calls.
  * **Technical Analysis (TA-Lib)**: Calculates various technical indicators and detects candlestick patterns using the optimized `TA-Lib` library. Supports dynamic period configuration for indicators like SMA, EMA, and RSI.
  * **Comprehensive Configuration (`config.yaml` & `configs/defaults/`)**: All environment, agent, data, and run-specific parameters are managed via a central `config.yaml` that **overrides** default settings loaded from `configs/defaults/` files, ensuring easy customization and reproducibility.
  * **Configuration Hashing**: Training runs are uniquely identified by a hash of their core configuration parameters, ensuring organized logging and easy model retrieval.
  * **Model Evaluation Script (`src/agents/evaluate_agent.py`)**: A dedicated script to evaluate a trained agent on unseen historical data, providing performance summaries, trade history, and interactive plots of price, equity, and trade signals.
  * **Live Trading Script (`src/agents/live_trader.py`)**: Connects to Binance WebSockets for real-time tick data, feeds observations to the agent, and can be configured to execute actual market trades (on Testnet or Mainnet) with quantity quantization.
  * **Hyperparameter Optimization (`src/agents/hyperparameter_optimization.py`)**: Integrates Optuna for automated hyperparameter tuning, maximizing agent performance by exploring different parameter combinations for any supported algorithm.

## Project Structure

```
.
├── src/                                # All core Python source code
│   ├── environments/                   # Environment definitions
│   │   ├── base_env.py                 # Custom Gymnasium trading environment
│   │   └── custom_wrappers.py          # Custom Gymnasium environment wrappers
│   ├── data/                           # Data handling utilities
│   │   ├── utils.py                    # Data fetching, TA calculation, config handling
│   │   ├── data_downloader_manager.py  # Script for managing data downloads
│   │   └── check_tick_cache.py         # Script for validating cached data
│   └── agents/                         # Agent-related scripts
│       ├── train_agent.py              # Unified training script for all algorithms
│       ├── evaluate_agent.py           # Unified evaluation script
│       ├── live_trader.py              # Unified live trading script
│       └── hyperparameter_optimization.py # Optuna optimization script
│
├── configs/                            # Configuration files
│   ├── config.yaml                     # Main config (overrides defaults), specifies agent_type
│   ├── config.sample.yaml              # Full example of all default options
│   └── defaults/                       # Default parameters for all components/algorithms
│       ├── run_settings.yaml
│       ├── environment.yaml
│       ├── binance_settings.yaml
│       ├── evaluation_data.yaml
│       ├── hash_keys.yaml
│       ├── ppo_params.yaml
│       ├── sac_params.yaml
│       ├── ddpg_params.yaml
│       ├── a2c_params.yaml
│       └── recurrent_ppo_params.yaml
│
├── tests/                              # Unit and integration tests
│   ├── environments/                   # Tests for environment components
│   ├── data/                           # Tests for data handling components
│   └── agents/                         # Tests for agent training/evaluation logic
│
├── data_cache/                         # Cached historical K-line and tick data (auto-generated)
├── logs/                               # Output directory for logs
│   ├── training/                       # Training run logs (hashed subdirectories)
│   ├── evaluation/                     # Evaluation run logs (timestamped subdirectories)
│   ├── live_trading/                   # Live trading session logs
│   └── tensorboard_logs/               # Base directory for TensorBoard logs
│
├── optuna_studies/                     # Optuna SQLite database and best param JSONs (auto-generated)
├── scripts/                            # Various utility scripts (e.g., view_kline_data.py)
├── .gitignore                          # Specifies files/directories to ignore in Git
└── README.md                           # This file
└── requirements.txt                    # List of Python dependencies
```

## Setup and Installation

### Prerequisites

  * Python 3.8+ (recommended using Anaconda/Miniconda)
  * Access to Binance API keys

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:trevone/RL-Tick-Based-Trading-Agent.git
    cd RL-Tick-Based-Trading-Agent.git
    ```
2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n rl_trading_env python=3.9 # Or your preferred Python version
    conda activate rl_trading_env
    ```
3.  **Install core dependencies:**
    ```bash
    pip install pandas numpy gymnasium stable-baselines3[extra] python-binance PyYAML matplotlib optuna sb3-contrib tqdm
    # For plotting optimization results (optional)
    pip install plotly
    ```

### Binance API Keys

You need API keys from Binance to fetch historical data and (optionally) execute live trades.

1.  **Generate API Keys:**

      * **Mainnet:** Log in to your Binance account, go to "API Management" (or similar section) to create new API keys. Grant appropriate permissions (e.g., "Read Info", "Spot Trading").
      * **Testnet:** For testing live trading safely, create separate API keys on the Binance Testnet: `https://testnet.binance.vision/` (URL may vary, search for "Binance Testnet"). These keys are distinct from Mainnet keys.

2.  **Configure `config.yaml`:**
    It's **highly recommended** to set API keys as environment variables (e.g., `BINANCE_API_KEY`, `BINANCE_API_SECRET`). If set this way, `api_key: null` and `api_secret: null` (or simply omitted) in `configs/defaults/binance_settings.yaml` will make the scripts read from environment variables. If you directly set them in `config.yaml`, ensure they are not committed to version control.

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
    wget [http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz)
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

## Configuration

The project uses a layered configuration system:

1.  **Default Settings (`configs/defaults/`)**: These files (e.g., `run_settings.yaml`, `environment.yaml`, `ppo_params.yaml`) contain all the base parameters for different components and algorithms.
2.  **Main Configuration (`config.yaml`)**: This file in the project root is your primary interface. It **overrides** any values from the default files.
    * **Crucially**, you must specify the `agent_type` (e.g., `PPO`, `SAC`, `RecurrentPPO`) in `config.yaml`. The system will then use the corresponding `_params.yaml` from `configs/defaults/`.
    * Only include parameters in `config.yaml` that you wish to change from their default values.

    **Example `config.yaml`:**
    ```yaml
    # config.yaml
    agent_type: "PPO" # Choose your algorithm: PPO, SAC, DDPG, A2C, RecurrentPPO

    # === Data Period Overrides (Unified for K-line and Ticks) ===
    binance_settings:
      start_date_train: "2024-01-10 00:00:00"
      end_date_train: "2024-01-20 23:59:59"

    evaluation_data:
      start_date_eval: "2024-01-21 00:00:00"
      end_date_eval: "2024-01-22 23:59:59"
    
    # === Other Example Overrides (Optional) ===
    run_settings:
      model_name: "my_custom_ppo_run"
      log_level: "detailed" # Override default "normal" for more verbose output

    environment:
      initial_balance: 5000.0
      kline_window_size: 30
      tick_resample_interval_ms: 500 # Resample tick data to 500ms resolution

    ppo_params: # Only applies if agent_type is PPO
      total_timesteps: 5000000
      learning_rate: 0.0001
    ```

## Usage

### 1\. Data Acquisition and Caching

The project uses a local data cache to speed up repeated runs and reduce API calls. The `data_downloader_manager.py` script automates this process.

**Important:** Always run this script from the **root directory of your project** (e.g., `RL-Tick-Based-Trading-Agent`). Use the `-m` flag to ensure Python correctly resolves internal package imports.

**Command Structure:**

```bash
python -m src.data.data_downloader_manager --start_date <YYYY-MM-DD> --end_date <YYYY-MM-DD> --symbol <SYMBOL> --data_type <agg_trades|kline> [optional_kline_args]
```

**Arguments:**

* `--start_date`: The start date for data download (e.g., `2024-01-01`).
* `--end_date`: The end date for data download (e.g., `2024-01-07`).
* `--symbol`: The trading pair symbol (e.g., `BTCUSDT`).
* `--data_type`: Type of data to download.
    * `agg_trades`: For high-frequency aggregate trade (tick) data.
    * `kline`: For candlestick (OHLCV) data, optionally with technical indicators.
* `--interval`: (Required if `--data_type` is `kline`). The K-line interval (e.g., `1m`, `1h`, `1d`).
* `--kline_features`: (Optional, for `kline` data\_type). A space-separated list of features to include (e.g., `Open High Close Volume SMA_10 RSI_7`). These features will be calculated and saved with the K-line data.

**New Features in Download Manager:**

* **Progress Bar**: A `tqdm` progress bar now displays the daily download status (e.g., `Downloading BTCUSDT agg_trades: 14%|██████████████████ | 1/7 [00:00<00:00, 9.04day/s]`), making it easier to track long-running data acquisition tasks.
* **Cleaner Output**: Detailed internal logging from data fetching and validation utilities is suppressed, providing a neater and more focused output in the console. Messages related to file existence, download status, and validation success/failure will be shown concisely.

**Examples:**

1.  **Download Aggregate Trade Data for a week:**
    ```bash
    python -m src.data.data_downloader_manager --start_date 2024-01-01 --end_date 2024-01-07 --symbol BTCUSDT --data_type agg_trades
    ```
    *(You will see a progress bar indicating progress by day.)*

2.  **Download 1-hour K-line Data with specific technical indicators:**
    ```bash
    python -m src.data.data_downloader_manager --start_date 2024-01-01 --end_date 2024-01-07 --symbol BTCUSDT --data_type kline --interval 1h --kline_features Open High Low Close Volume SMA_10 RSI_7 MACD
    ```

Data will be saved to the `data_cache/` directory in Parquet format. The manager will check existing files and only download missing or invalid data.

### 2\. Data Verification

After downloading, it's crucial to verify the integrity and structure of your cached data.

### Using `scripts/read_cache_sample.py`

This script allows you to quickly inspect a sample of your cached data for a specific day.

**Important:** Run this script from the **root directory of your project**.

**Command Structure:**

```bash
python -m scripts.read_cache_sample --symbol <SYMBOL> --date <YYYY-MM-DD> --data_type <agg_trades|kline> [optional_kline_args]
```

**Arguments:**

* `--symbol`: The trading pair symbol (e.g., `BTCUSDT`).
* `--date`: The specific date of the data to inspect (e.g., `2024-01-01`).
* `--data_type`: The type of data (`agg_trades` or `kline`).
* `--interval`: (Required if `--data_type` is `kline`). The K-line interval (e.g., `1h`).
* `--features`: (Optional, for `kline` data\_type). **Crucially, these must match the features used when you downloaded the K-line data** so the script can locate the correct file.

**Examples:**

1.  **Inspect Aggregate Trade Data for a day:**
    ```bash
    python -m scripts.read_cache_sample --symbol BTCUSDT --date 2024-01-01 --data_type agg_trades
    ```

2.  **Inspect K-line Data with specific features for a day:**
    ```bash
    python -m scripts.read_cache_sample --symbol BTCUSDT --date 2024-01-01 --data_type kline --interval 1h --features Open High Low Close Volume SMA_10 RSI_7 MACD
    ```

The script will print the head and tail of the DataFrame, its shape, time range, and columns, helping you confirm your data is as expected.

### Data Integrity Checks (`src/data/check_tick_cache.py`)

The `validate_daily_data` function, which is called internally by the download manager, performs comprehensive checks on each cached file (monotonic index, column types, missing values, timestamp ranges). You can also run `check_tick_cache.py` directly for detailed validation:

```bash
python -m src.data.check_tick_cache --filepath data_cache/bn_aggtrades_BTCUSDT_2024-01-01.parquet --log_level detailed
```
You can also use `--log_level normal` or `--log_level none` to control verbosity.

### Tick Data Resampling for Training

A new configuration option allows you to control the resolution of tick data used by the environment:
* `environment.tick_resample_interval_ms`: Located in `configs/defaults/environment.yaml`. Set this to an integer (e.g., `100` for 100ms, `1000` for 1-second resolution) to resample tick data. Set to `null` to use original tick resolution. This helps "thin out" data for faster training while maintaining chronological order.

### 3\. Agent Training

To train your RL agent:

1.  **Configure `config.yaml`**:
      * Set `agent_type` to your desired algorithm (e.g., `PPO`, `SAC`).
      * **Set the training data period** by defining `binance_settings.start_date_train` and `binance_settings.end_date_train`.
      * Adjust any other parameters in `environment`, `run_settings`, or the algorithm-specific sections (e.g., `ppo_params`) as desired.
      * Consider setting `environment.tick_resample_interval_ms` to a value that provides a suitable data resolution for your training goals.
2.  **Run the training script**:
    ```bash
    python -m src.agents.train_agent
    ```
      * Training logs, TensorBoard files, and the trained model (`trained_model_final.zip` and `best_model/best_model.zip`) will be saved in a unique subdirectory under `logs/training/` named after the configuration hash and model name.
      * You can monitor training progress using TensorBoard: `tensorboard --logdir logs/tensorboard_logs/`

### 4\. Hyperparameter Optimization

To find optimal hyperparameters using Optuna:

1.  **Configure `src/agents/hyperparameter_optimization.py`**:
      * Adjust `OPTIMIZATION_CONFIG` for number of trials, sampler, pruner, and the SQLite database URL.
      * Define the `HYPERPARAMETER_SEARCH_SPACE` with ranges for the the parameters you want to optimize. Ensure these parameters are also listed in `configs/defaults/hash_keys.yaml` under the appropriate `agent_params` section for consistent hashing.
2.  **Run the optimization script**:
    ```bash
    python -m src.agents.hyperparameter_optimization
    ```
      * Optuna will run multiple training trials, saving progress to `optuna_studies/optuna_study.db`.
      * Upon completion, it will print the best found hyperparameters and their corresponding metric value. It will also save these to `optuna_studies/best_hyperparameters.json`.
      * If `plotly` is installed, it can show interactive plots of the optimization history.

### 5\. Model Evaluation

To evaluate a trained model on unseen data:

1.  **Configure `config.yaml`**:
      * In the `evaluation_data` section, specify `start_date_eval` and `end_date_eval` for a period *different* from your training data.
      * In `run_settings`, set `model_path` to the path of your trained model (`.zip` file) you wish to evaluate (e.g., `logs/training/<YOUR_RUN_ID>/best_model/best_model.zip`). The script will attempt to auto-resolve if `model_path` is `null`.
      * Adjust `n_evaluation_episodes` in `configs/defaults/run_settings.yaml` (or override in `config.yaml`) as needed.
2.  **Run the evaluation script**:
    ```bash
    python -m src.agents.evaluate_agent
    ```
      * The script will fetch evaluation data, load the specified model, run evaluation episodes, and print a summary of performance metrics (total reward, final equity, profit/loss percentage).
      * It will generate and save a performance chart (`<eval_run_id>_performance_chart.png`) and trade history (`<eval_run_id>_trade_history.json`) in `logs/evaluation/`.

### 6\. Live Trading (Testnet/Mainnet)

The `src/agents/live_trader.py` script allows real-time market data consumption and agent-driven trade execution.

**WARNING: Live trading involves real money and significant financial risk. Always test thoroughly on Testnet before deploying to Mainnet.**

1.  **Configure `config.yaml`**:
      * Set `binance_settings.testnet: true` for Testnet trading, or `false` for Mainnet.
      * **Crucially, provide your corresponding Binance API keys** (via environment variables or directly in `configs/defaults/binance_settings.yaml`, overridden by `config.yaml`).
      * Set `run_settings.model_path` to the `.zip` file of the trained model you want to use.
2.  **Enable Actual Trading (Modify `src/agents/live_trader.py`)**:
    By default, trade execution is simulated. To enable real trades:
      * Open `src/agents/live_trader.py`.
      * Locate the `--- ACTUAL BINANCE API CALL (UNCOMMENT TO ENABLE REAL TRADES) ---` sections within the `_execute_trade` function.
      * **Uncomment** the actual `self.client.order_limit_buy(...)` and `self.client.order_limit_sell(...)` lines.
      * Ensure the `quantity` parameter passed to these calls is correctly quantized based on Binance's rules (the script includes `quantize_quantity` helpers if implemented).
3.  **Run the live trader**:
    ```bash
    python -m src.agents.live_trader
    ```
      * The script will connect to the Binance WebSocket, start fetching real-time ticks, and when the agent makes a decision, it will attempt to send real orders to your configured Binance account (Testnet or Mainnet).
      * Console output will show received ticks, agent decisions, and the status of order attempts.

## Troubleshooting

  * **`ModuleNotFoundError`**: Ensure you are running scripts from the **project root directory** using the `python -m` command. This ensures Python can correctly find all modules within the `src` package. Also, confirm all dependencies listed in "Environment Setup" are installed.
  * **`TypeError` in `pd.to_datetime` related to `tz` keyword**: This can occur with specific pandas versions. Ensure you're using `pd.to_datetime(...).tz_localize('UTC')` for robustness if directly passing a timezone string is problematic.
  * **`FileNotFoundError` for config files**: Ensure all default `.yaml` files exist in `configs/defaults/` as specified.
  * **`BinanceAPIException` (during trade execution)**: Check your API keys and secrets. Ensure they are for the correct network (Testnet vs. Mainnet) and have sufficient permissions. Verify trading pair symbols and quantities adhere to Binance's exchange rules (e.g., minimum trade amount, decimal precision). Adjust `recv_window_ms` in `binance_api_client` section of `config.yaml` if you get timestamp errors.
  * **`TA-Lib` Installation Issues**: Refer to the "TA-Lib Installation" section in this README for platform-specific instructions, especially for the C library.
  * **`AssertionError` in tests related to floating-point comparisons**: Use `pytest.approx` for comparing floating-point numbers in tests due to precision variations.
  * **Tests failing due to `MagicMock` type errors**: Ensure your mocked objects return the correct data types (e.g., integers for timestamps, floats for prices) when numerical operations are expected.
