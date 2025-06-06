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
  * **Multi-Algorithm Support**: Supports various Stable Baselines3 algorithms, including **PPO**, **SAC**, **DDPG**, **A2C**, and **RecurrentPPO** (from `sb3-contrib`), configurable via `config.yaml`.
  * **Unified Configuration (`configs/`)**: A layered configuration system where a main `config.yaml` overrides defaults from the `configs/defaults/` directory. **`run_settings.yaml`** now centralizes all data period, symbol, and logging configurations.
  * **Action Space Wrapper (`src/environments/custom_wrappers.py`)**: Includes a custom `FlattenAction` wrapper to convert the environment's `Tuple` action space into a `Box` space, making it compatible with standard Stable Baselines3 algorithms.
  * **Binance Data Fetching (`src/data/binance_client.py`)**: Fetches historical K-line and aggregate trade (tick) data directly from Binance (Mainnet or Testnet) using `python-binance`.
  * **Data Caching**: Implements smart caching of fetched data to local Parquet files, significantly speeding up subsequent runs and reducing API calls.
  * **Technical Analysis (TA-Lib)**: Calculates various technical indicators and detects candlestick patterns using the optimized `TA-Lib` library.
  * **Configuration Hashing**: Training runs are uniquely identified by a hash of their core configuration parameters, ensuring organized logging and easy model retrieval.
  * **Model Evaluation Script (`src/agents/evaluate_agent.py`)**: A dedicated script to evaluate a trained agent on unseen historical data, providing performance summaries and trade history plots.
  * **Live Trading Script (`src/agents/live_trader.py`)**: Connects to Binance WebSockets for real-time tick data and executes agent-driven trades (on Testnet or Mainnet).
  * **Hyperparameter Optimization (`src/agents/hyperparameter_optimization.py`)**: Integrates Optuna for automated hyperparameter tuning to maximize agent performance.

## Project Structure

```
.
├── src/                                # All core Python source code
│   ├── environments/                   # Environment definitions
│   │   ├── base_env.py                 # Custom Gymnasium trading environment
│   │   └── custom_wrappers.py          # Custom Gymnasium environment wrappers
│   ├── data/                           # Data handling utilities
│   │   ├── binance_client.py           # Functions for fetching data from Binance API
│   │   ├── config_loader.py            # YAML configuration loading and merging
│   │   ├── data_loader.py              # High-level functions for loading data ranges
│   │   ├── feature_engineer.py         # Technical indicator calculation
│   │   ├── path_manager.py             # Manages file paths for cached data
│   │   ├── data_manager.py             # Script for managing data downloads
│   │   └── data_validator.py           # Script for validating cached data
│   └── agents/                         # Agent-related scripts
│       ├── train_agent.py              # Unified training script for all algorithms
│       ├── evaluate_agent.py           # Unified evaluation script
│       ├── live_trader.py              # Unified live trading script
│       └── hpo.py                      # Optuna optimization script
│
├── configs/                            # Configuration files
│   ├── config.yaml                     # Main config (overrides defaults), specifies agent_type
│   ├── config.sample.yaml              # Full example of all default options
│   └── defaults/                       # Default parameters for all components/algorithms
│       ├── run_settings.yaml           # NEW: Centralized settings for runs, data, and logging
│       ├── environment.yaml
│       ├── binance_settings.yaml
│       ├── ppo_params.yaml
│       └── ... (other algo_params, hash_keys, etc.)
│
├── tests/                              # Unit and integration tests
├── data_cache/                         # Cached historical K-line and tick data (auto-generated)
├── logs/                               # Output directory for logs
├── optuna_studies/                     # Optuna SQLite database and best param JSONs (auto-generated)
├── scripts/                            # Various utility scripts
├── .gitignore
└── README.md
└── requirements.txt
```

## Setup and Installation

### Prerequisites

  * Python 3.8+ (recommended using Anaconda/Miniconda)
  * Access to Binance API keys

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:trevone/RL-Tick-Based-Trading-Agent.git
    cd RL-Tick-Based-Trading-Agent
    ```
2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n rl_trading_env python=3.9
    conda activate rl_trading_env
    ```
3.  **Install core dependencies:**
    ```bash
    pip install pandas numpy gymnasium stable-baselines3[extra] python-binance PyYAML matplotlib optuna sb3-contrib tqdm
    # For plotting optimization results (optional)
    pip install plotly
    ```

### Binance API Keys

You need API keys from Binance to fetch historical data and execute live trades.

1.  **Generate API Keys** on the Binance website (or Testnet for safe testing).
2.  **Configure API Keys**: It's **highly recommended** to set API keys as environment variables (`BINANCE_API_KEY`, `BINANCE_API_SECRET`). The scripts will automatically use them if the key/secret fields in `configs/defaults/binance_settings.yaml` are empty.

### TA-Lib Installation

`TA-Lib` is a high-performance library for technical analysis that requires a C library to be installed first. Follow platform-specific instructions (e.g., `brew install ta-lib` on macOS, or downloading binaries for Windows) before running `pip install TA-Lib`.

## Configuration

The project uses a layered configuration system for maximum flexibility and reproducibility.

1.  **Default Settings (`configs/defaults/`)**: These files contain the base parameters for every component.
    * **`run_settings.yaml`**: **This is the new central configuration file.** It controls logging, model naming, and crucially, all data settings like the symbol, cache directory, and date ranges for both training and evaluation.
    * **`environment.yaml`**: Defines the trading environment (observation space, rewards, initial balance).
    * **`ppo_params.yaml`**, **`sac_params.yaml`**, etc.: Contain the default hyperparameters for each specific RL algorithm.
    * **`binance_settings.yaml`**: Now only contains API credentials and testnet status.

2.  **Main Configuration (`config.yaml`)**: This file in the project root is your primary interface. Any parameter you set here **will override** the value from the default files.
    * **Crucially**, you must specify the `agent_type` (e.g., `PPO`, `SAC`, `RecurrentPPO`) in `config.yaml`. The system will then load the appropriate `..._params.yaml` file from the defaults.
    * You only need to add parameters to `config.yaml` that you want to change.

    **Example `config.yaml`:**
    ```yaml
    # config.yaml
    
    # 1. Choose the RL algorithm to use for training.
    agent_type: "PPO" 

    # 2. Override any settings from the 'run_settings.yaml' default file.
    run_settings:
      model_name: "my_ppo_btc_run_1"
      log_level: "detailed" # Use "detailed" for verbose output, "normal" for standard.
      
      # Override the default data periods for this experiment
      start_date_train: "2024-02-01 00:00:00"
      end_date_train: "2024-02-15 23:59:59"
      start_date_eval: "2024-02-16 00:00:00"
      end_date_eval: "2024-02-17 23:59:59"

    # 3. Override any environment settings.
    environment:
      initial_balance: 5000.0
      tick_resample_interval_ms: 500 # Resample tick data to 500ms intervals.
      kline_price_features: # Use a smaller set of features for this run
        - "Open"
        - "High"
        - "Low"
        - "Close"
        - "Volume"
        - "SMA_10"
        - "RSI_7"

    # 4. Override any algorithm-specific hyperparameters.
    #    This section should match the agent_type above.
    ppo_params:
      total_timesteps: 5000000
      learning_rate: 0.0001
      n_steps: 4096

    ```

## Usage

**Important:** All commands should be run from the **root directory** of the project.

### 1. Data Acquisition and Caching

Use the `data_manager.py` script to fetch and cache historical data from Binance.

**Command Structure:**

```bash
python -m src.data.data_manager --start_date <YYYY-MM-DD> --end_date <YYYY-MM-DD> --symbol <SYMBOL> --data_type <agg_trades|kline> [kline_options]
```

**Examples:**

1.  **Download Aggregate Trade (Tick) Data:**
    ```bash
    python -m src.data.data_manager --start_date 2024-01-01 --end_date 2024-01-07 --symbol BTCUSDT --data_type agg_trades
    ```

2.  **Download 1-hour K-line Data with TAs:**
    ```bash
    python -m src.data.data_manager --start_date 2024-01-01 --end_date 2024-01-07 --symbol BTCUSDT --data_type kline --interval 1h --kline_features Open Close Volume SMA_10 RSI_7
    ```

### 2. Data Verification

Use the `data_validator.py` script to check the integrity of all cached data for a symbol, or a single file.

**Examples:**

1.  **Check all cached files:**
    ```bash
    python -m src.data.data_validator --cache_dir data_cache/
    ```

2.  **Check a specific file:**
    ```bash
    python -m src.data.data_validator --filepath data_cache/BTCUSDT/bn_aggtrades_BTCUSDT_2024-01-01.parquet
    ```

### 3. Agent Training

1.  **Configure Your Experiment**:
    * Open `config.yaml`.
    * Set the `agent_type`.
    * In the `run_settings` section, define your `start_date_train`, `end_date_train`, and other desired parameters.
    * Adjust any hyperparameters in the corresponding `..._params` section (e.g., `ppo_params`).
2.  **Run the Training Script**:
    ```bash
    python -m src.agents.train_agent
    ```
    * Logs and models will be saved to a unique subdirectory in `logs/training/`.
    * Monitor progress with TensorBoard: `tensorboard --logdir logs/tensorboard_logs/`

### 4. Hyperparameter Optimization

1.  **Configure Optimization**:
      * Open `configs/defaults/hyperparameter_optimization.yaml` to define the study name, number of trials, and the search space for hyperparameters.
2.  **Run the Optimization Script**:
    ```bash
    python -m src.agents.hpo
    ```
      * Results are saved in `optuna_studies/`. The best parameters will be printed and saved to a `.json` file.

### 5. Model Evaluation

1.  **Configure Evaluation**:
      * Open `config.yaml`.
      * In the `run_settings` section, set the `start_date_eval` and `end_date_eval` to a period the model has not been trained on.
      * Set `model_path` to the path of your trained model `.zip` file. If `null`, the script will try to find it based on the other configs.
2.  **Run the Evaluation Script**:
    ```bash
    python -m src.agents.evaluate_agent
    ```
    * A performance summary will be printed to the console.
    * A performance chart and detailed trade history are saved to a timestamped subdirectory in `logs/evaluation/`.

### 6. Live Trading (Testnet/Mainnet)

**WARNING: Live trading involves real money and significant financial risk. Always test thoroughly on Testnet before deploying to Mainnet.**

1.  **Configure Live Trading**:
      * Open `config.yaml`.
      * In `binance_settings`, set `testnet: true` for Testnet or `false` for Mainnet.
      * Ensure your API keys are correctly configured.
      * In `run_settings`, set `model_path` to the `.zip` file of the trained model you want to deploy.
2.  **Run the Live Trader**:
    ```bash
    python -m src.agents.live_trader
    ```
    * The script will connect to the Binance WebSocket, receive live data, and print agent decisions to the console. By default, it runs in a paper-trading mode. To execute real trades, you must manually edit the `_execute_trade` function in `src/agents/live_trader.py`.

## Troubleshooting

  * **`ModuleNotFoundError`**: Ensure you are running scripts from the **project root directory** using the `python -m` command.
  * **`KeyError`**: If you get a `KeyError` after changing configs, ensure the key exists in the correct default `.yaml` file (e.g., `start_date_train` must be in `run_settings.yaml`).
  * **`BinanceAPIException`**: Check your API keys, permissions, and ensure the symbol and quantities are valid for the selected exchange (Testnet vs. Mainnet).