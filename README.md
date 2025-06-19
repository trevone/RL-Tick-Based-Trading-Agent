
# RL Tick-Based Trading Agent

This project implements a Reinforcement Learning (RL) agent for algorithmic trading, designed to operate on real-time tick and K-line market data using the Binance exchange API. The agent is trained using various RL algorithms from Stable Baselines3 within a custom, extensible Gymnasium environment framework.

## Table of Contents

1.  [Features](https://www.google.com/search?q=%23features)
2.  [Project Structure](https://www.google.com/search?q=%23project-structure)
3.  [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
      * [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      * [Environment Setup](https://www.google.com/search?q=%23environment-setup)
      * [Binance API Keys](https://www.google.com/search?q=%23binance-api-keys)
      * [TA-Lib Installation](https://www.google.com/search?q=%23ta-lib-installation)
4.  [Configuration](https://www.google.com/search?q=%23configuration)
5.  [Usage](https://www.google.com/search?q=%23usage)
      * [1. Data Acquisition and Caching](https://www.google.com/search?q=%231-data-acquisition-and-caching)
      * [2. Data Verification](https://www.google.com/search?q=%232-data-verification)
      * [3. Agent Training](https://www.google.com/search?q=%233-agent-training)
      * [4. Hyperparameter Optimization](https://www.google.com/search?q=%234-hyperparameter-optimization)
      * [5. Model Evaluation](https://www.google.com/search?q=%235-model-evaluation)
      * [6. Live Trading (Testnet/Mainnet)](https://www.google.com/search?q=%236-live-trading-testnetmainnet)
6.  [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)
7.  [Contributing](https://www.google.com/search?q=%23contributing)
8.  [License](https://www.google.com/search?q=%23license)

## Features

  * **Extensible Environment Framework**: Includes a base trading environment (`src/environments/base_env.py`) and a dynamic loader (`src/environments/env_loader.py`) that can discover and run experimental environments from the `src/environments/experiments/` directory. This allows for rapid testing of new reward structures and state representations.
  * **Dynamic & Configurable Feature Engineering**: The feature engineer (`src/data/feature_engineer.py`) is highly flexible.
      * Calculate any TA-Lib indicator on any data source (`Open`, `High`, `Low`, `Close`) directly from the `config.yaml`.
      * Easily add custom, non-TA-Lib indicators (e.g., "Envelopes") in a dedicated file.
  * **Observation Space**: Integrates granular tick data, comprehensive K-line data, and a vast array of technical indicators and candlestick patterns for a rich market observation.
  * **Multi-Algorithm Support**: Supports various Stable Baselines3 algorithms, including **PPO**, **SAC**, **DDPG**, **A2C**, and **RecurrentPPO** (from `sb3-contrib`), configurable via `config.yaml`.
  * **Unified and Powerful Configuration**: A layered YAML configuration system where a main `config.yaml` overrides defaults. Features are now defined with a powerful dictionary structure, allowing for precise control over parameters and data sources.
  * **Binance Data Integration**: Fetches historical K-line and aggregate trade (tick) data directly from Binance (Mainnet or Testnet) and implements smart caching to local Parquet files, significantly speeding up subsequent runs.
  * **Hyperparameter Optimization**: Integrates Optuna for automated hyperparameter tuning to maximize agent performance.
  * **Complete Trading Workflow**: Includes scripts for every stage of the process: data acquisition, validation, agent training, evaluation, and live trading.
  * **Utility & Analysis Scripts**: Provides a `scripts/` directory with helpful tools for viewing cached data and verifying environment observations.

## Project Structure

```
.
├── src/
│   ├── environments/
│   │   ├── base_env.py                 # The core trading environment
│   │   ├── custom_wrappers.py          # Action space wrapper
│   │   ├── env_loader.py               # Dynamically loads all environments
│   │   └── experiments/                # Directory for experimental envs
│   │       ├── profit_driven_env.py
│   │       └── ... (other custom envs)
│   ├── data/
│   │   ├── binance_client.py
│   │   ├── config_loader.py
│   │   ├── custom_indicators.py        # NEW: Home for custom TA indicators
│   │   ├── data_loader.py
│   │   ├── data_manager.py
│   │   ├── data_validator.py
│   │   ├── feature_engineer.py         # NEW: Highly flexible feature engine
│   │   └── path_manager.py
│   └── agents/
│       ├── agent_utils.py              # Helper functions for agents
│       ├── evaluate_agent.py
│       ├── hpo.py
│       ├── live_trader.py
│       ├── simple_eval.py              # A simplified evaluation script
│       └── train_agent.py
│
├── configs/
│   ├── config.yaml                     # Main config (you edit this)
│   ├── config.sample.yaml              # A full example of every option
│   └── defaults/                       # Default parameters for all components
│       ├── run_settings.yaml
│       ├── environment.yaml
│       ├── ppo_params.yaml
│       └── ...
│
├── scripts/                            # NEW: Utility scripts
│   ├── read_cache_sample.py
│   ├── verify_observation.py
│   └── ...
│
├── tests/
├── data_cache/
├── logs/
└── ...
```

## Setup and Installation

### Prerequisites

  * Python 3.8+
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
    ```

### Binance API Keys

You need API keys from Binance to fetch data and trade. It's **highly recommended** to set API keys as environment variables (`BINANCE_API_KEY`, `BINANCE_API_SECRET`).

### TA-Lib Installation

`TA-Lib` is a high-performance library that requires a C library to be installed first. Follow platform-specific instructions before running `pip install TA-Lib`.

## Configuration

The project uses a layered configuration system. You primarily edit `config.yaml` in the project root to override the defaults found in `configs/defaults/`.

**Crucially**, technical indicators are now defined with a flexible dictionary format that allows you to specify the function, its parameters, and the data source (`High`, `Low`, `Close`, etc.).

**Example `config.yaml`:**

```yaml
# 1. Choose the RL algorithm
agent_type: "PPO"

# 2. Override run settings
run_settings:
  model_name: "ema_crossover_agent"
  log_level: "normal"
  start_date_train: "2024-03-01 00:00:00"
  end_date_train: "2024-03-31 23:59:59"
  start_date_eval: "2024-04-01 00:00:00"
  end_date_eval: "2024-04-03 23:59:59"

# 3. Override environment settings
environment:
  initial_balance: 5000.0
  kline_window_size: 30

  # --- Define technical analysis features ---
  kline_price_features:
    # Base features
    Open: {}
    High: {}
    Low: {}
    Close: {}
    Volume: {}
    
    # Example: EMA on High prices
    EMA_High_10:
      function: "EMA"
      params:
        timeperiod: 10
      data_source: "High" # Use the 'High' price for this EMA

    # Example: EMA on Low prices
    EMA_Low_10:
      function: "EMA"
      params:
        timeperiod: 10
      data_source: "Low" # Use the 'Low' price for this EMA
      
    # Example: Bollinger Bands on the default 'Close' price
    BBANDS_Upper:
      function: "BBANDS"
      params:
        timeperiod: 20
        nbdevup: 2
      output_field: 0 # 0 for Upper, 1 for Middle, 2 for Lower

# 4. Override algorithm-specific hyperparameters
ppo_params:
  total_timesteps: 2000000
  learning_rate: 0.0001
  n_steps: 4096
```

## Usage

**Important:** All commands should be run from the **root directory** of the project.

### 1\. Data Acquisition and Caching

Use the `data_manager.py` script to fetch and cache historical data from Binance.

```bash
python -m src.data.data_manager --start_date <YYYY-MM-DD> --end_date <YYYY-MM-DD> --symbol <SYMBOL> --data_type <agg_trades|kline> [kline_options]
```

### 2\. Data Verification

Use the `data_validator.py` script to check the integrity of cached data.

```bash
python -m src.data.data_validator --cache_dir data_cache/
```

### 3\. Agent Training

Configure `config.yaml` and run the training script:

```bash
python -m src.agents.train_agent
```

### 4\. Hyperparameter Optimization

Configure `configs/defaults/hyperparameter_optimization.yaml` and run the optimization script:

```bash
python -m src.agents.hpo
```

### 5\. Model Evaluation

Configure the evaluation period and model path in `config.yaml` and run:

```bash
python -m src.agents.evaluate_agent
```

### 6\. Live Trading (Testnet/Mainnet)

**WARNING: Live trading involves real financial risk. Always test thoroughly on Testnet.**

Configure your API keys and model path in `config.yaml` and run:

```bash
python -m src.agents.live_trader
```

## Troubleshooting

  * **`ModuleNotFoundError`**: Ensure you are running scripts from the project root directory using `python -m`.
  * **`KeyError`**: If you get a `KeyError` after changing configs, ensure the key exists in the correct default `.yaml` file.
  * **`BinanceAPIException`**: Check your API keys and permissions.