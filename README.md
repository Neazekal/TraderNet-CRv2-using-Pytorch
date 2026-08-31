# TraderNet-CRv2 (PyTorch Version)

**TraderNet-CRv2** is an advanced Deep Reinforcement Learning (DRL) system for cryptocurrency trading that combines **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)** with technical analysis and rule-based safety mechanisms.

This repository is an extended and modernized PyTorch implementation of the original TraderNet-CR architecture, featuring:
- **Agents**: PPO (Proximal Policy Optimization) and DQN (Deep Q-Network).
- **Safety Mechanisms**: N-Consecutive trend monitoring and Smurfing (optional) to mitigate risk.
- **Environment**: Custom Trading Environment compatible with OpenAI Gym / Gymnasium and Stable-Baselines3 APIs.

## Features

*   **Deep Reinforcement Learning**: Train agents using state-of-the-art RL algorithms (PPO, DQN).
*   **Technical Analysis**: Integration of popular indicators (MACD, RSI, Bollinger Bands, etc.) as state features.
*   **Rule-Based Overlays**: Hybrid approach combining RL with "N-Consecutive" trend rules for safer entry/exit.
*   **Flexible Configuration**: Centralized configuration for agents, environments, and datasets.

## Installation

### Prerequisites
*   Python >= 3.10
*   pip

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd TraderNet-CRv2-using-Pytorch
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (conda or venv).
    ```bash
    pip install -r requirements.txt
    ```
    For development and testing:
    ```bash
    pip install -r requirements-dev.txt
    ```

## Usage Settings

All major configurations (agent hyperparameters, training settings, supported coins) are managed in **`config.py`**. Modify this file to change:
*   `datasets_dict`: The list of cryptocurrencies to train/evaluate on.
*   `env_config`: Environment parameters like fees, leverage, and timeframe.
*   `agent_config`: Hyperparameters for PPO and DQN agents.

## Workflow

### 1. Download Data
Download historical OHLCV (Open, High, Low, Close, Volume) data from Binance.

```bash
# Example: Download DOGEUSDT 1-hour data from 2020-01-01 to 2023-01-01
python download_olhcv.py --symbol DOGEUSDT --interval 1h --start "2020-01-01" --end "2023-01-01"
```
The data will be saved to `data/`.

### 2. Process Data
Convert the raw downloaded CSVs into the dataset format used by the training environment. This calculates technical indicators and normalizes the data.

```bash
python database/build_dataset.py --data_dir data
```
Processed datasets are saved to `database/storage/datasets/`.

### 3. Validate Environment
Verify Gymnasium and Stable-Baselines3 environment compatibility and lifecycle behavior:

```bash
python validate_environment.py
```

### 4. Train Agents
Train RL agents on the processed datasets.

Standard TraderNet:
```bash
python train.py
```

Smurf-wrapped TraderNet:
```bash
python train_smurf.py
```

*   Checkpoints are saved to: `database/storage/checkpoints/`
*   Training logs (TensorBoard) are included in the checkpoint directories.
*   Results (Evaluation metrics) are saved to `experiments/tradernet/` or `experiments/smurf/`.

### 5. Evaluate Agents
Evaluate trained agents on unseen data or specific test sets.

Standard evaluation:
```bash
python eval.py
```

Integrated / Hybrid evaluation (TraderNet + Smurf):
```bash
python integrated.py
```

Metrics and PnL (Profit and Loss) curves will be saved to `experiments/tradernet/` and `experiments/integrated/`.

### 6. Run Tests
Run automated unit and integration tests:

```bash
pytest
```

## Project Structure

*   `agents/`: Implementation of RL agents (PPO, DQN).
*   `environments/`: Custom trading environments, factory, and reward functions.
*   `database/`: Scripts for data management, preprocessing, and dataset utilities.
*   `metrics/`: Performance metrics calculations (Sharpe, Sortino, Drawdown, etc.).
*   `rules/`: Safety rules like N-Consecutive.
*   `tests/`: Unit and integration tests.
*   `config.py`: Global configuration file.
*   `train.py`: Main training script for standard TraderNet.
*   `train_smurf.py`: Training script for Smurf reward wrapper.
*   `eval.py`: Evaluation script for standalone models.
*   `integrated.py`: Evaluation script for hybrid / integrated models.
*   `validate_environment.py`: Synthetic SB3/lifecycle environment validator.
*   `download_olhcv.py`: Data downloader.

## Disclaimer

**Important Note**: This software is for **educational and research purposes only**. It is not a commercial product and should not be used as financial advice. Trading cryptocurrencies involves significant risk.
