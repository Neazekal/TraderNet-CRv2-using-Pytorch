# TraderNet-CRv2 (PyTorch Version)

**TraderNet-CRv2** is an advanced Deep Reinforcement Learning (DRL) system for cryptocurrency trading that combines **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)** with technical analysis and rule-based safety mechanisms.

This repository is an extended and modernized PyTorch implementation of the original TraderNet-CR architecture, featuring:
- **Agents**: PPO (Proximal Policy Optimization) and DQN (Deep Q-Network).
- **Environment**: Gymnasium and Stable-Baselines3 compatible `TradingEnvironment` powered by a realistic single-asset `PortfolioSimulator`.
- **Safety Mechanisms**: N-Consecutive trend monitoring and Smurf gating mechanism.
- **Metrics**: Standardized arithmetic financial metrics (Cumulative Return, Loss Rate, Sharpe Ratio, Sortino Ratio, Maximum Drawdown).

## Phase 2: Realistic Portfolio Simulator & Causal Alignment

In Phase 2, the legacy heuristic reward matrix was replaced with a causal, single-asset portfolio simulator:
*   **Causal Next-Open Execution**: Decisions made from the observation window ending at bar $t$ are executed at the opening price of bar $t+1$ (`execution_open`) and marked to market at the closing price of bar $t+1$ (`mark_close`). No future prices (high/low/max/min) are leaked into fills or rewards.
*   **4 Discrete Actions**:
    *   `BUY` (0): Target position $+1.0$ (Long).
    *   `SELL` (1): Target position $-1.0$ (Short).
    *   `HOLD` (2): Retain current position (no turnover or rebalancing).
    *   `FLAT` (3): Target position $0.0$ (Close to Cash).
*   **Realistic Cost Modeling**:
    *   Adverse slippage: $\text{fill\_price} = p \cdot (1 + \text{slippage\_rate} \cdot \text{sign}(\Delta \text{units}))$.
    *   Transaction fees: $\text{fee} = \text{fee\_rate} \cdot |\Delta \text{units}| \cdot \text{fill\_price}$.
    *   Position reversal (Long $\leftrightarrow$ Short) incurs two fills (closing old position, opening new position) with two fees and slippages.
    *   Terminal liquidation: Open positions are liquidated at current mark close upon episode end or data end.
*   **Observation Space**: Flattened state window + portfolio state vector: $\text{Box}(W \times F + 2)$ of dtype `float32`, containing the technical feature matrix, current position sign, and relative return.
*   **Cost-Aware Buy-and-Hold Baseline**:
    *   Evaluates an aligned benchmark executing `BUY` at the first evaluation execution open and `HOLD` thereafter.
    *   Runs with identical capital parameters (`initial_equity`, `fee_rate`, `slippage_rate`, `position_size`, `leverage`) and terminal mark-close liquidation.
    *   Evaluated in a separate rule-free environment (`n_consecutive_window=None`) for immediate step-0 entry and pairwise step validation.
    *   Summary metrics CSV (`_Portfolio-Simulator_metrics.csv`) records `buy_and_hold_final_equity`, `buy_and_hold_cumulative_return`, and `excess_cumulative_return` (`cumulative_return - buy_and_hold_cumulative_return`).
    *   Step records CSV (`_Portfolio-Simulator_eval_cumul_pnls.csv`) records `buy_and_hold_position`, `buy_and_hold_equity`, `buy_and_hold_cumulative_pnl`, and `buy_and_hold_cumulative_return`.
*   **Legacy Artifact Incompatibility**: Legacy scenario artifacts (`Market-Orders`, `Market-Limit Orders`) are incompatible inputs, not migration candidates, and are ignored by evaluation and visualization tools. Canonical outputs strictly follow the `Portfolio-Simulator` scenario.
*   **Retrain Requirement**: Due to action space change (`Discrete(4)`) and observation space expansion (`Box(W*F+2,)`), models trained on Phase 1 are incompatible and must be retrained.
*   **Phase 2 Scope & Assumptions**:
    *   Single-asset Long/Short execution with fixed units between rebalances.
    *   No maintenance margin or intrabar liquidation.
    *   No funding rate mechanics or order book volume impact.

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
*   `env_config`: Environment parameters like `initial_equity`, `fee_rate`, `slippage_rate`, `position_size`, `leverage`, and `timeframe_size`.
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
Verify Gymnasium and Stable-Baselines3 environment compatibility and portfolio lifecycle behavior:

```bash
python validate_environment.py
```

### 4. Train Agents
Train RL agents on the processed datasets using the `Portfolio-Simulator` scenario.

Standard TraderNet:
```bash
python train.py
```

Smurf-wrapped TraderNet (Phase 2 canonical setup):
```bash
python train_smurf.py
```

*   Checkpoints are saved to: `database/storage/checkpoints/experiments/{tradernet,smurf}/{agent}/{dataset}/Portfolio-Simulator/`
*   Training logs (TensorBoard) are included in the checkpoint directories.
*   Results (Evaluation metrics) are saved to `experiments/{tradernet,smurf}/{agent}/{dataset}_Portfolio-Simulator.csv`.

### 5. Evaluate Agents
Evaluate trained agents on unseen data.

Standard evaluation:
```bash
python eval.py
```

Integrated / Hybrid evaluation (TraderNet + Smurf):
```bash
python integrated.py
```

Standalone evaluation writes aligned buy-and-hold baseline and excess-return fields under `experiments/tradernet/`. Integrated evaluation writes its existing strategy artifacts under `experiments/integrated/`; `plot_results.py` combines canonical standard, integrated, and deduplicated baseline curves.

Visualize results:
```bash
python plot_results.py
```
`plot_results.py` consumes only canonical `*_Portfolio-Simulator_eval_cumul_pnls.csv` files and plots standard, integrated, and deduplicated buy-and-hold curves.
### 6. Run Tests
Run automated unit and integration tests:

```bash
pytest
```

## Project Structure

*   `agents/`: Implementation of RL agents (PPO, DQN).
*   `environments/`: Custom trading environment, portfolio simulator, and environment factory.
*   `database/`: Scripts for data management, preprocessing, and dataset utilities.
*   `metrics/`: Performance metrics calculations (Cumulative Return, Loss Rate, Sharpe, Sortino, Drawdown).
*   `rules/`: Safety rules like N-Consecutive.
*   `tests/`: Unit and integration tests.
*   `config.py`: Global configuration file.
*   `train.py`: Main training script for standard TraderNet under Portfolio-Simulator scenario.
*   `train_smurf.py`: Training script for Smurf scenario.
*   `eval.py`: Evaluation script for standalone models.
*   `integrated.py`: Evaluation script for hybrid / integrated models.
*   `validate_environment.py`: Synthetic SB3/lifecycle environment validator.
*   `download_olhcv.py`: Data downloader.

## Disclaimer

**Important Note**: This software is for **educational and research purposes only**. It is not a commercial product and should not be used as financial advice. Trading cryptocurrencies involves significant risk.
