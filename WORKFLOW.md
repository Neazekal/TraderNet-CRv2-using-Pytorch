# Project Workflow

This document outlines the standard workflow for training and evaluating agents in the TraderNet-CRv2 project (PyTorch version) under Phase 2 realistic portfolio simulation.

## 1. Data Preparation
Before training, ensure you have the necessary datasets.
- **Script**: `database/build_dataset.py` (or `download_olhcv.py` followed by data preprocessing).
- **Input**: Raw OHLCV data from Binance.
- **Output**: Processed CSV files in `database/storage/datasets/` (e.g., `DOGEUSDT.csv`).
- **Dataset Contract**:
  - Validated required columns: `date`, `open`, `close`, and technical feature columns.
  - Causal temporal split: Scaler is fit only on training raw rows `[:N - E]`, then applied to the entire dataset.
  - `MarketData` dataclass guarantees causal next-open alignment without lookahead bias.

## 2. Environment Validation
Before training, verify Gymnasium and Stable-Baselines3 environment compatibility and portfolio simulation mechanics:
- **Command**:
  ```bash
  python validate_environment.py
  ```
- **Success Criteria**: Passes `check_env`, validates all 4 discrete actions (`BUY`, `SELL`, `HOLD`, `FLAT`), tests episode cap, data-end, bankruptcy flags, and terminal liquidation.

## 3. Training
Train RL agents (PPO, DQN) on the processed datasets using the `Portfolio-Simulator` scenario.
- **Scripts**:
  - `train.py`: Main training script for TraderNet agents.
  - `train_smurf.py`: Smurf scenario runner (Phase 2 canonical setup; gating specialization is scheduled for Phase 3).
- **Configuration**:
  - Configure environment parameters in `config.py`: `initial_equity`, `fee_rate`, `slippage_rate`, `position_size`, `leverage`, `timeframe_size`, `n_consecutive_window`.
  - Configure agent hyperparameters in `config.agent_config`.
- **Command**:
  ```bash
  python train.py
  ```
- **Outputs**:
  - Checkpoints: `database/storage/checkpoints/experiments/{tradernet,smurf}/{agent_name}/{dataset_name}/Portfolio-Simulator/model.zip`
  - TensorBoard Logs: Located within the checkpoint directory.
  - Metrics CSV: `experiments/{tradernet,smurf}/{agent_name}/{dataset_name}_Portfolio-Simulator.csv`

## 4. Evaluation
Evaluate trained agents on the evaluation split using the updated 4-action portfolio environment.
- **Scripts**:
  - `eval.py`: Standalone model evaluation.
  - `integrated.py`: Hybrid / integrated evaluation combining TraderNet trade direction with Smurf HOLD gating.
- **Commands**:
  ```bash
  python eval.py
  python integrated.py
  ```
- **Outputs**:
  - Metrics CSV: `experiments/{tradernet,integrated}/{agent_name}/{dataset_name}_Portfolio-Simulator_metrics.csv`
    - Contains: `average_returns`, `final_equity`, `cumulative_return`, `Cumulative Return`, `Loss Rate`, `Sharpe`, `Sortino`, `Maximum Drawdown`.
  - PnL / Step Records CSV: `experiments/{tradernet,integrated}/{agent_name}/{dataset_name}_Portfolio-Simulator_eval_cumul_pnls.csv`
    - Contains: `step_index`, `timestamp`, `position`, `equity`, `cumulative_pnl`, `cumulative_return`, `fee_paid`, `slippage_cost`, `turnover`, etc.

## 5. Visualization
Visualize the evaluation performance across different agents and strategies.
- **Script**: `plot_results.py`.
- **Input**: `_eval_cumul_pnls.csv` files from `experiments/tradernet/` and `experiments/integrated/`.
- **Command**:
  ```bash
  python plot_results.py
  ```
- **Output**: `experiments/comparison_plot.png`.

## 6. Execution Semantics & Invariants
- **Causal Execution**: Action based on observations up to bar $t$ fills at the opening price of bar $t+1$ (`execution_open`) and is marked to market at the closing price of bar $t+1$ (`mark_close`).
- **4 Actions**:
  - `BUY` (0): $+1.0$ (Long)
  - `SELL` (1): $-1.0$ (Short)
  - `HOLD` (2): Retain current position (no turnover or rebalancing)
  - `FLAT` (3): $0.0$ (Close to cash)
- **Cost Modeling**:
  - Fill price with adverse slippage: $p \cdot (1 + \text{slippage\_rate} \cdot \text{sign}(\Delta \text{units}))$.
  - Transaction fees: $\text{fee\_rate} \cdot |\Delta \text{units}| \cdot \text{fill\_price}$.
  - Reversals (Long $\leftrightarrow$ Short) execute as two fills with two fees and slippages.
  - Terminal liquidation closes open positions at final mark close.
- **Retrain Requirement**: All Phase 1 models are incompatible due to action space expansion (`Discrete(4)`) and flattened observation vector (`Box(W*F+2,)`) and must be retrained.
- **Phase 2 Scope & Assumptions**: Single-asset Long/Short, fixed units between rebalances, no maintenance margin / intrabar liquidation, no funding rates, no order book volume impact.
