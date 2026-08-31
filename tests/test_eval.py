import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv
from database.datasets.utils import MarketData
from environments.actions import Action
from environments.factory import build_trading_environment
from eval import eval_tradernet


class ConstantAgent:
    def __init__(self, action: int):
        self._action = action

    def predict(self, obs, deterministic=True):
        # SB3 predict returns (action_array, state)
        return np.array([self._action]), None


def _create_eval_market_data(num_states=11):
    num_transitions = num_states - 1
    states = np.zeros((num_states, 4, 2), dtype=np.float32)
    opens = np.linspace(100.0, 110.0, num_transitions, dtype=np.float64)
    closes = np.linspace(102.0, 112.0, num_transitions, dtype=np.float64)
    timestamps = np.array([f"2023-01-01 {i:02d}:00:00" for i in range(num_transitions)])
    return MarketData(
        states=states,
        execution_opens=opens,
        mark_closes=closes,
        timestamps=timestamps,
    )


def test_evaluator_step_records_and_schema():
    mdata = _create_eval_market_data(num_states=11)
    episode_steps = 5

    def make_env():
        return build_trading_environment(
            market_data=mdata,
            episode_steps=episode_steps,
            n_consecutive_window=3,
        )

    vec_env = DummyVecEnv([make_env])
    agent = ConstantAgent(action=Action.BUY.value)

    cumul_rewards, step_records = eval_tradernet(agent, vec_env)

    assert isinstance(cumul_rewards, float)
    assert len(step_records) == 5

    df = pd.DataFrame(step_records)
    expected_cols = [
        'step_index', 'timestamp', 'requested_action', 'effective_action',
        'position_before', 'position', 'execution_open', 'mark_close',
        'units', 'trade_count', 'turnover', 'fee_paid', 'slippage_cost',
        'net_pnl', 'step_return', 'equity', 'cumulative_pnl',
        'cumulative_return', 'terminal_liquidation', 'bankrupt'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column '{col}' in step_records DataFrame"

    # Step 0 and 1: filtered to HOLD -> position 0
    assert step_records[0]['effective_action'] == Action.HOLD.value
    assert step_records[1]['effective_action'] == Action.HOLD.value
    # Step 2, 3, 4: passed as BUY
    assert step_records[2]['effective_action'] == Action.BUY.value
    assert step_records[3]['effective_action'] == Action.BUY.value
    assert step_records[4]['effective_action'] == Action.BUY.value


def test_evaluator_stops_on_terminal():
    mdata = _create_eval_market_data(num_states=11)
    episode_steps = 4

    def make_env():
        return build_trading_environment(
            market_data=mdata,
            episode_steps=episode_steps,
            n_consecutive_window=None,
        )

    vec_env = DummyVecEnv([make_env])
    agent = ConstantAgent(action=Action.BUY.value)

    cumul_rewards, step_records = eval_tradernet(agent, vec_env)
    assert len(step_records) == 4


def test_metric_aggregation_k_episodes_one_row():
    mdata = _create_eval_market_data(num_states=21)
    episode_steps = 5

    def make_env():
        return build_trading_environment(
            market_data=mdata,
            episode_steps=episode_steps,
            n_consecutive_window=None,
        )

    vec_env = DummyVecEnv([make_env])
    agent = ConstantAgent(action=Action.BUY.value)

    # Run 2 evaluation episodes (K=2)
    for _ in range(2):
        eval_tradernet(agent, vec_env)

    base_env = vec_env.envs[0].unwrapped
    metrics = base_env.get_metrics()

    for m in metrics:
        assert len(m.episode_metrics) == 2

    metrics_dict = {
        'average_returns': [0.05],
        'final_equity': [10500.0],
        'cumulative_return': [0.05],
        **{
            m.name: [float(np.mean(m.episode_metrics))] if len(m.episode_metrics) > 0 else [float(m.result())]
            for m in metrics
        }
    }
    df = pd.DataFrame(metrics_dict)
    assert len(df) == 1
    assert 'Cumulative Return' in df.columns
    assert 'Sharpe' in df.columns
    assert 'Sortino' in df.columns
    assert 'Loss Rate' in df.columns
    assert 'Maximum Drawdown' in df.columns
