import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv
from database.datasets.utils import MarketData
from environments.actions import Action
from environments.factory import build_trading_environment
from eval import eval_tradernet, eval_buy_and_hold


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


def test_eval_buy_and_hold_behavior_and_accounting():
    mdata = _create_eval_market_data(num_states=6)
    episode_steps = 5
    initial_equity = 10000.0
    fee_rate = 0.005
    slippage_rate = 0.001

    def make_env():
        return build_trading_environment(
            market_data=mdata,
            episode_steps=episode_steps,
            initial_equity=initial_equity,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=1.0,
            leverage=1.0,
            n_consecutive_window=None,
        )

    vec_env = DummyVecEnv([make_env])
    cumul_rewards, step_records = eval_buy_and_hold(vec_env)

    assert isinstance(cumul_rewards, float)
    assert len(step_records) == episode_steps

    # Step 0: requested BUY, effective BUY, immediate position entry
    assert step_records[0]['requested_action'] == Action.BUY.value
    assert step_records[0]['effective_action'] == Action.BUY.value
    assert step_records[0]['position'] == 1.0
    assert step_records[0]['turnover'] > 0.0
    assert step_records[0]['fee_paid'] > 0.0
    assert step_records[0]['slippage_cost'] > 0.0

    # Intermediate steps (1 to 3): requested HOLD, effective HOLD, zero turnover
    for t in range(1, episode_steps - 1):
        assert step_records[t]['requested_action'] == Action.HOLD.value
        assert step_records[t]['effective_action'] == Action.HOLD.value
        assert step_records[t]['position'] == 1.0
        assert step_records[t]['turnover'] == 0.0
        assert step_records[t]['fee_paid'] == 0.0
        assert step_records[t]['slippage_cost'] == 0.0

    # Terminal step (step 4): requested HOLD, effective HOLD, terminal liquidation
    terminal_rec = step_records[-1]
    assert terminal_rec['requested_action'] == Action.HOLD.value
    assert terminal_rec['effective_action'] == Action.HOLD.value
    assert terminal_rec['terminal_liquidation'] is True
    assert terminal_rec['position'] == 0.0
    assert terminal_rec['units'] == 0.0
    assert terminal_rec['turnover'] > 0.0
    assert terminal_rec['fee_paid'] > 0.0
    assert terminal_rec['slippage_cost'] > 0.0

    # Cost-aware final accounting
    total_fees = sum(r['fee_paid'] for r in step_records)
    total_slippage = sum(r['slippage_cost'] for r in step_records)
    assert total_fees > 0.0
    assert total_slippage > 0.0
    assert terminal_rec['cumulative_pnl'] == pytest.approx(terminal_rec['equity'] - initial_equity)
    assert terminal_rec['cumulative_return'] == pytest.approx((terminal_rec['equity'] - initial_equity) / initial_equity)


def test_eval_buy_and_hold_rule_isolation():
    mdata = _create_eval_market_data(num_states=6)
    episode_steps = 5

    # 1. Baseline env with no rule: enters at step one (index 0)
    def make_bh_env():
        return build_trading_environment(
            market_data=mdata,
            episode_steps=episode_steps,
            n_consecutive_window=None,
        )

    bh_vec_env = DummyVecEnv([make_bh_env])
    _, bh_records = eval_buy_and_hold(bh_vec_env)

    assert bh_records[0]['requested_action'] == Action.BUY.value
    assert bh_records[0]['effective_action'] == Action.BUY.value
    assert bh_records[0]['position'] == 1.0

    # 2. Configured NConsecutive(3) env with BUY-only agent: delays repeated BUY
    def make_rule_env():
        return build_trading_environment(
            market_data=mdata,
            episode_steps=episode_steps,
            n_consecutive_window=3,
        )

    rule_vec_env = DummyVecEnv([make_rule_env])
    agent = ConstantAgent(action=Action.BUY.value)
    _, rule_records = eval_tradernet(agent, rule_vec_env)

    # Step 0 & 1: delayed by rule to HOLD, position remains 0
    assert rule_records[0]['effective_action'] == Action.HOLD.value
    assert rule_records[0]['position'] == 0.0
    assert rule_records[1]['effective_action'] == Action.HOLD.value
    assert rule_records[1]['position'] == 0.0
    # Step 2: passes NConsecutive(3) filter -> BUY
    assert rule_records[2]['effective_action'] == Action.BUY.value
    assert rule_records[2]['position'] == 1.0
