import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.actions import Action
from environments.factory import build_trading_environment
from eval import eval_tradernet


class ConstantAgent:
    def __init__(self, action: int):
        self._action = action

    def predict(self, obs, deterministic=True):
        # SB3 predict returns (action_array, state)
        return np.array([self._action]), None


class CustomRewardFunction:
    def __init__(self, size=10):
        self._size = size
        # Columns: BUY, SELL, HOLD
        # Note: HOLD reward is non-zero (0.0055)
        self._rewards = np.zeros((size, 3), dtype=np.float32)
        self._rewards[:, 0] = 0.10    # BUY
        self._rewards[:, 1] = -0.10   # SELL
        self._rewards[:, 2] = 0.0055  # HOLD (positive/non-zero)

    def get_reward(self, i: int, action: int) -> float:
        return float(self._rewards[i, action])

    def get_reward_fn_shape(self):
        return self._rewards.shape

    @property
    def reward_fn(self):
        return self._rewards


def test_evaluator_uses_info_log_pnl():
    num_states = 11
    states = np.zeros((num_states, 2, 2), dtype=np.float32)
    reward_fn = CustomRewardFunction(size=num_states - 1)
    episode_steps = 5

    # NConsecutive(3): first 2 BUYs are filtered to HOLD (log_pnl = 0.0, but reward = 0.0055)
    # 3rd, 4th, 5th BUYs are passed as BUY (log_pnl = 0.10, reward = 0.10)
    def make_env():
        return build_trading_environment(
            states=states,
            reward_fn=reward_fn,
            episode_steps=episode_steps,
            n_consecutive_window=3
        )

    vec_env = DummyVecEnv([make_env])
    agent = ConstantAgent(action=Action.BUY.value)

    cumul_rewards, pnls = eval_tradernet(agent, vec_env)

    # Expected cumulative rewards:
    # 2 * 0.0055 + 3 * 0.10 = 0.011 + 0.30 = 0.311
    assert cumul_rewards == pytest.approx(0.311)

    # Expected cumulative PnLs:
    # Step 1: 0.0
    # Step 2: 0.0
    # Step 3: 0.10
    # Step 4: 0.20
    # Step 5: 0.30
    assert pnls[0] == pytest.approx(0.0)
    assert pnls[1] == pytest.approx(0.0)
    assert pnls[2] == pytest.approx(0.10)
    assert pnls[3] == pytest.approx(0.20)
    assert pnls[4] == pytest.approx(0.30)


def test_evaluator_stops_on_terminal():
    num_states = 11
    states = np.zeros((num_states, 2, 2), dtype=np.float32)
    reward_fn = CustomRewardFunction(size=num_states - 1)
    episode_steps = 4

    def make_env():
        return build_trading_environment(
            states=states,
            reward_fn=reward_fn,
            episode_steps=episode_steps,
            n_consecutive_window=None
        )

    vec_env = DummyVecEnv([make_env])
    agent = ConstantAgent(action=Action.BUY.value)

    cumul_rewards, pnls = eval_tradernet(agent, vec_env)
    assert len(pnls) == 4


def test_metric_aggregation_k_episodes_one_row():
    num_states = 21
    states = np.zeros((num_states, 2, 2), dtype=np.float32)
    reward_fn = CustomRewardFunction(size=num_states - 1)
    episode_steps = 5

    def make_env():
        return build_trading_environment(
            states=states,
            reward_fn=reward_fn,
            episode_steps=episode_steps,
            n_consecutive_window=None
        )

    vec_env = DummyVecEnv([make_env])
    agent = ConstantAgent(action=Action.BUY.value)

    # Run 2 evaluation episodes (K=2)
    for _ in range(2):
        eval_tradernet(agent, vec_env)

    base_env = vec_env.envs[0].unwrapped
    metrics = base_env.get_metrics()

    # Each metric should have 2 registered episodes
    for m in metrics:
        assert len(m.episode_metrics) == 2

    # Aggregate metrics across K episodes into a single-row DataFrame
    metrics_dict = {
        'average_returns': [0.5],
        **{
            m.name: [float(np.mean(m.episode_metrics))] if len(m.episode_metrics) > 0 else [float(m.result())]
            for m in metrics
        }
    }
    df = pd.DataFrame(metrics_dict)
    assert len(df) == 1
    assert 'Cumulative Log Returns' in df.columns
    assert 'Sharpe' in df.columns
