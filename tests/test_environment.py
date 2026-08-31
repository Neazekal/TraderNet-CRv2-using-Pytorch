import numpy as np
import pytest
from environments.actions import Action
from environments.factory import build_trading_environment
from rules.nconsecutive import NConsecutive


class MockRewardFunction:
    def __init__(self, size=20):
        self._size = size
        # Columns: BUY, SELL, HOLD
        self._rewards = np.ones((size, 3), dtype=np.float32)
        self._rewards[:, 0] = 0.05   # BUY reward
        self._rewards[:, 1] = -0.05  # SELL reward
        self._rewards[:, 2] = -0.01  # HOLD reward

    def get_reward(self, i: int, action: int) -> float:
        return float(self._rewards[i, action])

    def get_reward_fn_shape(self):
        return self._rewards.shape

    @property
    def reward_fn(self):
        return self._rewards


def test_sequential_episodes_and_cursor_management():
    # 21 states -> 20 transitions (indices 0..19)
    num_states = 21
    states = np.zeros((num_states, 4, 2), dtype=np.float32)
    for i in range(num_states):
        states[i, :, :] = i  # Mark state with index

    reward_fn = MockRewardFunction(size=num_states - 1)
    episode_steps = 5

    env = build_trading_environment(
        states=states,
        reward_fn=reward_fn,
        episode_steps=episode_steps,
        n_consecutive_window=None
    )

    # Episode 1
    obs, _ = env.reset()
    assert obs[0, 0] == 0.0  # Starts at state 0

    for step_i in range(episode_steps):
        next_obs, reward, terminated, truncated, info = env.step(Action.BUY.value)
        if step_i == episode_steps - 1:
            assert terminated is True
        else:
            assert terminated is False

    # Episode 2: cursor should continue at state 5
    obs, _ = env.reset()
    assert obs[0, 0] == 5.0

    for step_i in range(episode_steps):
        next_obs, reward, terminated, truncated, info = env.step(Action.BUY.value)
        if step_i == episode_steps - 1:
            assert terminated is True
        else:
            assert terminated is False

    # Episode 3: cursor should continue at state 10
    obs, _ = env.reset()
    assert obs[0, 0] == 10.0


def test_dataset_end_wrap():
    # 11 states -> 10 transitions
    num_states = 11
    states = np.zeros((num_states, 2, 2), dtype=np.float32)
    for i in range(num_states):
        states[i, :, :] = i

    reward_fn = MockRewardFunction(size=num_states - 1)
    # Episode steps 6: ep1 does 6 steps (0->6), ep2 does 4 steps (6->10 == end)
    episode_steps = 6

    env = build_trading_environment(
        states=states,
        reward_fn=reward_fn,
        episode_steps=episode_steps,
        n_consecutive_window=None
    )

    # Episode 1: steps 0 to 6
    env.reset()
    for _ in range(episode_steps):
        _, _, terminated, _, _ = env.step(Action.BUY.value)
    assert terminated is True

    # Episode 2: steps 6 to 10 (num_states - 1 = 10)
    obs, _ = env.reset()
    assert obs[0, 0] == 6.0

    terminated = False
    step_count = 0
    while not terminated:
        _, _, terminated, _, _ = env.step(Action.BUY.value)
        step_count += 1

    assert step_count == 4  # Terminated at dataset end (10 - 6 = 4 steps)
    # On dataset end, cursor wraps to 0
    obs, _ = env.reset()
    assert obs[0, 0] == 0.0


def test_metrics_lifecycle_and_history():
    num_states = 21
    states = np.zeros((num_states, 2, 2), dtype=np.float32)
    reward_fn = MockRewardFunction(size=num_states - 1)
    episode_steps = 5

    env = build_trading_environment(
        states=states,
        reward_fn=reward_fn,
        episode_steps=episode_steps,
        n_consecutive_window=None
    )

    metrics = env.get_metrics()
    assert isinstance(metrics, tuple)
    assert len(metrics) == 5

    # Before running, metric history is empty
    for m in metrics:
        assert len(m.episode_metrics) == 0

    # Run episode 1
    env.reset()
    for _ in range(episode_steps):
        env.step(Action.BUY.value)

    # After episode 1 terminal step, each metric should have exactly 1 history entry
    for m in metrics:
        assert len(m.episode_metrics) == 1

    # Reset for episode 2: history must be preserved, accumulator reset
    env.reset()
    for m in metrics:
        assert len(m.episode_metrics) == 1

    # Run episode 2
    for _ in range(episode_steps):
        env.step(Action.BUY.value)

    # After episode 2, each metric should have 2 history entries
    for m in metrics:
        assert len(m.episode_metrics) == 2


def test_rule_filter_and_info_log_pnl():
    num_states = 21
    states = np.zeros((num_states, 2, 2), dtype=np.float32)
    reward_fn = MockRewardFunction(size=num_states - 1)
    episode_steps = 10

    # Environment with NConsecutive(3)
    env = build_trading_environment(
        states=states,
        reward_fn=reward_fn,
        episode_steps=episode_steps,
        n_consecutive_window=3
    )

    env.reset()

    # Step 1: BUY -> filtered to HOLD -> info action is HOLD (2), log_pnl is 0.0
    _, reward, _, _, info1 = env.step(Action.BUY.value)
    assert info1['action'] == Action.HOLD.value
    assert info1['log_pnl'] == 0.0

    # Step 2: BUY -> filtered to HOLD -> info action is HOLD (2), log_pnl is 0.0
    _, reward, _, _, info2 = env.step(Action.BUY.value)
    assert info2['action'] == Action.HOLD.value
    assert info2['log_pnl'] == 0.0

    # Step 3: BUY -> passed as BUY (0) -> log_pnl is reward (0.05)
    _, reward, _, _, info3 = env.step(Action.BUY.value)
    assert info3['action'] == Action.BUY.value
    assert info3['log_pnl'] == pytest.approx(0.05)
