import numpy as np
import pytest
from database.datasets.utils import MarketData
from environments.actions import Action
from environments.factory import build_trading_environment


def _create_mock_market_data(num_states=21, w=4, f=2):
    num_transitions = num_states - 1
    states = np.zeros((num_states, w, f), dtype=np.float32)
    for i in range(num_states):
        states[i, :, :] = float(i)

    opens = np.linspace(100.0, 120.0, num_transitions, dtype=np.float64)
    closes = np.linspace(101.0, 121.0, num_transitions, dtype=np.float64)
    timestamps = np.array([f"2023-01-01 {i:02d}:00:00" for i in range(num_transitions)])

    return MarketData(
        states=states,
        execution_opens=opens,
        mark_closes=closes,
        timestamps=timestamps,
    )


def test_observation_and_action_spaces():
    mdata = _create_mock_market_data(num_states=21, w=4, f=2)
    env = build_trading_environment(
        market_data=mdata,
        episode_steps=5,
    )

    expected_obs_dim = 4 * 2 + 2  # 10
    assert env.observation_space.shape == (expected_obs_dim,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 4

    obs, info = env.reset()
    assert obs.shape == (expected_obs_dim,)
    assert obs.dtype == np.float32
    assert info['equity'] == 10000.0
    assert info['position'] == 0.0


def test_sequential_episodes_and_cursor_management():
    # 21 states -> 20 transitions (indices 0..19)
    mdata = _create_mock_market_data(num_states=21, w=4, f=2)
    episode_steps = 5

    env = build_trading_environment(
        market_data=mdata,
        episode_steps=episode_steps,
        n_consecutive_window=None,
    )

    # Episode 1
    obs, info = env.reset()
    assert info['step_index'] == 0
    # First element of state 0 is 0.0
    assert obs[0] == 0.0

    for step_i in range(episode_steps):
        next_obs, reward, terminated, truncated, info = env.step(Action.BUY.value)
        if step_i == episode_steps - 1:
            assert truncated is True
            assert terminated is False
        else:
            assert truncated is False
            assert terminated is False

    # Episode 2: cursor should continue at state 5
    obs, info = env.reset()
    assert info['step_index'] == 5
    assert obs[0] == 5.0

    for step_i in range(episode_steps):
        next_obs, reward, terminated, truncated, info = env.step(Action.BUY.value)
        if step_i == episode_steps - 1:
            assert truncated is True
        else:
            assert truncated is False

    # Episode 3: cursor should continue at state 10
    obs, info = env.reset()
    assert info['step_index'] == 10
    assert obs[0] == 10.0


def test_dataset_end_wrap():
    # 11 states -> 10 transitions
    mdata = _create_mock_market_data(num_states=11, w=2, f=2)
    episode_steps = 6

    env = build_trading_environment(
        market_data=mdata,
        episode_steps=episode_steps,
        n_consecutive_window=None,
    )

    # Episode 1: steps 0 to 6
    env.reset()
    for _ in range(episode_steps):
        _, _, terminated, truncated, _ = env.step(Action.BUY.value)
    assert truncated is True

    # Episode 2: steps 6 to 10 (total 4 steps to data end)
    obs, info = env.reset()
    assert info['step_index'] == 6
    assert obs[0] == 6.0

    done = False
    step_count = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(Action.BUY.value)
        done = terminated or truncated
        step_count += 1

    assert step_count == 4
    assert terminated is True
    assert truncated is False

    # On dataset end, cursor wraps to 0
    obs, info = env.reset()
    assert info['step_index'] == 0
    assert obs[0] == 0.0


def test_reset_options_start_index():
    mdata = _create_mock_market_data(num_states=21, w=2, f=2)
    env = build_trading_environment(
        market_data=mdata,
        episode_steps=5,
    )

    # Reset with specific start_index
    obs, info = env.reset(options={'start_index': 12})
    assert info['step_index'] == 12
    assert obs[0] == 12.0

    # Invalid start_index raises ValueError
    with pytest.raises(ValueError, match="start_index"):
        env.reset(options={'start_index': -1})
    with pytest.raises(ValueError, match="start_index"):
        env.reset(options={'start_index': 20})  # max valid is 19


def test_metrics_lifecycle_and_history():
    mdata = _create_mock_market_data(num_states=21, w=2, f=2)
    episode_steps = 5

    env = build_trading_environment(
        market_data=mdata,
        episode_steps=episode_steps,
        n_consecutive_window=None,
    )

    metrics = env.get_metrics()
    assert isinstance(metrics, tuple)
    assert len(metrics) == 5

    # Initially empty history
    for m in metrics:
        assert len(m.episode_metrics) == 0

    # Run episode 1
    env.reset()
    for _ in range(episode_steps):
        env.step(Action.BUY.value)

    for m in metrics:
        assert len(m.episode_metrics) == 1

    # Reset for episode 2: history preserved
    env.reset()
    for m in metrics:
        assert len(m.episode_metrics) == 1

    # Run episode 2
    for _ in range(episode_steps):
        env.step(Action.BUY.value)

    for m in metrics:
        assert len(m.episode_metrics) == 2


def test_rule_filter_and_info_contract():
    mdata = _create_mock_market_data(num_states=21, w=2, f=2)
    episode_steps = 10

    env = build_trading_environment(
        market_data=mdata,
        episode_steps=episode_steps,
        n_consecutive_window=3,
    )

    env.reset()

    # Step 1: requested BUY -> filtered to HOLD (2)
    _, reward, _, _, info1 = env.step(Action.BUY.value)
    assert info1['requested_action'] == Action.BUY.value
    assert info1['effective_action'] == Action.HOLD.value
    assert info1['position'] == 0.0
    assert info1['trade_count'] == 0

    # Step 2: requested BUY -> filtered to HOLD (2)
    _, reward, _, _, info2 = env.step(Action.BUY.value)
    assert info2['requested_action'] == Action.BUY.value
    assert info2['effective_action'] == Action.HOLD.value
    assert info2['position'] == 0.0

    # Step 3: requested BUY -> passed as BUY (0)
    _, reward, _, _, info3 = env.step(Action.BUY.value)
    assert info3['requested_action'] == Action.BUY.value
    assert info3['effective_action'] == Action.BUY.value
    assert info3['position'] == 1.0
    assert info3['trade_count'] == 1

    # Check all info keys are present and finite
    required_keys = [
        'step_index', 'timestamp', 'requested_action', 'effective_action',
        'position_before', 'position', 'execution_open', 'mark_close',
        'units', 'trade_count', 'turnover', 'fee_paid', 'slippage_cost',
        'net_pnl', 'step_return', 'equity', 'cumulative_pnl',
        'cumulative_return', 'terminal_liquidation', 'bankrupt'
    ]
    for k in required_keys:
        assert k in info3
        val = info3[k]
        if isinstance(val, (int, float, np.floating, np.integer)):
            assert np.isfinite(val)


def test_data_end_precedes_cap_when_dataset_is_shorter():
    mdata = _create_mock_market_data(num_states=4, w=2, f=2)
    env = build_trading_environment(
        market_data=mdata,
        episode_steps=5,
        n_consecutive_window=None,
    )

    env.reset()
    done = False
    step_count = 0
    while not done:
        _, _, terminated, truncated, info = env.step(Action.BUY.value)
        done = terminated or truncated
        step_count += 1

    assert step_count == 3
    assert terminated is True
    assert truncated is False
    assert info['terminal_liquidation'] is True
