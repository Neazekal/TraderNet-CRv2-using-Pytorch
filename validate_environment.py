import sys
import numpy as np
from stable_baselines3.common.env_checker import check_env
from database.datasets.utils import MarketData
from environments.actions import Action
from environments.factory import build_trading_environment


def main():
    try:
        # 1. Generate synthetic normalized float32 states and positive OHLC
        num_samples = 200
        num_features = 19
        timeframe_len = 12
        num_transitions = num_samples - timeframe_len
        episode_steps = 50

        # States: (T + 1, W, F)
        states = np.linspace(0.1, 0.9, (num_transitions + 1) * timeframe_len * num_features, dtype=np.float32).reshape(
            num_transitions + 1, timeframe_len, num_features
        )
        opens = np.linspace(100.0, 200.0, num_transitions, dtype=np.float64)
        closes = opens + 1.0
        timestamps = np.array([f"2023-01-01 {i:04d}:00:00" for i in range(num_transitions)])

        market_data = MarketData(
            states=states,
            execution_opens=opens,
            mark_closes=closes,
            timestamps=timestamps,
        )

        # 2. Check SB3 environment compatibility
        env = build_trading_environment(
            market_data=market_data,
            episode_steps=episode_steps,
            initial_equity=10000.0,
            fee_rate=0.007,
            slippage_rate=0.0005,
            position_size=1.0,
            leverage=1.0,
            n_consecutive_window=3,
        )
        check_env(env, warn=True)

        # 3. Lifecycle rollout covering all 4 actions, cap, liquidation, and reset
        rollout_env = build_trading_environment(
            market_data=market_data,
            episode_steps=episode_steps,
            initial_equity=10000.0,
            fee_rate=0.007,
            slippage_rate=0.0005,
            position_size=1.0,
            leverage=1.0,
            n_consecutive_window=3,
        )

        obs, info = rollout_env.reset()
        assert obs.shape == (timeframe_len * num_features + 2,), f"Unexpected obs shape: {obs.shape}"
        assert obs.dtype == np.float32, f"Obs dtype must be float32, got {obs.dtype}"
        assert info['equity'] == 10000.0, f"Expected initial equity 10000, got {info['equity']}"

        actions_sequence = [Action.BUY.value, Action.SELL.value, Action.HOLD.value, Action.FLAT.value]

        # Rollout 2 episodes
        for ep in range(2):
            done = False
            step_count = 0
            while not done:
                action = actions_sequence[step_count % len(actions_sequence)]
                next_obs, reward, terminated, truncated, step_info = rollout_env.step(action)

                assert isinstance(reward, float) and np.isfinite(reward), f"Reward must be finite float, got {reward}"
                assert isinstance(terminated, bool), f"Terminated must be bool, got {type(terminated)}"
                assert isinstance(truncated, bool), f"Truncated must be bool, got {type(truncated)}"
                assert next_obs.shape == (timeframe_len * num_features + 2,), f"Obs shape mismatch: {next_obs.shape}"
                assert next_obs.dtype == np.float32, f"Obs dtype mismatch: {next_obs.dtype}"

                # Verify all info fields exist and are finite
                required_info_fields = [
                    'step_index', 'timestamp', 'requested_action', 'effective_action',
                    'position_before', 'position', 'execution_open', 'mark_close',
                    'units', 'trade_count', 'turnover', 'fee_paid', 'slippage_cost',
                    'net_pnl', 'step_return', 'equity', 'cumulative_pnl',
                    'cumulative_return', 'terminal_liquidation', 'bankrupt'
                ]
                for field in required_info_fields:
                    assert field in step_info, f"Missing '{field}' in step info"
                    val = step_info[field]
                    if isinstance(val, (int, float, np.floating, np.integer)):
                        assert np.isfinite(val), f"Field '{field}' has non-finite value: {val}"

                step_count += 1
                done = terminated or truncated

            assert step_count == episode_steps, f"Episode length {step_count} != episode_steps {episode_steps}"
            assert truncated is True, "Cap termination should set truncated=True"
            assert step_info['terminal_liquidation'] is True or step_info['position'] == 0.0, "Terminal step must liquidate"

            obs, info = rollout_env.reset()

        # Check data-end termination using start_index
        short_env = build_trading_environment(
            market_data=market_data,
            episode_steps=50,
            initial_equity=10000.0,
            fee_rate=0.007,
            slippage_rate=0.0005,
            position_size=1.0,
            leverage=1.0,
            n_consecutive_window=None,
        )
        # Start at transition num_transitions - 5 -> should terminate in 5 steps at data end
        short_env.reset(options={'start_index': num_transitions - 5})
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = short_env.step(Action.BUY.value)
            done = terminated or truncated
            steps += 1
        assert steps == 5, f"Expected 5 steps to data end, got {steps}"
        assert terminated is True, "Data end termination must set terminated=True"
        assert truncated is False, "Data end termination must set truncated=False"

        # Check metrics
        metrics = rollout_env.get_metrics()
        assert isinstance(metrics, tuple), f"get_metrics must return tuple, got {type(metrics)}"
        assert len(metrics) == 5, f"Expected 5 metrics, got {len(metrics)}"
        for metric in metrics:
            assert len(metric.episode_metrics) == 2, (
                f"Metric {metric.name} expected 2 episode metrics, got {len(metric.episode_metrics)}"
            )

        print("Environment passed SB3 compatibility and portfolio lifecycle checks!")
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
