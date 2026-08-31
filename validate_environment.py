import sys
import numpy as np
from stable_baselines3.common.env_checker import check_env
from environments.actions import Action
from environments.factory import build_trading_environment
from environments.rewards.marketlimitorder import MarketLimitOrderRF
from database.datasets.utils import construct_timeframes


def main():
    try:
        # 1. Generate synthetic normalized float32 states and monotonic OHLC
        num_samples = 200
        num_features = 19
        timeframe_len = 12
        target_horizon_len = 20
        fees = 0.007
        episode_steps = 50

        # Feature matrix normalized in [0.0, 1.0]
        raw_features = np.linspace(0.1, 0.9, num_samples * num_features, dtype=np.float32).reshape(num_samples, num_features)
        states = construct_timeframes(raw_features, timeframe_len=timeframe_len, target_horizon_len=target_horizon_len)

        # Monotonic OHLC prices
        closes = np.linspace(100.0, 200.0, num_samples, dtype=np.float32)
        highs = closes + 1.0
        lows = closes - 1.0

        reward_fn = MarketLimitOrderRF(
            timeframe_size=timeframe_len,
            target_horizon_len=target_horizon_len,
            highs=highs,
            lows=lows,
            closes=closes,
            fees_percentage=fees,
            position_size=1.0,
            leverage=1.0
        )

        # 2. Check environment compatibility with Stable-Baselines3
        env = build_trading_environment(
            states=states,
            reward_fn=reward_fn,
            episode_steps=episode_steps,
            n_consecutive_window=3
        )
        check_env(env, warn=True)

        # 3. Lifecycle rollout on fresh env
        rollout_env = build_trading_environment(
            states=states,
            reward_fn=reward_fn,
            episode_steps=episode_steps,
            n_consecutive_window=3
        )

        obs, info = rollout_env.reset()
        assert obs.shape == (timeframe_len, num_features), f"Unexpected obs shape: {obs.shape}"

        # Rollout 2 episodes
        for episode in range(2):
            done = False
            step_count = 0
            while not done:
                action = step_count % len(Action)
                next_obs, reward, terminated, truncated, step_info = rollout_env.step(action)
                assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"
                assert isinstance(terminated, bool), f"Terminated must be bool, got {type(terminated)}"
                assert isinstance(truncated, bool), f"Truncated must be bool, got {type(truncated)}"
                assert 'action' in step_info, "Info missing 'action'"
                assert 'log_pnl' in step_info, "Info missing 'log_pnl'"
                step_count += 1
                done = terminated or truncated

            assert step_count == episode_steps, f"Episode length {step_count} != episode_steps {episode_steps}"
            obs, info = rollout_env.reset()

        metrics = rollout_env.get_metrics()
        assert isinstance(metrics, tuple), f"get_metrics must return tuple, got {type(metrics)}"
        assert len(metrics) == 5, f"Expected 5 metrics, got {len(metrics)}"
        for metric in metrics:
            assert len(metric.episode_metrics) == 2, f"Metric {metric.name} expected 2 episode metrics, got {len(metric.episode_metrics)}"

        print("Environment passed SB3 compatibility and lifecycle checks!")
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
