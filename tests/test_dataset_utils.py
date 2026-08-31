import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from database.datasets.utils import (
    construct_timeframes,
    prepare_train_eval_dataset,
    prepare_eval_dataset,
    split_train_test
)
from environments.rewards.marketlimitorder import MarketLimitOrderRF


def _create_sample_csv(num_rows=100, feature_names=None, eval_outlier=False):
    if feature_names is None:
        feature_names = ['feat1', 'feat2']

    data = {
        'date': [f'2023-01-01 {i:02d}:00:00' for i in range(num_rows)],
        'close': np.linspace(100.0, 200.0, num_rows, dtype=np.float32),
        'high': np.linspace(101.0, 201.0, num_rows, dtype=np.float32),
        'low': np.linspace(99.0, 199.0, num_rows, dtype=np.float32),
    }
    for col in feature_names:
        data[col] = np.linspace(10.0, 20.0, num_rows, dtype=np.float32)

    if eval_outlier:
        # Put an extreme outlier in the eval state window but outside scaler-fit prefix
        data[feature_names[0]][-6] = 10000.0

    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name


def test_construct_timeframes():
    samples = np.linspace(0.0, 1.0, 50 * 4, dtype=np.float32).reshape(50, 4)
    timeframe_len = 5
    target_horizon_len = 10

    # Total expected windows = 50 - 5 - 10 + 1 = 36
    result = construct_timeframes(samples, timeframe_len, target_horizon_len)
    assert result.shape == (36, 5, 4)
    assert result.dtype == np.float32


def test_construct_timeframes_invalid():
    samples = np.ones((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        construct_timeframes(samples, timeframe_len=1, target_horizon_len=2)
    with pytest.raises(ValueError):
        construct_timeframes(samples, timeframe_len=5, target_horizon_len=0)
    with pytest.raises(ValueError):
        construct_timeframes(samples, timeframe_len=8, target_horizon_len=5)  # 8+5 > 10


def test_prepare_train_eval_dataset():
    num_rows = 100
    features = ['feat1', 'feat2']
    csv_path = _create_sample_csv(num_rows=num_rows, feature_names=features)

    W = 10
    H = 15
    E = 20
    fees = 0.005

    try:
        x_train, train_rf, x_eval, eval_rf = prepare_train_eval_dataset(
            dataset_path=csv_path,
            feature_columns=features,
            timeframe_size=W,
            target_horizon_len=H,
            num_eval_samples=E,
            fees=fees,
            reward_fn_factory=MarketLimitOrderRF
        )

        total_windows = num_rows - W - H + 1  # 100 - 10 - 15 + 1 = 76
        assert x_train.shape[0] + x_eval.shape[0] == total_windows
        assert x_eval.shape[0] == E
        assert x_train.shape[0] == total_windows - E
        assert x_train.dtype == np.float32
        assert x_eval.dtype == np.float32

        # Reward alignment
        assert train_rf.get_reward_fn_shape()[0] == x_train.shape[0]
        assert eval_rf.get_reward_fn_shape()[0] == x_eval.shape[0]

        # Check prepare_eval_dataset produces identical results
        x_eval2, eval_rf2 = prepare_eval_dataset(
            dataset_path=csv_path,
            feature_columns=features,
            timeframe_size=W,
            target_horizon_len=H,
            num_eval_samples=E,
            fees=fees,
            reward_fn_factory=MarketLimitOrderRF
        )
        np.testing.assert_allclose(x_eval, x_eval2)
        np.testing.assert_allclose(eval_rf.reward_fn, eval_rf2.reward_fn)

    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_scaler_fit_only_on_train_prefix():
    num_rows = 100
    features = ['feat1', 'feat2']
    # Outlier is in the eval partition
    csv_path = _create_sample_csv(num_rows=num_rows, feature_names=features, eval_outlier=True)

    W = 5
    H = 5
    E = 10
    fees = 0.001

    try:
        x_train, _, x_eval, _ = prepare_train_eval_dataset(
            dataset_path=csv_path,
            feature_columns=features,
            timeframe_size=W,
            target_horizon_len=H,
            num_eval_samples=E,
            fees=fees,
            reward_fn_factory=MarketLimitOrderRF
        )

        # In train data, values should be scaled properly in [0, 1]
        assert np.min(x_train) >= 0.0
        assert np.max(x_train) <= 1.0

        # Because the outlier was in eval data and NOT part of the fit, the eval scaled values for that feature
        # should exceed 1.0 significantly
        assert np.max(x_eval) > 10.0

    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_prepare_dataset_invalid_inputs():
    num_rows = 50
    features = ['feat1', 'feat2']
    csv_path = _create_sample_csv(num_rows=num_rows, feature_names=features)

    try:
        # Missing feature
        with pytest.raises(ValueError):
            prepare_train_eval_dataset(
                dataset_path=csv_path,
                feature_columns=['non_existent'],
                timeframe_size=5,
                target_horizon_len=5,
                num_eval_samples=10,
                fees=0.001,
                reward_fn_factory=MarketLimitOrderRF
            )

        # Invalid timeframe_size
        with pytest.raises(ValueError):
            prepare_train_eval_dataset(
                dataset_path=csv_path,
                feature_columns=features,
                timeframe_size=1,
                target_horizon_len=5,
                num_eval_samples=10,
                fees=0.001,
                reward_fn_factory=MarketLimitOrderRF
            )

        # Eval samples too large (leaves no train windows)
        with pytest.raises(ValueError):
            prepare_train_eval_dataset(
                dataset_path=csv_path,
                feature_columns=features,
                timeframe_size=5,
                target_horizon_len=5,
                num_eval_samples=100,  # total windows is 50-5-5+1 = 41
                fees=0.001,
                reward_fn_factory=MarketLimitOrderRF
            )

    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_prepare_eval_dataset_calls_reward_factory_once():
    num_rows = 100
    features = ['feat1', 'feat2']
    csv_path = _create_sample_csv(num_rows=num_rows, feature_names=features)

    W = 10
    H = 15
    E = 20
    fees = 0.005

    call_count = 0

    def spy_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return MarketLimitOrderRF(*args, **kwargs)

    try:
        x_eval, eval_rf = prepare_eval_dataset(
            dataset_path=csv_path,
            feature_columns=features,
            timeframe_size=W,
            target_horizon_len=H,
            num_eval_samples=E,
            fees=fees,
            reward_fn_factory=spy_factory
        )
        assert call_count == 1
        assert x_eval.shape[0] == E
        assert eval_rf.get_reward_fn_shape()[0] == E
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
