import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from database.datasets.utils import (
    MarketData,
    construct_timeframes,
    prepare_train_eval_dataset,
    prepare_eval_dataset,
)


def _create_sample_csv(num_rows=100, feature_names=None, eval_outlier=False, duplicate_date=False, non_mono_date=False, bad_date=False, neg_price=False, nan_price=False, nan_feature=False):
    if feature_names is None:
        feature_names = ['feat1', 'feat2']

    dates = pd.date_range('2023-01-01', periods=num_rows, freq='h').astype(str).tolist()
    if duplicate_date:
        dates[1] = dates[0]
    if non_mono_date:
        dates[5], dates[6] = dates[6], dates[5]
    if bad_date:
        dates[2] = 'not-a-valid-date'

    data = {
        'date': dates,
        'open': np.linspace(100.0, 150.0, num_rows, dtype=np.float64),
        'high': np.linspace(101.0, 151.0, num_rows, dtype=np.float64),
        'low': np.linspace(99.0, 149.0, num_rows, dtype=np.float64),
        'close': np.linspace(100.5, 150.5, num_rows, dtype=np.float64),
    }

    if neg_price:
        data['open'][10] = -5.0
    if nan_price:
        data['close'][10] = np.nan

    for idx, f in enumerate(feature_names):
        vals = np.linspace(float(idx), float(idx + 10), num_rows, dtype=np.float64)
        if eval_outlier and idx == 0:
            vals[-5:] = 999.0  # Big outlier in eval section
        if nan_feature and idx == 0:
            vals[10] = np.nan
        data[f] = vals

    df = pd.DataFrame(data)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def test_construct_timeframes():
    samples = np.linspace(0.0, 1.0, 50 * 4, dtype=np.float32).reshape(50, 4)
    timeframe_len = 5
    result = construct_timeframes(samples, timeframe_len)
    assert result.shape == (46, 5, 4)
    assert result.dtype == np.float32


def test_construct_timeframes_invalid():
    samples = np.ones((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        construct_timeframes(samples, timeframe_len=1)
    with pytest.raises(ValueError):
        construct_timeframes(samples, timeframe_len=0)
    with pytest.raises(ValueError):
        construct_timeframes(samples, timeframe_len=True)
    with pytest.raises(ValueError):
        construct_timeframes(samples, timeframe_len=15)  # 15 > 10
    with pytest.raises(ValueError):
        construct_timeframes(np.ones((10, 2, 2)), timeframe_len=5)  # 3D array


def test_market_data_dataclass_invariants():
    states = np.zeros((11, 5, 2), dtype=np.float32)
    opens = np.ones(10, dtype=np.float64)
    closes = np.ones(10, dtype=np.float64)
    stamps = np.array([f"t{i}" for i in range(10)])

    mdata = MarketData(states=states, execution_opens=opens, mark_closes=closes, timestamps=stamps)
    assert len(mdata.states) == 11
    assert len(mdata.execution_opens) == 10
    assert len(mdata.mark_closes) == 10
    assert len(mdata.timestamps) == 10

    # Wrong dtype
    with pytest.raises(ValueError):
        MarketData(states=states.astype(np.float64), execution_opens=opens, mark_closes=closes, timestamps=stamps)
    with pytest.raises(ValueError):
        MarketData(states=states, execution_opens=opens.astype(np.float32), mark_closes=closes, timestamps=stamps)
    with pytest.raises(ValueError):
        MarketData(states=states, execution_opens=opens, mark_closes=closes.astype(np.float32), timestamps=stamps)

    # Length mismatch
    with pytest.raises(ValueError):
        MarketData(states=states[:10], execution_opens=opens, mark_closes=closes, timestamps=stamps)
    with pytest.raises(ValueError):
        MarketData(states=states, execution_opens=opens[:9], mark_closes=closes, timestamps=stamps)

    # Shape, finiteness, and price invariants
    with pytest.raises(ValueError, match="shape"):
        MarketData(states=states.reshape(11, 10), execution_opens=opens, mark_closes=closes, timestamps=stamps)
    with pytest.raises(ValueError, match="1D"):
        MarketData(states=states, execution_opens=opens.reshape(10, 1), mark_closes=closes, timestamps=stamps)

    bad_states = states.copy()
    bad_states[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        MarketData(states=bad_states, execution_opens=opens, mark_closes=closes, timestamps=stamps)

    bad_opens = opens.copy()
    bad_opens[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        MarketData(states=states, execution_opens=bad_opens, mark_closes=closes, timestamps=stamps)


def test_prepare_train_eval_dataset_shapes_and_equality():
    num_rows = 100
    features = ['feat1', 'feat2']
    csv_path = _create_sample_csv(num_rows=num_rows, feature_names=features)
    W = 10
    E = 20
    T = num_rows - W  # 90
    K = T - E         # 70

    try:
        train_data, eval_data = prepare_train_eval_dataset(
            dataset_path=csv_path,
            feature_columns=features,
            timeframe_size=W,
            num_eval_samples=E,
        )

        assert train_data.states.shape == (K + 1, W, 2)
        assert train_data.states.dtype == np.float32
        assert train_data.execution_opens.shape == (K,)
        assert train_data.execution_opens.dtype == np.float64
        assert train_data.mark_closes.shape == (K,)
        assert train_data.mark_closes.dtype == np.float64
        assert train_data.timestamps.shape == (K,)

        assert eval_data.states.shape == (E + 1, W, 2)
        assert eval_data.states.dtype == np.float32
        assert eval_data.execution_opens.shape == (E,)
        assert eval_data.execution_opens.dtype == np.float64
        assert eval_data.mark_closes.shape == (E,)
        assert eval_data.mark_closes.dtype == np.float64
        assert eval_data.timestamps.shape == (E,)

        eval_only = prepare_eval_dataset(
            dataset_path=csv_path,
            feature_columns=features,
            timeframe_size=W,
            num_eval_samples=E,
        )
        np.testing.assert_allclose(eval_data.states, eval_only.states)
        np.testing.assert_allclose(eval_data.execution_opens, eval_only.execution_opens)
        np.testing.assert_allclose(eval_data.mark_closes, eval_only.mark_closes)
        assert np.array_equal(eval_data.timestamps, eval_only.timestamps)
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_transition_alignment_and_causal_split():
    num_rows = 12
    timeframe_size = 3
    num_eval_samples = 4
    dates = pd.date_range('2023-01-01', periods=num_rows, freq='h').astype(str)
    frame = pd.DataFrame({
        'date': dates,
        'open': np.arange(100.0, 100.0 + num_rows),
        'close': np.arange(200.0, 200.0 + num_rows),
        'feat1': np.arange(float(num_rows)),
        'feat2': np.arange(float(num_rows)) * 2.0,
    })
    csv_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
    frame.to_csv(csv_path, index=False)

    try:
        train_data, eval_data = prepare_train_eval_dataset(
            csv_path,
            ['feat1', 'feat2'],
            timeframe_size,
            num_eval_samples,
        )

        assert train_data.states.shape == (6, 3, 2)
        assert eval_data.states.shape == (5, 3, 2)
        np.testing.assert_allclose(train_data.states[0, -1, 0], 2.0 / 7.0)
        np.testing.assert_allclose(train_data.states[1, -1, 0], 3.0 / 7.0)
        np.testing.assert_array_equal(train_data.execution_opens, np.arange(103.0, 108.0))
        np.testing.assert_array_equal(eval_data.execution_opens, np.arange(108.0, 112.0))
        np.testing.assert_array_equal(eval_data.timestamps, dates[8:])
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_prepare_train_eval_causal_no_leakage():
    num_rows = 100
    features = ['feat1', 'feat2']
    csv_normal = _create_sample_csv(num_rows=num_rows, feature_names=features, eval_outlier=False)
    csv_outlier = _create_sample_csv(num_rows=num_rows, feature_names=features, eval_outlier=True)
    W = 10
    E = 20

    try:
        train_norm, eval_norm = prepare_train_eval_dataset(csv_normal, features, W, E)
        train_outlier, eval_outlier = prepare_train_eval_dataset(csv_outlier, features, W, E)

        # Train data must be completely identical despite massive outlier in eval raw rows
        np.testing.assert_allclose(train_norm.states, train_outlier.states)
        np.testing.assert_allclose(train_norm.execution_opens, train_outlier.execution_opens)
        np.testing.assert_allclose(train_norm.mark_closes, train_outlier.mark_closes)

        # Eval data scaled with train scaler should reflect outlier > 1.0 (unclipped)
        assert eval_outlier.states.max() > 10.0
    finally:
        if os.path.exists(csv_normal):
            os.remove(csv_normal)
        if os.path.exists(csv_outlier):
            os.remove(csv_outlier)


def test_prepare_dataset_validations():
    features = ['feat1', 'feat2']
    valid_csv = _create_sample_csv(100, features)
    dup_csv = _create_sample_csv(100, features, duplicate_date=True)
    non_mono_csv = _create_sample_csv(100, features, non_mono_date=True)
    bad_date_csv = _create_sample_csv(100, features, bad_date=True)
    neg_price_csv = _create_sample_csv(100, features, neg_price=True)
    nan_price_csv = _create_sample_csv(100, features, nan_price=True)
    nan_feat_csv = _create_sample_csv(100, features, nan_feature=True)

    try:
        # Missing feature
        with pytest.raises(ValueError, match="Missing required feature columns"):
            prepare_train_eval_dataset(valid_csv, ['unknown_feat'], 10, 20)

        # Missing required column
        df_no_close = pd.read_csv(valid_csv).drop(columns=['close'])
        tmp_no_close = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df_no_close.to_csv(tmp_no_close.name, index=False)
        tmp_no_close.close()
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                prepare_train_eval_dataset(tmp_no_close.name, features, 10, 20)
        finally:
            if os.path.exists(tmp_no_close.name):
                os.remove(tmp_no_close.name)

        # Invalid W
        with pytest.raises(ValueError, match="timeframe_size"):
            prepare_train_eval_dataset(valid_csv, features, 1, 20)
        with pytest.raises(ValueError, match="timeframe_size"):
            prepare_train_eval_dataset(valid_csv, features, True, 20)

        # Invalid E
        with pytest.raises(ValueError, match="num_eval_samples"):
            prepare_train_eval_dataset(valid_csv, features, 10, 0)
        with pytest.raises(ValueError, match="num_eval_samples"):
            prepare_train_eval_dataset(valid_csv, features, 10, 95)  # T=90, 95 > T

        # Bad timestamps
        with pytest.raises(ValueError, match="unique"):
            prepare_train_eval_dataset(dup_csv, features, 10, 20)
        with pytest.raises(ValueError, match="strictly increasing"):
            prepare_train_eval_dataset(non_mono_csv, features, 10, 20)
        with pytest.raises(ValueError, match="valid datetimes"):
            prepare_train_eval_dataset(bad_date_csv, features, 10, 20)

        # Bad prices / features
        with pytest.raises(ValueError, match="strictly positive"):
            prepare_train_eval_dataset(neg_price_csv, features, 10, 20)
        with pytest.raises(ValueError, match="non-finite"):
            prepare_train_eval_dataset(nan_price_csv, features, 10, 20)
        with pytest.raises(ValueError, match="non-finite"):
            prepare_train_eval_dataset(nan_feat_csv, features, 10, 20)

    finally:
        for p in [valid_csv, dup_csv, non_mono_csv, bad_date_csv, neg_price_csv, nan_price_csv, nan_feat_csv]:
            if os.path.exists(p):
                os.remove(p)
