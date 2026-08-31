import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def split_train_test(
        inputs: np.ndarray,
        targets: np.ndarray,
        num_eval_samples: int or float
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    assert inputs.shape[0] == targets.shape[0], \
        'AssertionError: Mismatch between input and target shapes: ' \
        f'Inputs: {inputs.shape[0]}, Targets: {targets.shape[0]}'

    assert (isinstance(num_eval_samples, int) and not isinstance(num_eval_samples, bool) and num_eval_samples > 0) or \
           (isinstance(num_eval_samples, float) and 0 < num_eval_samples < 1.0), \
        'AssertionError: num_eval_samples should be an integer greater than zero or a float value between (0, 1.0), '

    if isinstance(num_eval_samples, float):
        num_eval_samples = int(num_eval_samples * inputs.shape[0])

    assert 0 < num_eval_samples < inputs.shape[0], \
        f'AssertionError: num_eval_samples should be greater than 0 and less than input samples, got {num_eval_samples}'

    n_train_samples = inputs.shape[0] - num_eval_samples
    x_train = inputs[: n_train_samples]
    y_train = targets[: n_train_samples]
    x_test = inputs[n_train_samples:]
    y_test = targets[n_train_samples:]
    return x_train, y_train, x_test, y_test


def construct_timeframes(
        samples: np.ndarray,
        timeframe_len: int,
        target_horizon_len: int,
) -> np.ndarray:
    if not isinstance(timeframe_len, int) or isinstance(timeframe_len, bool) or timeframe_len <= 1:
        raise ValueError(f'timeframe_len is expected to be an integer greater than 1, got {timeframe_len}')

    if not isinstance(target_horizon_len, int) or isinstance(target_horizon_len, bool) or target_horizon_len < 1:
        raise ValueError(f'target_horizon_len is expected to be an integer >= 1, got {target_horizon_len}')

    if timeframe_len + target_horizon_len > samples.shape[0]:
        raise ValueError('Cannot build inputs, because too few samples are provided')

    return np.float32([
        samples[i: i + timeframe_len] for i in range(samples.shape[0] - target_horizon_len - timeframe_len + 1)
    ])


def _prepare_dataset_arrays(
        dataset_path: str,
        feature_columns: list or tuple,
        timeframe_size: int,
        target_horizon_len: int,
        num_eval_samples: int,
        fees: float,
        position_size: float,
        leverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if not isinstance(timeframe_size, int) or isinstance(timeframe_size, bool) or timeframe_size <= 1:
        raise ValueError(f'timeframe_size must be an integer > 1, got {timeframe_size}')

    if not isinstance(target_horizon_len, int) or isinstance(target_horizon_len, bool) or target_horizon_len < 1:
        raise ValueError(f'target_horizon_len must be an integer >= 1, got {target_horizon_len}')

    if not isinstance(num_eval_samples, int) or isinstance(num_eval_samples, bool) or num_eval_samples <= 0:
        raise ValueError(f'num_eval_samples must be an integer > 0, got {num_eval_samples}')

    if not isinstance(fees, (int, float)) or isinstance(fees, bool) or fees < 0:
        raise ValueError(f'fees must be a non-negative number, got {fees}')

    if not isinstance(position_size, (int, float)) or isinstance(position_size, bool) or position_size <= 0:
        raise ValueError(f'position_size must be a positive number, got {position_size}')

    if not isinstance(leverage, (int, float)) or isinstance(leverage, bool) or leverage <= 0:
        raise ValueError(f'leverage must be a positive number, got {leverage}')

    if not isinstance(feature_columns, (list, tuple)) or len(feature_columns) == 0:
        raise ValueError('feature_columns must be a non-empty list or tuple of column names')

    # Read dataset (let read errors propagate)
    df = pd.read_csv(dataset_path)

    # Validate columns
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f'Missing required feature columns in dataset: {missing_features}')

    required_price_columns = ['close', 'high', 'low']
    missing_prices = [col for col in required_price_columns if col not in df.columns]
    if missing_prices:
        raise ValueError(f'Missing required price columns in dataset: {missing_prices}')

    n_samples = df.shape[0]
    total_windows = n_samples - timeframe_size - target_horizon_len + 1
    if total_windows <= 0:
        raise ValueError(
            f'Dataset has insufficient rows ({n_samples}) for timeframe_size={timeframe_size} and target_horizon_len={target_horizon_len}'
        )

    num_train_inputs = total_windows - num_eval_samples
    if num_eval_samples >= total_windows or num_train_inputs <= 0:
        raise ValueError(
            f'num_eval_samples ({num_eval_samples}) must be less than total windows ({total_windows})'
        )

    # Scaling features
    samples_df = df[list(feature_columns)]
    samples = samples_df.to_numpy(dtype=np.float32)

    # Fit on every raw row used by training windows, while keeping eval-only
    # rows out of the scaler fit.
    num_train_scale_samples = num_train_inputs + timeframe_size - 1
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    samples[:num_train_scale_samples] = scaler.fit_transform(samples[:num_train_scale_samples])
    samples[num_train_scale_samples:] = scaler.transform(samples[num_train_scale_samples:])

    # Constructing timeframes
    inputs = construct_timeframes(samples, timeframe_len=timeframe_size, target_horizon_len=target_horizon_len)

    x_train = inputs[:num_train_inputs]
    x_eval = inputs[num_train_inputs:]

    closes = df['close'].to_numpy(dtype=np.float32)
    highs = df['high'].to_numpy(dtype=np.float32)
    lows = df['low'].to_numpy(dtype=np.float32)

    return x_train, x_eval, highs, lows, closes, n_samples


def prepare_train_eval_dataset(
        dataset_path: str,
        feature_columns: list or tuple,
        timeframe_size: int,
        target_horizon_len: int,
        num_eval_samples: int,
        fees: float,
        reward_fn_factory,
        position_size: float = 1.0,
        leverage: float = 1.0,
        reward_wrapper=None
) -> tuple[np.ndarray, object, np.ndarray, object]:
    x_train, x_eval, highs, lows, closes, n_samples = _prepare_dataset_arrays(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        num_eval_samples=num_eval_samples,
        fees=fees,
        position_size=position_size,
        leverage=leverage,
    )

    # Constructing reward functions
    train_reward_fn = reward_fn_factory(
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        highs=highs[: n_samples - num_eval_samples],
        lows=lows[: n_samples - num_eval_samples],
        closes=closes[: n_samples - num_eval_samples],
        fees_percentage=fees,
        position_size=position_size,
        leverage=leverage
    )
    if reward_wrapper is not None:
        train_reward_fn = reward_wrapper(train_reward_fn)

    eval_price_start = n_samples - num_eval_samples - timeframe_size - target_horizon_len + 1
    eval_reward_fn = reward_fn_factory(
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        highs=highs[eval_price_start:],
        lows=lows[eval_price_start:],
        closes=closes[eval_price_start:],
        fees_percentage=fees,
        position_size=position_size,
        leverage=leverage
    )
    if reward_wrapper is not None:
        eval_reward_fn = reward_wrapper(eval_reward_fn)

    if x_train.shape[0] != train_reward_fn.get_reward_fn_shape()[0]:
        raise ValueError(
            f'DimensionMismatch: x_train: {x_train.shape}, train_reward_fn: {train_reward_fn.get_reward_fn_shape()}'
        )
    if x_eval.shape[0] != eval_reward_fn.get_reward_fn_shape()[0]:
        raise ValueError(
            f'DimensionMismatch: x_eval: {x_eval.shape}, eval_reward_fn: {eval_reward_fn.get_reward_fn_shape()}'
        )

    return x_train, train_reward_fn, x_eval, eval_reward_fn


def prepare_eval_dataset(
        dataset_path: str,
        feature_columns: list or tuple,
        timeframe_size: int,
        target_horizon_len: int,
        num_eval_samples: int,
        fees: float,
        reward_fn_factory,
        position_size: float = 1.0,
        leverage: float = 1.0,
        reward_wrapper=None
) -> tuple[np.ndarray, object]:
    _, x_eval, highs, lows, closes, n_samples = _prepare_dataset_arrays(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        num_eval_samples=num_eval_samples,
        fees=fees,
        position_size=position_size,
        leverage=leverage,
    )

    eval_price_start = n_samples - num_eval_samples - timeframe_size - target_horizon_len + 1
    eval_reward_fn = reward_fn_factory(
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        highs=highs[eval_price_start:],
        lows=lows[eval_price_start:],
        closes=closes[eval_price_start:],
        fees_percentage=fees,
        position_size=position_size,
        leverage=leverage
    )
    if reward_wrapper is not None:
        eval_reward_fn = reward_wrapper(eval_reward_fn)

    if x_eval.shape[0] != eval_reward_fn.get_reward_fn_shape()[0]:
        raise ValueError(
            f'DimensionMismatch: x_eval: {x_eval.shape}, eval_reward_fn: {eval_reward_fn.get_reward_fn_shape()}'
        )

    return x_eval, eval_reward_fn
