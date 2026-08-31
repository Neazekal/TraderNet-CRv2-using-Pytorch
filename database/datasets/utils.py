from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class MarketData:
    states: np.ndarray          # (T + 1, W, F), float32
    execution_opens: np.ndarray # (T,), float64
    mark_closes: np.ndarray     # (T,), float64
    timestamps: np.ndarray      # (T,), execution-bar timestamps

    def __post_init__(self):
        if not isinstance(self.states, np.ndarray) or self.states.dtype != np.float32:
            raise ValueError(f"states must be a float32 np.ndarray, got {getattr(self.states, 'dtype', type(self.states))}")
        if not isinstance(self.execution_opens, np.ndarray) or self.execution_opens.dtype != np.float64:
            raise ValueError(f"execution_opens must be a float64 np.ndarray, got {getattr(self.execution_opens, 'dtype', type(self.execution_opens))}")
        if not isinstance(self.mark_closes, np.ndarray) or self.mark_closes.dtype != np.float64:
            raise ValueError(f"mark_closes must be a float64 np.ndarray, got {getattr(self.mark_closes, 'dtype', type(self.mark_closes))}")
        if not isinstance(self.timestamps, np.ndarray):
            raise ValueError(f"timestamps must be a np.ndarray, got {type(self.timestamps)}")

        if self.states.ndim != 3 or self.states.shape[1] <= 0 or self.states.shape[2] <= 0:
            raise ValueError(f"states must have shape (T + 1, W, F) with W and F > 0, got {self.states.shape}")
        for name, values in (
            ("execution_opens", self.execution_opens),
            ("mark_closes", self.mark_closes),
            ("timestamps", self.timestamps),
        ):
            if values.ndim != 1:
                raise ValueError(f"{name} must be a 1D array, got shape {values.shape}")

        if len(self.execution_opens) == 0:
            raise ValueError("MarketData must contain at least one transition.")
        if not np.isfinite(self.states).all():
            raise ValueError("states must contain only finite values.")
        if not np.isfinite(self.execution_opens).all() or not np.isfinite(self.mark_closes).all():
            raise ValueError("execution_opens and mark_closes must contain only finite values.")
        if (self.execution_opens <= 0).any() or (self.mark_closes <= 0).any():
            raise ValueError("execution_opens and mark_closes must be strictly positive (> 0).")

        t_states = len(self.states)
        t_opens = len(self.execution_opens)
        t_closes = len(self.mark_closes)
        t_stamps = len(self.timestamps)
        if not (t_states == t_opens + 1 == t_closes + 1 == t_stamps + 1):
            raise ValueError(
                f"MarketData length mismatch: states={t_states}, opens={t_opens}, closes={t_closes}, timestamps={t_stamps}. "
                f"Expected len(states) == len(execution_opens) + 1 == len(mark_closes) + 1 == len(timestamps) + 1"
            )


def construct_timeframes(
    samples: np.ndarray,
    timeframe_len: int,
) -> np.ndarray:
    if not isinstance(timeframe_len, int) or isinstance(timeframe_len, bool) or timeframe_len <= 1:
        raise ValueError(f"timeframe_len is expected to be an integer greater than 1, got {timeframe_len}")

    if not isinstance(samples, np.ndarray):
        samples = np.asarray(samples)

    if samples.ndim != 2:
        raise ValueError(f"samples must be a 2D array (N, F), got shape {samples.shape}")

    n_samples = samples.shape[0]
    if n_samples < timeframe_len:
        raise ValueError(f"Cannot build timeframes: samples rows ({n_samples}) < timeframe_len ({timeframe_len})")

    num_windows = n_samples - timeframe_len + 1
    windows = [samples[i: i + timeframe_len] for i in range(num_windows)]
    return np.asarray(windows, dtype=np.float32)


def _load_and_process_dataset(
    dataset_path: str,
    feature_columns: Sequence[str],
    timeframe_size: int,
    num_eval_samples: int,
) -> tuple[MarketData, MarketData]:
    if not isinstance(timeframe_size, int) or isinstance(timeframe_size, bool) or timeframe_size <= 1:
        raise ValueError(f"timeframe_size must be an integer > 1, got {timeframe_size}")

    if not isinstance(num_eval_samples, int) or isinstance(num_eval_samples, bool) or num_eval_samples <= 0:
        raise ValueError(f"num_eval_samples must be an integer > 0, got {num_eval_samples}")

    if not isinstance(feature_columns, (list, tuple)) or len(feature_columns) == 0:
        raise ValueError("feature_columns must be a non-empty list or tuple of column names")

    df = pd.read_csv(dataset_path)

    # Validate required columns
    required_cols = ['date', 'open', 'close']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in dataset: {missing_required}")

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in dataset: {missing_features}")

    # Validate timestamps: parseable, strictly increasing, unique
    parsed_timestamps = pd.to_datetime(df['date'], errors='coerce')
    if parsed_timestamps.isna().any():
        raise ValueError("One or more timestamps in 'date' column could not be parsed as valid datetimes.")
    if not parsed_timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps in 'date' column must be strictly increasing.")
    if parsed_timestamps.duplicated().any():
        raise ValueError("Timestamps in 'date' column must be unique.")

    # Validate prices and features are finite and open/close > 0
    opens = df['open'].to_numpy(dtype=np.float64)
    closes = df['close'].to_numpy(dtype=np.float64)

    if not np.isfinite(opens).all() or not np.isfinite(closes).all():
        raise ValueError("Prices (open, close) contain non-finite values (NaN or Inf).")

    if (opens <= 0).any() or (closes <= 0).any():
        raise ValueError("Prices (open, close) must be strictly positive (> 0).")

    raw_features = df[list(feature_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(raw_features).all():
        raise ValueError("Feature columns contain non-finite values (NaN or Inf).")

    n_samples = df.shape[0]
    W = timeframe_size
    T = n_samples - W  # total transitions
    if T <= 0:
        raise ValueError(f"Dataset has insufficient rows ({n_samples}) for timeframe_size={W}.")

    E = num_eval_samples
    if not (0 < E < T):
        raise ValueError(f"num_eval_samples must satisfy 0 < num_eval_samples < {T} (total transitions), got {E}")

    K = T - E  # train transitions

    # Scaler fit only on feature raw rows [:N - E], then transform all raw rows
    # Number of raw rows used by train transitions is N - E = W + K
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_raw_rows = n_samples - E
    scaled_features = np.empty_like(raw_features, dtype=np.float32)
    scaled_features[:train_raw_rows] = scaler.fit_transform(raw_features[:train_raw_rows])
    scaled_features[train_raw_rows:] = scaler.transform(raw_features[train_raw_rows:])

    # Construct all states: shape (T + 1, W, F)
    states_all = construct_timeframes(scaled_features, timeframe_len=W)

    # Date array as 1D array of timestamps
    timestamps_all = df['date'].to_numpy()

    # Train MarketData: K transitions, K + 1 states
    train_states = states_all[:K + 1]
    train_opens = opens[W: W + K]
    train_closes = closes[W: W + K]
    train_timestamps = timestamps_all[W: W + K]

    train_data = MarketData(
        states=train_states,
        execution_opens=train_opens,
        mark_closes=train_closes,
        timestamps=train_timestamps,
    )

    # Eval MarketData: E transitions, E + 1 states
    eval_states = states_all[K:]
    eval_opens = opens[W + K:]
    eval_closes = closes[W + K:]
    eval_timestamps = timestamps_all[W + K:]

    eval_data = MarketData(
        states=eval_states,
        execution_opens=eval_opens,
        mark_closes=eval_closes,
        timestamps=eval_timestamps,
    )

    return train_data, eval_data


def prepare_train_eval_dataset(
    dataset_path: str,
    feature_columns: list | tuple,
    timeframe_size: int,
    num_eval_samples: int,
) -> tuple[MarketData, MarketData]:
    return _load_and_process_dataset(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        timeframe_size=timeframe_size,
        num_eval_samples=num_eval_samples,
    )


def prepare_eval_dataset(
    dataset_path: str,
    feature_columns: list | tuple,
    timeframe_size: int,
    num_eval_samples: int,
) -> MarketData:
    _, eval_data = _load_and_process_dataset(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        timeframe_size=timeframe_size,
        num_eval_samples=num_eval_samples,
    )
    return eval_data
