import numpy as np
from metrics.metric import Metric


class SharpeRatio(Metric):
    def __init__(self):
        super().__init__(name='Sharpe')
        self._returns = []

    def reset(self):
        self._returns = []

    def update(self, step_return: float):
        self._returns.append(float(step_return))

    def result(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        arr = np.asarray(self._returns, dtype=np.float64)
        std_val = float(np.std(arr, ddof=1))
        if std_val == 0.0 or not np.isfinite(std_val):
            return 0.0
        mean_val = float(np.mean(arr))
        return float(mean_val / std_val)
