import numpy as np
from metrics.metric import Metric


class SortinoRatio(Metric):
    def __init__(self):
        super().__init__(name='Sortino')
        self._returns = []

    def reset(self):
        self._returns = []

    def update(self, step_return: float):
        self._returns.append(float(step_return))

    def result(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        arr = np.asarray(self._returns, dtype=np.float64)
        downside = np.minimum(arr, 0.0)
        downside_rms = float(np.sqrt(np.mean(downside ** 2)))
        if downside_rms == 0.0 or not np.isfinite(downside_rms):
            return 0.0
        mean_val = float(np.mean(arr))
        return float(mean_val / downside_rms)
