from metrics.metric import Metric


class MaximumDrawdown(Metric):
    def __init__(self):
        super().__init__(name='Maximum Drawdown')
        self._wealth = 1.0
        self._peak = 1.0
        self._max_drawdown = 0.0

    def reset(self):
        self._wealth = 1.0
        self._peak = 1.0
        self._max_drawdown = 0.0

    def update(self, step_return: float):
        self._wealth *= (1.0 + step_return)
        if self._wealth <= 0.0:
            self._max_drawdown = 1.0
            self._wealth = 0.0
        else:
            if self._wealth > self._peak:
                self._peak = self._wealth
            dd = (self._peak - self._wealth) / self._peak
            if dd > self._max_drawdown:
                self._max_drawdown = dd

    def result(self) -> float:
        return float(self._max_drawdown)
