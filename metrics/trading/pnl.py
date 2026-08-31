from metrics.metric import Metric


class CumulativeReturn(Metric):
    def __init__(self):
        super().__init__(name='Cumulative Return')
        self._wealth = 1.0

    def reset(self):
        self._wealth = 1.0

    def update(self, step_return: float):
        self._wealth *= (1.0 + step_return)

    def result(self) -> float:
        return float(self._wealth - 1.0)
