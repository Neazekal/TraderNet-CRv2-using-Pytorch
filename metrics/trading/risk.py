from metrics.metric import Metric


class LossRate(Metric):
    def __init__(self):
        super().__init__(name='Loss Rate')
        self._neg_count = 0
        self._nonzero_count = 0

    def reset(self):
        self._neg_count = 0
        self._nonzero_count = 0

    def update(self, step_return: float):
        if step_return < 0:
            self._neg_count += 1
            self._nonzero_count += 1
        elif step_return > 0:
            self._nonzero_count += 1

    def result(self) -> float:
        if self._nonzero_count == 0:
            return 0.0
        return float(self._neg_count / self._nonzero_count)
