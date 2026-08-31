import numpy as np
from environments.actions import Action
from rules.rule import Rule


class NConsecutive(Rule):
    def __init__(self, window_size: int):
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError(f'window_size must be an integer greater than 0, got {window_size}')
        self._window_size = window_size
        self._actions_queue = []

    def reset(self) -> None:
        self._actions_queue.clear()

    def filter(self, action: int) -> int:
        self._actions_queue.append(int(action))
        if len(self._actions_queue) > self._window_size:
            self._actions_queue.pop(0)

        if len(self._actions_queue) < self._window_size:
            return Action.HOLD.value

        if len(set(self._actions_queue)) == 1:
            return int(action)
        return Action.HOLD.value
