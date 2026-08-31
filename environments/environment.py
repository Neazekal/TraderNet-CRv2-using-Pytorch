import gymnasium as gym
import numpy as np
from environments.actions import Action
from metrics.metric import Metric


class TradingEnvironment(gym.Env):
    def __init__(self, env_config: dict):
        super().__init__()

        assert 'states' in env_config, 'AssertionError: Expected "states" in env_config'
        assert 'reward_fn' in env_config, 'AssertionError: Expected "reward_fn" in env_config'
        assert 'episode_steps' in env_config, 'AssertionError: Expected "episode_steps" in env_config'
        assert 'metrics' in env_config, 'AssertionError: Expected "metrics" in env_config'

        self._states = env_config['states']
        self._reward_function = env_config['reward_fn']
        self._episode_steps = env_config['episode_steps']

        self._metrics = env_config['metrics']
        self._rules = env_config.get('rules', [])

        if self._metrics is None:
            self._metrics = []
        if self._rules is None:
            self._rules = []

        self._num_states = self._states.shape[0] - 1

        assert self._num_states >= self._episode_steps, \
            'AssertionError: Not enough states are provided in the environment: ' \
            f'num_states = {self._num_states}, episode_steps = {self._episode_steps}'

        self._state_index = 0
        self._episode_step_count = 0

        # Use fixed 0-1 bounds for observation space to ensure compatibility
        # between training and evaluation environments when data is normalized.
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._states.shape[1:],
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(n=len(Action))

        assert self._states.dtype == self.observation_space.dtype, \
            f'AssertionError: Expected states to have dtype = {self.observation_space.dtype}, got {self._states.dtype}'

    @property
    def metrics(self) -> list[Metric]:
        return list(self._metrics)

    def get_metrics(self) -> tuple[Metric, ...]:
        return tuple(self._metrics)

    def update_metrics(self, log_pnl: float):
        for metric in self._metrics:
            metric.update(log_pnl=log_pnl)

    def register_metrics(self):
        for metric in self._metrics:
            metric.register()

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._episode_step_count = 0
        for metric in self._metrics:
            metric.reset()
        for rule in self._rules:
            rule.reset()
        return self._states[self._state_index], {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        effective_action = int(action)
        # Apply safety rules to filter the action
        for rule in self._rules:
            effective_action = rule.filter(effective_action)

        reward = float(self._reward_function.get_reward(i=self._state_index, action=effective_action))

        self._state_index += 1
        self._episode_step_count += 1
        next_state = self._states[self._state_index]

        terminated = False
        if self._state_index >= self._num_states:
            terminated = True
            self._state_index = 0
        elif self._episode_step_count >= self._episode_steps:
            terminated = True

        truncated = False

        log_pnl = 0.0 if effective_action == Action.HOLD.value else reward
        self.update_metrics(log_pnl=log_pnl)

        if terminated:
            self.register_metrics()

        info = {
            "action": effective_action,
            "log_pnl": log_pnl
        }
        return next_state, reward, terminated, truncated, info

    def render(self):
        print('\n--- Current State ---')
        print(self._states[self._state_index])
