import gymnasium as gym
import numpy as np
from database.datasets.utils import MarketData
from environments.actions import Action
from environments.portfolio import PortfolioSimulator
from metrics.metric import Metric
from rules.rule import Rule


class TradingEnvironment(gym.Env):
    def __init__(self, env_config: dict):
        super().__init__()

        required_keys = [
            'market_data', 'episode_steps', 'initial_equity',
            'fee_rate', 'slippage_rate', 'position_size',
            'leverage', 'metrics'
        ]
        for key in required_keys:
            if key not in env_config:
                raise ValueError(f"Missing required key '{key}' in env_config")

        self._market_data: MarketData = env_config['market_data']
        self._episode_steps: int = env_config['episode_steps']
        self._initial_equity: float = float(env_config['initial_equity'])
        self._fee_rate: float = float(env_config['fee_rate'])
        self._slippage_rate: float = float(env_config['slippage_rate'])
        self._position_size: float = float(env_config['position_size'])
        self._leverage: float = float(env_config['leverage'])

        self._metrics: list[Metric] = env_config['metrics'] if env_config['metrics'] is not None else []
        self._rules: list[Rule] = env_config.get('rules', [])
        if self._rules is None:
            self._rules = []

        if not isinstance(self._episode_steps, int) or isinstance(self._episode_steps, bool) or self._episode_steps <= 0:
            raise ValueError(f"episode_steps must be a positive integer, got {self._episode_steps}")

        self._num_transitions = len(self._market_data.execution_opens)
        # Episode caps may exceed available transitions so data-end termination
        # can win with a shorter final episode.

        # Build simulator
        self._simulator = PortfolioSimulator(
            market_data=self._market_data,
            initial_equity=self._initial_equity,
            fee_rate=self._fee_rate,
            slippage_rate=self._slippage_rate,
            position_size=self._position_size,
            leverage=self._leverage,
        )

        # Observation shape: flatten states[k] (W * F) + [position, return] (2) -> W * F + 2
        w = self._market_data.states.shape[1]
        f = self._market_data.states.shape[2]
        obs_dim = w * f + 2

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(n=len(Action))

        self._cursor = 0
        self._episode_step_count = 0

    @property
    def simulator(self) -> PortfolioSimulator:
        return self._simulator

    @property
    def metrics(self) -> list[Metric]:
        return list(self._metrics)

    def get_metrics(self) -> tuple[Metric, ...]:
        return tuple(self._metrics)

    def update_metrics(self, step_return: float):
        for metric in self._metrics:
            metric.update(step_return=step_return)

    def register_metrics(self):
        for metric in self._metrics:
            metric.register()

    def _get_obs(self, cursor: int, position: float, equity: float) -> np.ndarray:
        raw_state = self._market_data.states[cursor].reshape(-1)
        rel_return = equity / self._initial_equity - 1.0
        portfolio_features = np.array([position, rel_return], dtype=np.float32)
        obs = np.concatenate([raw_state, portfolio_features]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._episode_step_count = 0

        if options is not None and 'start_index' in options:
            start_idx = options['start_index']
            if not isinstance(start_idx, int) or isinstance(start_idx, bool) or not (0 <= start_idx < self._num_transitions):
                raise ValueError(
                    f"options['start_index'] must satisfy 0 <= start_index < {self._num_transitions}, got {start_idx}"
                )
            self._cursor = start_idx

        # Check cursor bounds
        if self._cursor >= self._num_transitions:
            self._cursor = 0

        self._simulator.reset(start_index=self._cursor)

        for metric in self._metrics:
            metric.reset()
        for rule in self._rules:
            rule.reset()

        obs = self._get_obs(
            cursor=self._cursor,
            position=self._simulator.position,
            equity=self._simulator.equity,
        )
        info = {
            "step_index": self._cursor,
            "position": 0.0,
            "equity": float(self._initial_equity),
            "cumulative_return": 0.0,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        requested_action = int(action)
        effective_action = requested_action

        # Apply safety rules to filter the action
        for rule in self._rules:
            effective_action = rule.filter(effective_action)

        self._episode_step_count += 1
        next_cursor = self._simulator.cursor + 1

        is_data_end = (next_cursor >= self._num_transitions)
        is_cap = (self._episode_step_count >= self._episode_steps)
        is_terminal = is_data_end or is_cap

        sim_res = self._simulator.step(
            requested_action=requested_action,
            effective_action=effective_action,
            is_terminal_step=is_terminal,
        )

        if sim_res.bankrupt:
            terminated = True
            truncated = False
        elif is_data_end:
            terminated = True
            truncated = False
        elif is_cap:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        self.update_metrics(step_return=sim_res.step_return)

        if terminated or truncated:
            self.register_metrics()
            if is_data_end:
                self._cursor = 0
            else:
                self._cursor = next_cursor
        else:
            self._cursor = next_cursor

        next_obs_cursor = self._simulator.cursor
        if next_obs_cursor > self._num_transitions:
            next_obs_cursor = self._num_transitions

        next_obs = self._get_obs(
            cursor=next_obs_cursor,
            position=sim_res.position,
            equity=sim_res.equity,
        )

        info = {
            "step_index": sim_res.step_index,
            "timestamp": sim_res.timestamp,
            "requested_action": sim_res.requested_action,
            "effective_action": sim_res.effective_action,
            "position_before": sim_res.position_before,
            "position": sim_res.position,
            "execution_open": sim_res.execution_open,
            "mark_close": sim_res.mark_close,
            "units": sim_res.units,
            "trade_count": sim_res.trade_count,
            "turnover": sim_res.turnover,
            "fee_paid": sim_res.fee_paid,
            "slippage_cost": sim_res.slippage_cost,
            "net_pnl": sim_res.net_pnl,
            "step_return": sim_res.step_return,
            "equity": sim_res.equity,
            "cumulative_pnl": sim_res.cumulative_pnl,
            "cumulative_return": sim_res.cumulative_return,
            "terminal_liquidation": sim_res.terminal_liquidation,
            "bankrupt": sim_res.bankrupt,
        }

        return next_obs, sim_res.reward, terminated, truncated, info

    def render(self):
        print(
            f"Step: {self._simulator.cursor}, Position: {self._simulator.position}, "
            f"Equity: {self._simulator.equity:.2f}, Cumulative Return: {self._simulator.equity / self._initial_equity - 1.0:.4%}"
        )
