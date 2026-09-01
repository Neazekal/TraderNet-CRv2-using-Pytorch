from dataclasses import dataclass
from typing import Any, Callable, TypedDict
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class StepRecord(TypedDict):
    step_index: int
    timestamp: Any
    requested_action: int
    effective_action: int
    position_before: float
    position: float
    execution_open: float
    mark_close: float
    units: float
    trade_count: int
    turnover: float
    fee_paid: float
    slippage_cost: float
    net_pnl: float
    step_return: float
    equity: float
    cumulative_pnl: float
    cumulative_return: float
    terminal_liquidation: bool
    bankrupt: bool


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    steps: tuple[StepRecord, ...]


def run_episode(
    env: VecEnv,
    select_action: Callable[[np.ndarray], np.ndarray],
) -> EpisodeResult:
    obs = env.reset()
    total_reward = 0.0
    steps: list[StepRecord] = []

    while True:
        action = select_action(obs)
        obs, reward, done, info = env.step(action)

        total_reward += float(reward[0])
        steps.append(dict(info[0]))  # type: ignore[arg-type]

        if done[0]:
            break

    return EpisodeResult(total_reward=total_reward, steps=tuple(steps))
