from typing import Optional
from environments.environment import TradingEnvironment
from metrics.trading.pnl import CumulativeLogReturn
from metrics.trading.risk import InvestmentRisk
from metrics.trading.sharpe import SharpeRatio
from metrics.trading.sortino import SortinoRatio
from metrics.trading.drawdown import MaximumDrawdown
from rules.nconsecutive import NConsecutive


def build_trading_environment(
        states,
        reward_fn,
        episode_steps: int,
        n_consecutive_window: Optional[int] = None
) -> TradingEnvironment:
    metrics = [
        CumulativeLogReturn(),
        InvestmentRisk(),
        SharpeRatio(),
        SortinoRatio(),
        MaximumDrawdown()
    ]
    rules = []
    if n_consecutive_window is not None:
        rules.append(NConsecutive(window_size=n_consecutive_window))

    env_config = {
        'states': states,
        'reward_fn': reward_fn,
        'episode_steps': episode_steps,
        'metrics': metrics,
        'rules': rules
    }
    return TradingEnvironment(env_config=env_config)
