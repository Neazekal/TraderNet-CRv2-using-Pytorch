from typing import Optional
from database.datasets.utils import MarketData
from environments.environment import TradingEnvironment
from metrics.trading.pnl import CumulativeReturn
from metrics.trading.risk import LossRate
from metrics.trading.sharpe import SharpeRatio
from metrics.trading.sortino import SortinoRatio
from metrics.trading.drawdown import MaximumDrawdown
from rules.nconsecutive import NConsecutive


def build_trading_environment(
    market_data: MarketData,
    episode_steps: int,
    initial_equity: float = 10000.0,
    fee_rate: float = 0.007,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
    leverage: float = 1.0,
    n_consecutive_window: Optional[int] = None,
) -> TradingEnvironment:
    metrics = [
        CumulativeReturn(),
        LossRate(),
        SharpeRatio(),
        SortinoRatio(),
        MaximumDrawdown(),
    ]
    rules = []
    if n_consecutive_window is not None:
        rules.append(NConsecutive(window_size=n_consecutive_window))

    env_config = {
        'market_data': market_data,
        'episode_steps': episode_steps,
        'initial_equity': initial_equity,
        'fee_rate': fee_rate,
        'slippage_rate': slippage_rate,
        'position_size': position_size,
        'leverage': leverage,
        'metrics': metrics,
        'rules': rules,
    }
    return TradingEnvironment(env_config=env_config)
