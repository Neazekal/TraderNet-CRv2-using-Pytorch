from dataclasses import dataclass
from typing import Any
import numpy as np
from database.datasets.utils import MarketData
from environments.actions import Action


@dataclass(frozen=True)
class PortfolioStepResult:
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
    reward: float
    step_return: float
    equity: float
    cumulative_pnl: float
    cumulative_return: float
    terminal_liquidation: bool
    bankrupt: bool


class PortfolioSimulator:
    def __init__(
        self,
        market_data: MarketData,
        initial_equity: float = 10000.0,
        fee_rate: float = 0.007,
        slippage_rate: float = 0.0005,
        position_size: float = 1.0,
        leverage: float = 1.0,
    ):
        if not isinstance(initial_equity, (int, float)) or isinstance(initial_equity, bool) or initial_equity <= 0:
            raise ValueError(f"initial_equity must be a positive number (> 0), got {initial_equity}")
        if not isinstance(fee_rate, (int, float)) or isinstance(fee_rate, bool) or not (0 <= fee_rate < 1):
            raise ValueError(f"fee_rate must be in [0, 1), got {fee_rate}")
        if not isinstance(slippage_rate, (int, float)) or isinstance(slippage_rate, bool) or not (0 <= slippage_rate < 1):
            raise ValueError(f"slippage_rate must be in [0, 1), got {slippage_rate}")
        if not isinstance(position_size, (int, float)) or isinstance(position_size, bool) or not (0 < position_size <= 1):
            raise ValueError(f"position_size must be in (0, 1], got {position_size}")
        if not isinstance(leverage, (int, float)) or isinstance(leverage, bool) or leverage <= 0:
            raise ValueError(f"leverage must be a positive number (> 0), got {leverage}")

        self._market_data = market_data
        self._initial_equity = float(initial_equity)
        self._fee_rate = float(fee_rate)
        self._slippage_rate = float(slippage_rate)
        self._position_size = float(position_size)
        self._leverage = float(leverage)

        # Simulator state variables
        self._cursor: int = 0
        self._cash: float = self._initial_equity
        self._units: float = 0.0
        self._position: float = 0.0
        self._equity: float = self._initial_equity
        self._cumulative_fee: float = 0.0
        self._cumulative_slippage: float = 0.0
        self._trade_count: int = 0
        self._cumulative_pnl: float = 0.0
        self._bankrupt: bool = False
        self._done: bool = False

    @property
    def market_data(self) -> MarketData:
        return self._market_data

    @property
    def initial_equity(self) -> float:
        return self._initial_equity

    @property
    def fee_rate(self) -> float:
        return self._fee_rate

    @property
    def slippage_rate(self) -> float:
        return self._slippage_rate

    @property
    def position_size(self) -> float:
        return self._position_size

    @property
    def leverage(self) -> float:
        return self._leverage

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def units(self) -> float:
        return self._units

    @property
    def position(self) -> float:
        return self._position

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def cumulative_fee(self) -> float:
        return self._cumulative_fee

    @property
    def cumulative_slippage(self) -> float:
        return self._cumulative_slippage

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def cumulative_pnl(self) -> float:
        return self._cumulative_pnl

    @property
    def bankrupt(self) -> bool:
        return self._bankrupt

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, start_index: int = 0) -> None:
        num_transitions = len(self._market_data.execution_opens)
        if not isinstance(start_index, int) or isinstance(start_index, bool) or not (0 <= start_index < num_transitions):
            raise ValueError(f"start_index must satisfy 0 <= start_index < {num_transitions}, got {start_index}")

        self._cursor = start_index
        self._cash = self._initial_equity
        self._units = 0.0
        self._position = 0.0
        self._equity = self._initial_equity
        self._cumulative_fee = 0.0
        self._cumulative_slippage = 0.0
        self._trade_count = 0
        self._cumulative_pnl = 0.0
        self._bankrupt = False
        self._done = False

    def step(
        self,
        requested_action: int,
        effective_action: int,
        is_terminal_step: bool = False,
    ) -> PortfolioStepResult:
        if self._done:
            raise RuntimeError("Cannot step simulator when episode is already done. Call reset() first.")

        k = self._cursor
        num_transitions = len(self._market_data.execution_opens)
        if k >= num_transitions:
            raise RuntimeError(f"Simulator cursor {k} exceeds total transitions {num_transitions}.")

        equity_before = self._equity
        position_before = self._position
        open_p = float(self._market_data.execution_opens[k])
        close_p = float(self._market_data.mark_closes[k])
        timestamp = self._market_data.timestamps[k]

        # Action mapping
        if effective_action == Action.BUY.value or effective_action == Action.BUY:
            target_position = 1.0
        elif effective_action == Action.SELL.value or effective_action == Action.SELL:
            target_position = -1.0
        elif effective_action == Action.HOLD.value or effective_action == Action.HOLD:
            target_position = self._position
        elif effective_action == Action.FLAT.value or effective_action == Action.FLAT:
            target_position = 0.0
        else:
            raise ValueError(f"Invalid effective_action: {effective_action}")

        fee_paid = 0.0
        slippage_cost = 0.0
        turnover = 0.0
        terminal_liquidation = False

        # Order execution at execution_open
        if target_position != self._position:
            # 1. Close old position if open
            if self._position != 0.0 and self._units != 0.0:
                delta_units_close = -self._units
                sign_close = 1.0 if delta_units_close > 0 else -1.0
                fill_price_close = open_p * (1.0 + self._slippage_rate * sign_close)
                fee_close = self._fee_rate * abs(delta_units_close) * fill_price_close
                slip_close = abs(delta_units_close) * abs(fill_price_close - open_p)
                to_close = abs(delta_units_close) * fill_price_close

                self._cash = self._cash - delta_units_close * fill_price_close - fee_close
                self._units = 0.0
                self._position = 0.0
                self._cumulative_fee += fee_close
                self._cumulative_slippage += slip_close
                self._trade_count += 1

                fee_paid += fee_close
                slippage_cost += slip_close
                turnover += to_close

                equity_after_close = self._cash
                if equity_after_close <= 0.0:
                    self._equity = 0.0
                    self._cash = 0.0
                    self._units = 0.0
                    self._position = 0.0
                    self._bankrupt = True

            # 2. Open new position if target != 0 and not bankrupt
            if not self._bankrupt and target_position != 0.0:
                equity_after_close = self._cash
                if equity_after_close > 0.0:
                    target_dir = 1.0 if target_position > 0 else -1.0
                    opening_fill_price = open_p * (1.0 + self._slippage_rate * target_dir)
                    gross_notional = equity_after_close * self._position_size * self._leverage
                    new_units = target_position * gross_notional / opening_fill_price
                    delta_units_open = new_units

                    fee_open = self._fee_rate * abs(delta_units_open) * opening_fill_price
                    slip_open = abs(delta_units_open) * abs(opening_fill_price - open_p)
                    to_open = abs(delta_units_open) * opening_fill_price

                    self._cash = self._cash - delta_units_open * opening_fill_price - fee_open
                    self._units = new_units
                    self._position = target_position
                    self._cumulative_fee += fee_open
                    self._cumulative_slippage += slip_open
                    self._trade_count += 1

                    fee_paid += fee_open
                    slippage_cost += slip_open
                    turnover += to_open

        # 3. Mark to Market after execution
        if not self._bankrupt:
            raw_equity = self._cash + self._units * close_p
            if raw_equity <= 0.0:
                self._equity = 0.0
                self._cash = 0.0
                self._units = 0.0
                self._position = 0.0
                self._bankrupt = True
            else:
                self._equity = raw_equity

        # 4. Terminal Liquidation
        if not self._bankrupt and is_terminal_step and self._units != 0.0:
            delta_units_liq = -self._units
            sign_liq = 1.0 if delta_units_liq > 0 else -1.0
            fill_price_liq = close_p * (1.0 + self._slippage_rate * sign_liq)
            fee_liq = self._fee_rate * abs(delta_units_liq) * fill_price_liq
            slip_liq = abs(delta_units_liq) * abs(fill_price_liq - close_p)
            to_liq = abs(delta_units_liq) * fill_price_liq

            self._cash = self._cash - delta_units_liq * fill_price_liq - fee_liq
            self._units = 0.0
            self._position = 0.0
            self._cumulative_fee += fee_liq
            self._cumulative_slippage += slip_liq
            self._trade_count += 1

            fee_paid += fee_liq
            slippage_cost += slip_liq
            turnover += to_liq
            terminal_liquidation = True

            raw_equity_liq = self._cash
            if raw_equity_liq <= 0.0:
                self._equity = 0.0
                self._cash = 0.0
                self._bankrupt = True
            else:
                self._equity = raw_equity_liq

        # 5. PnL and returns
        if self._bankrupt:
            self._equity = 0.0
            self._cash = 0.0
            self._units = 0.0
            self._position = 0.0
            equity_after = 0.0
            net_pnl = equity_after - equity_before
            reward = net_pnl / self._initial_equity
            step_return = -1.0
            cumulative_return = -1.0
            cumulative_pnl = -self._initial_equity
            self._cumulative_pnl = cumulative_pnl
        else:
            equity_after = self._equity
            net_pnl = equity_after - equity_before
            reward = net_pnl / self._initial_equity
            step_return = net_pnl / equity_before if equity_before > 0 else -1.0
            cumulative_pnl = self._equity - self._initial_equity
            self._cumulative_pnl = cumulative_pnl
            cumulative_return = self._equity / self._initial_equity - 1.0

        if is_terminal_step or self._bankrupt:
            self._done = True

        result = PortfolioStepResult(
            step_index=k,
            timestamp=timestamp,
            requested_action=int(requested_action),
            effective_action=int(effective_action),
            position_before=float(position_before),
            position=float(self._position),
            execution_open=float(open_p),
            mark_close=float(close_p),
            units=float(self._units),
            trade_count=int(self._trade_count),
            turnover=float(turnover),
            fee_paid=float(fee_paid),
            slippage_cost=float(slippage_cost),
            net_pnl=float(net_pnl),
            reward=float(reward),
            step_return=float(step_return),
            equity=float(self._equity),
            cumulative_pnl=float(cumulative_pnl),
            cumulative_return=float(cumulative_return),
            terminal_liquidation=bool(terminal_liquidation),
            bankrupt=bool(self._bankrupt),
        )

        self._cursor += 1
        return result
