import numpy as np
import pytest
from database.datasets.utils import MarketData
from environments.actions import Action
from environments.portfolio import PortfolioSimulator


def _build_simple_market_data(opens, closes):
    num_t = len(opens)
    states = np.zeros((num_t + 1, 5, 2), dtype=np.float32)
    timestamps = np.array([f"2023-01-01 {i:02d}:00:00" for i in range(num_t)])
    return MarketData(
        states=states,
        execution_opens=np.asarray(opens, dtype=np.float64),
        mark_closes=np.asarray(closes, dtype=np.float64),
        timestamps=timestamps,
    )


def test_portfolio_constructor_validation():
    mdata = _build_simple_market_data([100.0], [105.0])

    # Valid
    sim = PortfolioSimulator(mdata, initial_equity=1000.0, fee_rate=0.005, slippage_rate=0.0001, position_size=0.5, leverage=2.0)
    assert sim.initial_equity == 1000.0
    assert sim.fee_rate == 0.005
    assert sim.slippage_rate == 0.0001
    assert sim.position_size == 0.5
    assert sim.leverage == 2.0

    # Invalid initial_equity
    with pytest.raises(ValueError, match="initial_equity"):
        PortfolioSimulator(mdata, initial_equity=0.0)
    with pytest.raises(ValueError, match="initial_equity"):
        PortfolioSimulator(mdata, initial_equity=-100.0)
    with pytest.raises(ValueError, match="initial_equity"):
        PortfolioSimulator(mdata, initial_equity=True)

    # Invalid fee_rate
    with pytest.raises(ValueError, match="fee_rate"):
        PortfolioSimulator(mdata, fee_rate=-0.01)
    with pytest.raises(ValueError, match="fee_rate"):
        PortfolioSimulator(mdata, fee_rate=1.0)

    # Invalid slippage_rate
    with pytest.raises(ValueError, match="slippage_rate"):
        PortfolioSimulator(mdata, slippage_rate=-0.001)
    with pytest.raises(ValueError, match="slippage_rate"):
        PortfolioSimulator(mdata, slippage_rate=1.0)

    # Invalid position_size
    with pytest.raises(ValueError, match="position_size"):
        PortfolioSimulator(mdata, position_size=0.0)
    with pytest.raises(ValueError, match="position_size"):
        PortfolioSimulator(mdata, position_size=1.5)

    # Invalid leverage
    with pytest.raises(ValueError, match="leverage"):
        PortfolioSimulator(mdata, leverage=0.0)
    with pytest.raises(ValueError, match="leverage"):
        PortfolioSimulator(mdata, leverage=-1.0)


def test_portfolio_buy_hold_flat_math():
    opens = [100.0, 105.0, 110.0]
    closes = [102.0, 108.0, 112.0]
    mdata = _build_simple_market_data(opens, closes)

    initial_eq = 10000.0
    fee_rate = 0.01
    slip_rate = 0.002
    sim = PortfolioSimulator(
        market_data=mdata,
        initial_equity=initial_eq,
        fee_rate=fee_rate,
        slippage_rate=slip_rate,
        position_size=1.0,
        leverage=1.0,
    )

    # Step 0: BUY
    # gross_notional = 10000.0
    # opening_fill_price = 100.0 * (1 + 0.002) = 100.2
    # new_units = 10000.0 / 100.2 = 99.8003992015968
    # fee = 0.01 * 10000.0 = 100.0
    # slippage_cost = 99.8003992015968 * 0.2 = 19.96007984031936
    # cash = 10000.0 - 99.8003992015968 * 100.2 - 100.0 = -100.0
    # raw_equity = -100.0 + 99.8003992015968 * 102.0 = 10079.640718562874
    res0 = sim.step(Action.BUY.value, Action.BUY.value, is_terminal_step=False)

    expected_fill_price = 100.2
    expected_units = 10000.0 / 100.2
    expected_fee = 100.0
    expected_slip = expected_units * (expected_fill_price - 100.0)
    expected_cash = -100.0
    expected_equity = expected_cash + expected_units * 102.0

    assert res0.position == 1.0
    assert res0.units == pytest.approx(expected_units)
    assert res0.fee_paid == pytest.approx(expected_fee)
    assert res0.slippage_cost == pytest.approx(expected_slip)
    assert res0.equity == pytest.approx(expected_equity)
    assert res0.trade_count == 1
    assert res0.turnover == pytest.approx(10000.0)
    assert res0.net_pnl == pytest.approx(expected_equity - initial_eq)

    # Step 1: HOLD
    # No new trades, units remain same
    # cash remains -100.0
    # raw_equity = -100.0 + units * 108.0
    res1 = sim.step(Action.HOLD.value, Action.HOLD.value, is_terminal_step=False)
    expected_equity_1 = expected_cash + expected_units * 108.0

    assert res1.position == 1.0
    assert res1.units == pytest.approx(expected_units)
    assert res1.fee_paid == 0.0
    assert res1.slippage_cost == 0.0
    assert res1.trade_count == 1
    assert res1.turnover == 0.0
    assert res1.equity == pytest.approx(expected_equity_1)

    # Step 2: FLAT (Terminal)
    # Close at open 110.0:
    # delta_units_close = -units
    # fill_price_close = 110.0 * (1 - 0.002) = 109.78
    # fee_close = 0.01 * units * 109.78
    # slip_close = units * (110.0 - 109.78)
    # cash_after_close = -100.0 - (-units) * 109.78 - fee_close = -100.0 + units * 109.78 * 0.99
    # units = 0.0, position = 0.0
    res2 = sim.step(Action.FLAT.value, Action.FLAT.value, is_terminal_step=True)
    expected_fill_close = 110.0 * (1 - 0.002)
    expected_fee_close = 0.01 * expected_units * expected_fill_close
    expected_slip_close = expected_units * (110.0 - expected_fill_close)
    expected_cash_close = expected_cash + expected_units * expected_fill_close - expected_fee_close

    assert res2.position == 0.0
    assert res2.units == 0.0
    assert res2.fee_paid == pytest.approx(expected_fee_close)
    assert res2.slippage_cost == pytest.approx(expected_slip_close)
    assert res2.equity == pytest.approx(expected_cash_close)
    assert res2.trade_count == 2
    assert sim.done is True


def test_portfolio_reverse_trade_two_fills():
    opens = [100.0, 110.0]
    closes = [105.0, 108.0]
    mdata = _build_simple_market_data(opens, closes)

    sim = PortfolioSimulator(
        market_data=mdata,
        initial_equity=10000.0,
        fee_rate=0.005,
        slippage_rate=0.001,
        position_size=1.0,
        leverage=1.0,
    )

    # Step 0: BUY (+1.0)
    res0 = sim.step(Action.BUY.value, Action.BUY.value, is_terminal_step=False)
    assert res0.trade_count == 1
    assert res0.position == 1.0
    units_long = res0.units

    # Step 1: SELL (-1.0) (Reversal)
    # Sub-step A: close long position (1 fill)
    # Sub-step B: open short position (1 fill)
    # Total 2 fills in this step -> trade_count becomes 3
    res1 = sim.step(Action.SELL.value, Action.SELL.value, is_terminal_step=False)
    assert res1.position == -1.0
    assert res1.units < 0.0
    assert res1.trade_count == 3  # 1 from step 0 + 2 from step 1


def test_portfolio_terminal_liquidation():
    opens = [100.0, 110.0]
    closes = [105.0, 108.0]
    mdata = _build_simple_market_data(opens, closes)

    sim = PortfolioSimulator(
        market_data=mdata,
        initial_equity=10000.0,
        fee_rate=0.005,
        slippage_rate=0.001,
        position_size=1.0,
        leverage=1.0,
    )

    # Step 0: BUY (+1.0)
    sim.step(Action.BUY.value, Action.BUY.value, is_terminal_step=False)

    # Step 1: HOLD, but is_terminal_step=True
    # Simulator should liquidate open units at mark_close (108.0)
    res1 = sim.step(Action.HOLD.value, Action.HOLD.value, is_terminal_step=True)
    assert res1.terminal_liquidation is True
    assert res1.units == 0.0
    assert res1.position == 0.0
    assert sim.done is True


def test_portfolio_bankruptcy_recovery():
    # Extreme price crash causing equity <= 0
    opens = [100.0, 10.0]
    closes = [1.0, 0.5]
    mdata = _build_simple_market_data(opens, closes)

    sim = PortfolioSimulator(
        market_data=mdata,
        initial_equity=10000.0,
        fee_rate=0.01,
        slippage_rate=0.001,
        position_size=1.0,
        leverage=1.0,
    )

    # Step 0: BUY at 100.0, price crashes to 1.0 at mark close
    # units = 10000 / 100.1 = 99.9
    # cash = -100
    # raw_equity = -100 + 99.9 * 1.0 = -0.1 <= 0 -> Bankruptcy!
    res0 = sim.step(Action.BUY.value, Action.BUY.value, is_terminal_step=False)
    assert res0.bankrupt is True
    assert res0.equity == 0.0
    assert res0.units == 0.0
    assert res0.position == 0.0
    assert res0.step_return == -1.0
    assert res0.cumulative_return == -1.0
    assert sim.done is True

    # Step after done must raise RuntimeError
    with pytest.raises(RuntimeError, match="done"):
        sim.step(Action.BUY.value, Action.BUY.value)

    # Reset recovers state
    sim.reset(start_index=0)
    assert sim.equity == 10000.0
    assert sim.bankrupt is False
    assert sim.done is False


def test_portfolio_flat_price_round_trip_cost_comparison():
    # Flat price series where all opens and closes are identical
    opens = [100.0, 100.0]
    closes = [100.0, 100.0]
    mdata = _build_simple_market_data(opens, closes)
    initial_equity = 10000.0

    # 1. Zero cost simulator
    sim_zero = PortfolioSimulator(
        market_data=mdata,
        initial_equity=initial_equity,
        fee_rate=0.0,
        slippage_rate=0.0,
        position_size=1.0,
        leverage=1.0,
    )
    res0_zero = sim_zero.step(Action.BUY.value, Action.BUY.value, is_terminal_step=False)
    res1_zero = sim_zero.step(Action.HOLD.value, Action.HOLD.value, is_terminal_step=True)

    assert res0_zero.fee_paid == 0.0
    assert res0_zero.slippage_cost == 0.0
    assert res1_zero.fee_paid == 0.0
    assert res1_zero.slippage_cost == 0.0
    # Zero fee/slippage round trip ends exactly at initial equity
    assert res1_zero.equity == pytest.approx(initial_equity)
    assert res1_zero.cumulative_pnl == pytest.approx(0.0)
    assert res1_zero.cumulative_return == pytest.approx(0.0)

    # 2. Identical nonzero-cost simulator
    sim_cost = PortfolioSimulator(
        market_data=mdata,
        initial_equity=initial_equity,
        fee_rate=0.005,
        slippage_rate=0.001,
        position_size=1.0,
        leverage=1.0,
    )
    res0_cost = sim_cost.step(Action.BUY.value, Action.BUY.value, is_terminal_step=False)
    res1_cost = sim_cost.step(Action.HOLD.value, Action.HOLD.value, is_terminal_step=True)

    # Nonzero fees and slippage on both entry and terminal liquidation
    assert res0_cost.fee_paid > 0.0
    assert res0_cost.slippage_cost > 0.0
    assert res1_cost.fee_paid > 0.0
    assert res1_cost.slippage_cost > 0.0

    total_fees = res0_cost.fee_paid + res1_cost.fee_paid
    total_slippage = res0_cost.slippage_cost + res1_cost.slippage_cost
    assert total_fees > 0.0
    assert total_slippage > 0.0

    # Negative net PnL and lower final equity due to transaction costs
    assert res1_cost.cumulative_pnl < 0.0
    assert res1_cost.net_pnl < 0.0
    assert res1_cost.equity < initial_equity
    assert res1_cost.equity < res1_zero.equity
