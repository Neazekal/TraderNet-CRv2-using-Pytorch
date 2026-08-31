import numpy as np
import pytest
from metrics.trading.pnl import CumulativeReturn
from metrics.trading.risk import LossRate
from metrics.trading.sharpe import SharpeRatio
from metrics.trading.sortino import SortinoRatio
from metrics.trading.drawdown import MaximumDrawdown


def test_cumulative_return_metric():
    cr = CumulativeReturn()
    assert cr.name == 'Cumulative Return'
    assert cr.result() == 0.0

    # Step 1: +10%
    cr.update(0.10)
    assert cr.result() == pytest.approx(0.10)

    # Step 2: -5% -> (1.10 * 0.95) - 1 = 1.045 - 1 = 0.045
    cr.update(-0.05)
    assert cr.result() == pytest.approx(0.045)

    cr.register()
    assert len(cr.episode_metrics) == 1
    assert cr.episode_metrics[0] == pytest.approx(0.045)

    # Reset
    cr.reset()
    assert cr.result() == 0.0
    assert len(cr.episode_metrics) == 1


def test_loss_rate_metric():
    lr = LossRate()
    assert lr.name == 'Loss Rate'
    assert lr.result() == 0.0

    # All zero returns -> 0.0
    lr.update(0.0)
    assert lr.result() == 0.0

    # 1 win, 1 loss, 1 zero
    lr.update(0.05)
    lr.update(-0.02)
    # loss rate: 1 / 2 = 0.5
    assert lr.result() == pytest.approx(0.5)

    lr.update(-0.03)
    # loss rate: 2 / 3 = 0.6666667
    assert lr.result() == pytest.approx(2.0 / 3.0)


def test_sharpe_ratio_metric():
    sr = SharpeRatio()
    assert sr.name == 'Sharpe'
    assert sr.result() == 0.0

    # Less than 2 samples -> 0.0
    sr.update(0.05)
    assert sr.result() == 0.0

    # Constant returns (std = 0) -> 0.0
    sr.update(0.05)
    assert sr.result() == 0.0

    # Varied returns: [0.02, 0.04, 0.06]
    sr.reset()
    returns = [0.02, 0.04, 0.06]
    for r in returns:
        sr.update(r)

    arr = np.array(returns)
    expected_mean = np.mean(arr)
    expected_std = np.std(arr, ddof=1)
    expected_sharpe = expected_mean / expected_std

    assert sr.result() == pytest.approx(expected_sharpe)


def test_sortino_ratio_metric():
    so = SortinoRatio()
    assert so.name == 'Sortino'
    assert so.result() == 0.0

    # Less than 2 samples -> 0.0
    so.update(0.05)
    assert so.result() == 0.0

    # All positive returns -> downside RMS = 0 -> 0.0
    so.update(0.02)
    assert so.result() == 0.0

    # Mixed returns
    so.reset()
    returns = [0.05, -0.02, 0.04, -0.01]
    for r in returns:
        so.update(r)

    arr = np.array(returns)
    downside = np.minimum(arr, 0.0)
    downside_rms = np.sqrt(np.mean(downside ** 2))
    expected_sortino = np.mean(arr) / downside_rms

    assert so.result() == pytest.approx(expected_sortino)


def test_maximum_drawdown_metric():
    md = MaximumDrawdown()
    assert md.name == 'Maximum Drawdown'
    assert md.result() == 0.0

    # Up 20% (wealth 1.2, peak 1.2, dd 0)
    md.update(0.20)
    assert md.result() == pytest.approx(0.0)

    # Down 10% (wealth 1.08, peak 1.2, dd (1.2 - 1.08)/1.2 = 0.1)
    md.update(-0.10)
    assert md.result() == pytest.approx(0.10)

    # Up 30% (wealth 1.08 * 1.3 = 1.404, peak 1.404, dd max still 0.1)
    md.update(0.30)
    assert md.result() == pytest.approx(0.10)

    # Down 50% (wealth 0.702, peak 1.404, dd (1.404 - 0.702)/1.404 = 0.5)
    md.update(-0.50)
    assert md.result() == pytest.approx(0.50)

    # Bankruptcy (wealth <= 0 -> max_dd = 1.0)
    md.reset()
    md.update(-1.0)
    assert md.result() == pytest.approx(1.0)
