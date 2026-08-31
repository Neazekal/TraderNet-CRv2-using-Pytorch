import pytest
from environments.actions import Action
from rules.nconsecutive import NConsecutive


def test_nconsecutive_n3_same_action():
    rule = NConsecutive(window_size=3)

    # Action 1: first step -> HOLD
    assert rule.filter(Action.BUY.value) == Action.HOLD.value

    # Action 2: second step -> HOLD
    assert rule.filter(Action.BUY.value) == Action.HOLD.value

    # Action 3: third step with same action -> BUY
    assert rule.filter(Action.BUY.value) == Action.BUY.value

    # Action 4: fourth step with same action -> BUY
    assert rule.filter(Action.BUY.value) == Action.BUY.value


def test_nconsecutive_mixed_sequence():
    rule = NConsecutive(window_size=3)

    # Step 1: BUY -> HOLD
    assert rule.filter(Action.BUY.value) == Action.HOLD.value

    # Step 2: BUY -> HOLD
    assert rule.filter(Action.BUY.value) == Action.HOLD.value

    # Step 3: SELL -> mixed window [BUY, BUY, SELL] -> HOLD
    assert rule.filter(Action.SELL.value) == Action.HOLD.value

    # Step 4: SELL -> mixed window [BUY, SELL, SELL] -> HOLD
    assert rule.filter(Action.SELL.value) == Action.HOLD.value

    # Step 5: SELL -> window [SELL, SELL, SELL] -> SELL
    assert rule.filter(Action.SELL.value) == Action.SELL.value


def test_nconsecutive_reset():
    rule = NConsecutive(window_size=3)

    assert rule.filter(Action.BUY.value) == Action.HOLD.value
    assert rule.filter(Action.BUY.value) == Action.HOLD.value

    # Reset clears the queue
    rule.reset()

    # After reset, we need 3 consecutive actions again
    assert rule.filter(Action.BUY.value) == Action.HOLD.value
    assert rule.filter(Action.BUY.value) == Action.HOLD.value
    assert rule.filter(Action.BUY.value) == Action.BUY.value


def test_nconsecutive_invalid_window():
    with pytest.raises(ValueError):
        NConsecutive(window_size=0)

    with pytest.raises(ValueError):
        NConsecutive(window_size=-1)

    with pytest.raises(ValueError):
        NConsecutive(window_size=1.5)

    with pytest.raises(ValueError):
        NConsecutive(window_size=True)
