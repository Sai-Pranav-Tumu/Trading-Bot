"""Paper broker and portfolio risk manager verification."""

import sqlite3

from indicators.fear_engine import FearRegime, FearState
from risk_and_execution.broker_adapter import PaperBroker
from risk_and_execution.risk_manager import RiskManager


def _fear(
    regime: FearRegime = FearRegime.NORMAL,
    score: float = 0.2,
    size: float = 1.0,
    stop: float = 2.0,
    cooldown: int = 0,
) -> FearState:
    return FearState(
        fear_score=score,
        regime=regime,
        position_size_multiplier=size,
        stop_loss_atr_multiplier=stop,
        vix_norm=0.0,
        velocity_drop=0.0,
        atr_spike=0.0,
        drawdown_factor=0.0,
        india_vix=14.0,
        current_atr=1.0,
        cooldown_candles=cooldown,
    )


def test_buy_pullback_trailing_stop_and_sell_are_persisted(tmp_path):
    database = tmp_path / "paper.db"
    broker = PaperBroker(initial_capital=100_000, database_path=database)
    risk = RiskManager()

    buy = broker.buy("RELIANCE.NS", 20, 100.0)
    assert buy.status == "FILLED"
    assert buy.fill_price > 100.0  # adverse buy slippage
    assert buy.fees > 0
    assert broker.get_position("RELIANCE.NS").quantity == 20

    partial = risk.apply_fear_pullback(
        broker,
        "RELIANCE.NS",
        101.0,
        _fear(FearRegime.HIGH_FEAR_PULLBACK, score=0.7, size=0.0, stop=0.75),
    )
    assert partial is not None and partial.status == "FILLED" and partial.quantity == 10

    position = broker.get_position("RELIANCE.NS")
    stop = risk.ratchet_trailing_stop(101.0, 1.0, _fear(), position.stop_price)
    tighter_stop = risk.ratchet_trailing_stop(101.0, 1.0, _fear(stop=1.2), stop)
    assert tighter_stop > stop
    broker.set_stop_price("RELIANCE.NS", tighter_stop)
    broker.update_market_price("RELIANCE.NS", tighter_stop - 0.1)
    stop_exit = broker.sell("RELIANCE.NS", 10, tighter_stop - 0.1)
    assert stop_exit.status == "FILLED"
    assert stop_exit.realized_pnl is not None
    assert broker.get_position("RELIANCE.NS") is None

    with sqlite3.connect(database) as connection:
        orders = connection.execute("SELECT side, status FROM orders ORDER BY order_id").fetchall()
        positions = connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    assert [row[0] for row in orders] == ["BUY", "SELL", "SELL"]
    assert all(row[1] == "FILLED" for row in orders)
    assert positions == 0


def test_quantity_sizing_daily_loss_guard_and_panic_squareoff(tmp_path):
    risk = RiskManager(risk_per_trade_pct=1.0)
    quantity = risk.calculate_quantity(100_000, 100.0, 95.0, _fear())
    assert quantity == 200
    assert risk.daily_loss_triggered(-2500, 100_000)
    assert not risk.can_open_position(-2500, 100_000)

    broker = PaperBroker(initial_capital=100_000, database_path=tmp_path / "panic.db")
    broker.buy("TCS.NS", 4, 100.0)
    panic = _fear(FearRegime.EXTREME_PANIC, score=0.9, size=0.0, stop=0.0, cooldown=5)
    exit_order = risk.apply_fear_pullback(broker, "TCS.NS", 95.0, panic)
    assert exit_order is not None and exit_order.quantity == 4
    assert risk.cooldown_remaining == 5
    assert risk.advance_candle() == 4

    daily_broker = PaperBroker(initial_capital=100_000, database_path=tmp_path / "daily.db")
    daily_broker.buy("INFY.NS", 2, 100.0)
    flattened = risk.enforce_daily_loss(daily_broker, -2500, 100_000)
    assert len(flattened) == 1 and flattened[0].side == "SELL"
    assert daily_broker.get_position("INFY.NS") is None
    assert not risk.can_open_position(0, 100_000)
    risk.reset_session()
    assert risk.can_open_position(0, 100_000)


def test_insufficient_cash_and_quantity_are_rejected(tmp_path):
    broker = PaperBroker(initial_capital=1000, database_path=tmp_path / "reject.db")
    rejected_buy = broker.buy("RELIANCE.NS", 100, 100.0)
    assert rejected_buy.status == "REJECTED"
    rejected_sell = broker.sell("RELIANCE.NS", 1, 100.0)
    assert rejected_sell.status == "REJECTED"


def test_portfolio_kill_switch_flattens_at_hard_floor(tmp_path):
    broker = PaperBroker(initial_capital=100_000, database_path=tmp_path / "floor.db")
    risk = RiskManager(initial_capital=100_000)
    broker.buy("HDFCBANK.NS", 900, 100.0)
    broker.update_market_price("HDFCBANK.NS", 94.0)
    exits = risk.enforce_portfolio_kill_switch(broker, broker.portfolio_value)
    assert exits and exits[0].side == "SELL"
    assert broker.get_position("HDFCBANK.NS") is None
    assert risk.capital_lock_active
    assert not risk.can_open_position(0, 100_000)
