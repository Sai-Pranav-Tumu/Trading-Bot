"""Paper execution engine and broker adapter interfaces."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config
from data.database import database_connection, initialize_database


@dataclass
class Position:
    ticker: str
    quantity: int
    average_price: float  # Entry price including buy-side fees, per share.
    current_price: float
    stop_price: float | None = None
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class Order:
    order_id: str
    ticker: str
    side: str
    quantity: int
    requested_price: float
    fill_price: float | None
    fees: float
    status: str
    realized_pnl: float | None = None
    reason: str = ""
    created_at: str = ""


class BrokerAdapter(ABC):
    """Interface implemented by paper and future live broker adapters."""

    @abstractmethod
    def submit_order(self, ticker: str, side: str, quantity: int, market_price: float) -> Order:
        """Submit an order and return its execution result."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Return currently open positions keyed by ticker."""


class PaperBroker(BrokerAdapter):
    """Cash-only paper broker with directional slippage, fees and SQLite audit."""

    def __init__(
        self,
        initial_capital: float = config.INITIAL_CAPITAL,
        database_path: str | Path | None = None,
        slippage_pct: float | None = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if slippage_pct is None:
            slippage_pct = config.TAX_SCHEDULE.simulated_slippage_pct
        if slippage_pct < 0 or slippage_pct >= 1:
            raise ValueError("slippage_pct must be in [0, 1)")
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.database_path = Path(database_path) if database_path else config.DATABASE_PATH
        self.slippage_pct = float(slippage_pct)
        self.positions: dict[str, Position] = {}
        initialize_database(self.database_path)
        self._restore_state()

    def submit_order(self, ticker: str, side: str, quantity: int, market_price: float) -> Order:
        return self.execute_order(ticker, side, quantity, market_price)

    def buy(self, ticker: str, quantity: int, market_price: float) -> Order:
        return self.execute_order(ticker, "BUY", quantity, market_price)

    def sell(self, ticker: str, quantity: int, market_price: float) -> Order:
        return self.execute_order(ticker, "SELL", quantity, market_price)

    def execute_order(self, ticker: str, side: str, quantity: int, market_price: float) -> Order:
        symbol = ticker.strip().upper()
        action = side.strip().upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if not symbol or quantity <= 0 or market_price <= 0:
            raise ValueError("ticker, positive quantity and positive market_price are required")

        order_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        fill_price = self._fill_price(action, market_price)
        fees = self._calculate_leg_charges(action, fill_price, quantity)
        realized_pnl: float | None = None
        rejection = ""
        position = self.positions.get(symbol)

        if action == "BUY":
            required_cash = fill_price * quantity + fees
            if required_cash > self.cash + 1e-9:
                rejection = "Insufficient paper cash"
            else:
                old_quantity = position.quantity if position else 0
                old_cost = position.average_price * old_quantity if position else 0.0
                average_cost = (old_cost + fill_price * quantity + fees) / (old_quantity + quantity)
                current_price = position.current_price if position else fill_price
                stop_price = position.stop_price if position else None
                position = Position(symbol, old_quantity + quantity, average_cost, current_price, stop_price)
                self.cash -= required_cash
        else:
            if position is None or quantity > position.quantity:
                rejection = "Insufficient open position quantity"
            else:
                realized_pnl = (fill_price - position.average_price) * quantity - fees
                self.cash += fill_price * quantity - fees
                remaining = position.quantity - quantity
                if remaining == 0:
                    position = None
                else:
                    position.quantity = remaining
                    position.unrealized_pnl = (position.current_price - position.average_price) * remaining

        status = "REJECTED" if rejection else "FILLED"
        order = Order(
            order_id=order_id,
            ticker=symbol,
            side=action,
            quantity=quantity,
            requested_price=float(market_price),
            fill_price=fill_price if status == "FILLED" else None,
            fees=fees if status == "FILLED" else 0.0,
            status=status,
            realized_pnl=realized_pnl,
            reason=rejection,
            created_at=created_at,
        )
        if status == "FILLED":
            if action == "BUY" or position is not None:
                self.positions[symbol] = position
            else:
                self.positions.pop(symbol, None)
        self._persist_order_and_position(order, self.positions.get(symbol))
        return order

    def update_market_price(self, ticker: str, market_price: float) -> Position | None:
        if market_price <= 0:
            raise ValueError("market_price must be positive")
        symbol = ticker.strip().upper()
        position = self.positions.get(symbol)
        if position is None:
            return None
        position.current_price = float(market_price)
        position.unrealized_pnl = (position.current_price - position.average_price) * position.quantity
        with database_connection(self.database_path) as connection:
            connection.execute(
                "UPDATE positions SET current_price=?, unrealized_pnl=?, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE ticker=?",
                (position.current_price, position.unrealized_pnl, symbol),
            )
        return position

    def set_stop_price(self, ticker: str, stop_price: float | None) -> None:
        symbol = ticker.strip().upper()
        position = self.positions.get(symbol)
        if position is None:
            raise ValueError(f"No open position for {symbol}")
        if stop_price is not None and stop_price <= 0:
            raise ValueError("stop_price must be positive")
        position.stop_price = stop_price
        with database_connection(self.database_path) as connection:
            connection.execute(
                "UPDATE positions SET stop_price=?, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE ticker=?",
                (stop_price, symbol),
            )

    def get_position(self, ticker: str) -> Position | None:
        return self.positions.get(ticker.strip().upper())

    def get_positions(self) -> dict[str, Position]:
        return dict(self.positions)

    @property
    def portfolio_value(self) -> float:
        return self.cash + sum(position.quantity * position.current_price for position in self.positions.values())

    @property
    def projected_liquidation_equity(self) -> float:
        """Estimate equity after immediately selling every open long position."""
        proceeds = 0.0
        for position in self.positions.values():
            fill = self._fill_price("SELL", position.current_price)
            proceeds += fill * position.quantity - self._calculate_leg_charges(
                "SELL", fill, position.quantity
            )
        return self.cash + proceeds

    def _fill_price(self, side: str, market_price: float) -> float:
        direction = 1.0 if side == "BUY" else -1.0
        return round(market_price * (1.0 + direction * self.slippage_pct), 4)

    def _calculate_leg_charges(self, side: str, fill_price: float, quantity: int) -> float:
        schedule = config.TAX_SCHEDULE
        turnover = fill_price * quantity
        brokerage = min(schedule.brokerage_flat_cap, turnover * schedule.brokerage_rate)
        exchange = round(turnover * schedule.nse_turnover_rate, 2)
        sebi = round(turnover * schedule.sebi_rate, 2)
        gst = round((brokerage + exchange + sebi) * schedule.gst_rate, 2)
        stt = round(turnover * schedule.stt_intraday_sell_rate, 2) if side == "SELL" else 0.0
        stamp = round(turnover * schedule.stamp_duty_rate, 2) if side == "BUY" else 0.0
        return round(brokerage + exchange + sebi + gst + stt + stamp, 2)

    def _restore_state(self) -> None:
        with database_connection(self.database_path) as connection:
            orders = connection.execute(
                "SELECT side, quantity, fill_price, fees FROM orders WHERE status='FILLED' ORDER BY order_id"
            ).fetchall()
            for order in orders:
                notional = float(order["fill_price"]) * int(order["quantity"])
                if order["side"] == "BUY":
                    self.cash -= notional + float(order["fees"])
                else:
                    self.cash += notional - float(order["fees"])
            rows = connection.execute("SELECT * FROM positions").fetchall()
            for row in rows:
                self.positions[row["ticker"]] = Position(
                    ticker=row["ticker"],
                    quantity=int(row["quantity"]),
                    average_price=float(row["average_price"]),
                    current_price=float(row["current_price"] or row["average_price"]),
                    stop_price=float(row["stop_price"]) if row["stop_price"] is not None else None,
                    unrealized_pnl=float(row["unrealized_pnl"] or 0),
                )

    def _persist_order_and_position(self, order: Order, position: Position | None) -> None:
        with database_connection(self.database_path) as connection:
            connection.execute(
                """INSERT INTO orders
                   (broker_order_id, ticker, side, order_type, quantity, requested_price,
                    fill_price, status, fees, realized_pnl, created_at, updated_at)
                   VALUES (?, ?, ?, 'MARKET', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order.order_id, order.ticker, order.side, order.quantity,
                    order.requested_price, order.fill_price, order.status, order.fees,
                    order.realized_pnl, order.created_at, order.created_at,
                ),
            )
            if position is None:
                connection.execute("DELETE FROM positions WHERE ticker=?", (order.ticker,))
            else:
                connection.execute(
                    """INSERT INTO positions
                       (ticker, quantity, average_price, current_price, stop_price, unrealized_pnl, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                       ON CONFLICT(ticker) DO UPDATE SET
                         quantity=excluded.quantity, average_price=excluded.average_price,
                         current_price=excluded.current_price, stop_price=excluded.stop_price,
                         unrealized_pnl=excluded.unrealized_pnl, updated_at=excluded.updated_at""",
                    (
                        position.ticker, position.quantity, position.average_price,
                        position.current_price, position.stop_price, position.unrealized_pnl,
                    ),
                )


class ZerodhaKiteAdapter(BrokerAdapter):
    """Integration placeholder; intentionally unable to send live orders yet."""

    def submit_order(self, ticker: str, side: str, quantity: int, market_price: float) -> Order:
        raise NotImplementedError("Zerodha live execution is not enabled in this paper-trading build")

    def get_positions(self) -> dict[str, Position]:
        raise NotImplementedError("Zerodha live positions are not connected")


class AngelOneSmartAPIAdapter(BrokerAdapter):
    """Integration placeholder; intentionally unable to send live orders yet."""

    def submit_order(self, ticker: str, side: str, quantity: int, market_price: float) -> Order:
        raise NotImplementedError("AngelOne live execution is not enabled in this paper-trading build")

    def get_positions(self) -> dict[str, Position]:
        raise NotImplementedError("AngelOne live positions are not connected")

