"""Portfolio risk controls for paper orders and fear-driven position management."""

from __future__ import annotations

import math
from dataclasses import dataclass

import config
from indicators.fear_engine import FearRegime, FearState


@dataclass(frozen=True)
class PullbackPlan:
    """Quantity to sell and the number of candles to lock new entries."""

    quantity_to_sell: int
    cooldown_candles: int
    reason: str


class RiskManager:
    """Calculate sizing, enforce a daily loss limit and manage fear responses."""

    def __init__(
        self,
        risk_per_trade_pct: float = config.RISK_PER_TRADE_PCT,
        initial_capital: float = config.INITIAL_CAPITAL,
    ) -> None:
        if not 0 < risk_per_trade_pct <= 100:
            raise ValueError("risk_per_trade_pct must be in (0, 100]")
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.risk_per_trade_pct = risk_per_trade_pct
        self.initial_capital = float(initial_capital)
        self.cooldown_remaining = 0
        self.daily_lock_active = False
        self.capital_lock_active = False
        self._high_fear_reduced: set[str] = set()

    def calculate_quantity(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
        fear_state: FearState,
    ) -> int:
        """Size a long position by stop risk, fear multiplier and available cash."""
        if capital <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0
        stop_distance = entry_price - stop_price
        if stop_distance <= 0 or fear_state.position_size_multiplier <= 0:
            return 0
        risk_budget = capital * (self.risk_per_trade_pct / 100.0)
        risk_budget *= min(1.0, fear_state.position_size_multiplier)
        risk_quantity = math.floor(risk_budget / stop_distance)
        cash_quantity = math.floor(capital / entry_price)
        return max(0, min(risk_quantity, cash_quantity))

    @staticmethod
    def ratchet_trailing_stop(
        current_price: float,
        atr_value: float,
        fear_state: FearState,
        current_stop: float | None = None,
    ) -> float:
        """Tighten a long trailing stop without ever lowering an existing stop."""
        if current_price <= 0 or atr_value <= 0:
            raise ValueError("current_price and atr_value must be positive")
        if fear_state.regime is FearRegime.EXTREME_PANIC:
            candidate = current_price
        else:
            candidate = current_price - atr_value * fear_state.stop_loss_atr_multiplier
        return max(candidate, current_stop) if current_stop is not None else candidate

    @staticmethod
    def daily_loss_triggered(daily_pnl: float, starting_equity: float) -> bool:
        """Return true after realized + unrealized PnL breaches the daily loss limit."""
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        loss_limit = starting_equity * config.DAILY_MAX_LOSS_PCT / 100.0
        return daily_pnl <= -loss_limit

    def can_open_position(self, daily_pnl: float, starting_equity: float) -> bool:
        """Whether new entries are currently allowed by cooldown and daily loss."""
        return not self.capital_lock_active and not self.daily_lock_active and self.cooldown_remaining == 0 and not self.daily_loss_triggered(
            daily_pnl, starting_equity
        )

    def enforce_portfolio_kill_switch(self, broker, current_equity: float) -> list:
        """Flatten positions and lock entries when liquidation would reach the hard floor."""
        floor = self.initial_capital * (1.0 - config.MAX_DRAWDOWN_PCT / 100.0)
        projected_equity = getattr(broker, "projected_liquidation_equity", current_equity)
        if projected_equity > floor:
            return []
        self.capital_lock_active = True
        orders = []
        for ticker, position in list(broker.get_positions().items()):
            order = broker.sell(ticker, position.quantity, position.current_price)
            if order.status == "FILLED":
                orders.append(order)
        return orders

    def enforce_daily_loss(self, broker, daily_pnl: float, starting_equity: float) -> list:
        """Latch the daily circuit breaker and close all positions at supplied marks.

        When a price map is not provided, the broker's latest recorded mark is
        used for each position. The lock stays active until ``reset_session``.
        """
        if not self.daily_loss_triggered(daily_pnl, starting_equity):
            return []
        self.daily_lock_active = True
        orders = []
        for ticker, position in list(broker.get_positions().items()):
            order = broker.sell(ticker, position.quantity, position.current_price)
            if order.status == "FILLED":
                orders.append(order)
        return orders

    def reset_session(self) -> None:
        """Clear daily loss lock at the start of a new trading session."""
        self.daily_lock_active = False
        self.cooldown_remaining = 0

    @staticmethod
    def plan_fear_pullback(open_quantity: int, fear_state: FearState) -> PullbackPlan:
        """Choose a half-position pullback or full emergency square-off."""
        if open_quantity <= 0:
            return PullbackPlan(0, 0, "No open quantity")
        if fear_state.regime is FearRegime.EXTREME_PANIC:
            return PullbackPlan(
                open_quantity,
                max(fear_state.cooldown_candles, config.PANIC_COOLDOWN_CANDLES),
                "Extreme fear: emergency full square-off",
            )
        if fear_state.regime is FearRegime.HIGH_FEAR_PULLBACK:
            return PullbackPlan(
                math.ceil(open_quantity / 2), 0,
                "High fear: reduce open position by approximately 50%",
            )
        return PullbackPlan(0, 0, "Fear regime does not require a pullback")

    def apply_fear_pullback(self, broker, ticker: str, market_price: float, fear_state: FearState):
        """Execute the risk plan through a broker adapter and arm panic cooldown."""
        position = broker.get_position(ticker)
        if position is None:
            self._high_fear_reduced.discard(ticker)
            return None
        if fear_state.regime not in {FearRegime.HIGH_FEAR_PULLBACK, FearRegime.EXTREME_PANIC}:
            self._high_fear_reduced.discard(ticker)
        plan = self.plan_fear_pullback(position.quantity, fear_state)
        if fear_state.regime is FearRegime.HIGH_FEAR_PULLBACK and ticker in self._high_fear_reduced:
            return None
        if plan.quantity_to_sell <= 0:
            return None
        order = broker.sell(ticker, plan.quantity_to_sell, market_price)
        if fear_state.regime is FearRegime.HIGH_FEAR_PULLBACK and order.status == "FILLED":
            self._high_fear_reduced.add(ticker)
        if fear_state.regime is FearRegime.EXTREME_PANIC and order.status == "FILLED":
            self.cooldown_remaining = plan.cooldown_candles
        return order

    def advance_candle(self) -> int:
        """Advance panic cooldown by one candle and return remaining candles."""
        self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
        return self.cooldown_remaining

