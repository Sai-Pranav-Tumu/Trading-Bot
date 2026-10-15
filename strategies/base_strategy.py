"""Shared signal types and helpers for strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from indicators.fear_engine import FearEngine, FearState
from indicators.technicals import calculate_indicators


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    PARTIAL_PULLBACK = "PARTIAL_PULLBACK"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


@dataclass(frozen=True)
class Signal:
    ticker: str
    action: SignalType
    price: float
    timestamp: Any
    reason: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    """Base class for evaluating the latest completed OHLCV candle."""

    name = "base"

    def __init__(self, fear_engine: FearEngine | None = None) -> None:
        self.fear_engine = fear_engine or FearEngine()

    def generate_signal(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        *,
        fear_state: FearState | None = None,
        india_vix_data: pd.DataFrame | pd.Series | float | int | None = None,
        portfolio_drawdown_pct: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal:
        raise NotImplementedError

    def prepare(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        fear_state: FearState | None,
        india_vix_data: pd.DataFrame | pd.Series | float | int | None,
        portfolio_drawdown_pct: float,
    ) -> tuple[pd.DataFrame, FearState | None, Signal | None]:
        """Build indicators/fear and provide an insufficient-history HOLD signal."""
        if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
            raise ValueError("ohlcv must be a non-empty DataFrame")
        data = calculate_indicators(ohlcv)
        price = float(data["close"].iloc[-1])
        timestamp = data.index[-1]
        if len(data) < 21 or data.iloc[-1][["ema_9", "ema_21", "rsi_14", "atr_14", "vwap", "bb_lower", "bb_middle", "bb_upper"]].isna().any():
            return data, fear_state, Signal(
                ticker=ticker,
                action=SignalType.HOLD,
                price=price,
                timestamp=timestamp,
                reason="Insufficient valid indicator history",
                metadata={"strategy": self.name},
            )
        if fear_state is None:
            fear_state = self.fear_engine.compute(
                ohlcv, india_vix_data, portfolio_drawdown_pct
            )
        return data, fear_state, None

    @staticmethod
    def make_signal(
        ticker: str,
        action: SignalType,
        data: pd.DataFrame,
        reason: str,
        confidence: float,
        fear_state: FearState,
        **metadata: Any,
    ) -> Signal:
        """Create a signal with common price, timestamp and fear metadata."""
        latest = data.iloc[-1]
        return Signal(
            ticker=ticker,
            action=action,
            price=float(latest["close"]),
            timestamp=data.index[-1],
            reason=reason,
            confidence=max(0.0, min(1.0, confidence)),
            metadata={
                "strategy": "",
                "fear_score": fear_state.fear_score,
                "fear_regime": fear_state.regime.value,
                **metadata,
            },
        )


def falling_knife_reason(data: pd.DataFrame, fear_state: FearState) -> str | None:
    """Return a reason to block a long entry during a fast/extending sell-off."""
    if fear_state.velocity_drop > 0.65:
        return f"Falling-knife filter: drop velocity {fear_state.velocity_drop:.2f} exceeds 0.65"
    if len(data) < 20:
        return None
    latest = data.iloc[-1]
    candle_range = float(latest["high"] - latest["low"])
    if candle_range <= 0 or pd.isna(latest["volume"]):
        return None
    volume_average = data["volume"].tail(20).mean()
    close_near_low = (float(latest["close"]) - float(latest["low"])) / candle_range <= 0.15
    expanding_red_volume = (
        float(latest["close"]) < float(latest["open"])
        and float(latest["volume"]) > 1.2 * float(volume_average)
    )
    if close_near_low and expanding_red_volume:
        return "Falling-knife filter: close near candle low with expanding red volume"
    return None

