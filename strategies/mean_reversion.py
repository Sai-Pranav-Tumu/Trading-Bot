"""Oversold bounce strategy with reversal confirmation and knife-catch guards."""

from __future__ import annotations

import pandas as pd

from indicators.fear_engine import FearState
from strategies.base_strategy import BaseStrategy, Signal, SignalType, falling_knife_reason


class MeanReversionStrategy(BaseStrategy):
    """Buy confirmed lower-band reversals and exit as price mean-reverts."""

    name = "mean_reversion"

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
        data, fear, early_signal = self.prepare(
            ticker, ohlcv, fear_state, india_vix_data, portfolio_drawdown_pct
        )
        if early_signal is not None:
            return early_signal
        current = data.iloc[-1]

        if has_open_position and (
            current["close"] >= current["bb_middle"] or current["rsi_14"] >= 50
        ):
            return self.make_signal(
                ticker, SignalType.SELL, data,
                "Mean-reversion exit: price reached the middle band or RSI recovered to 50", 0.75, fear,
                strategy=self.name,
            )
        if fear.fear_score >= 0.60:
            return self.make_signal(
                ticker, SignalType.HOLD, data,
                "Fear score blocks new mean-reversion entries", 0.0, fear,
                strategy=self.name,
            )
        if has_open_position:
            return self.make_signal(
                ticker, SignalType.HOLD, data, "Open position; mean-reversion exit conditions are absent", 0.0, fear,
                strategy=self.name,
            )

        knife_reason = falling_knife_reason(data, fear)
        if knife_reason:
            return self.make_signal(
                ticker, SignalType.HOLD, data, knife_reason, 0.0, fear,
                strategy=self.name,
            )

        bullish_reversal = self._is_bullish_hammer(data) or self._is_bullish_engulfing(data)
        velocity_decelerating = self._positive_price_acceleration(data)
        conditions = {
            "lower_band_touch": current["low"] <= current["bb_lower"],
            "oversold_rsi": current["rsi_14"] < 30,
            "bullish_reversal": bullish_reversal,
            "drop_decelerating": velocity_decelerating,
        }
        if all(conditions.values()):
            return self.make_signal(
                ticker, SignalType.BUY, data,
                "Lower Bollinger touch with oversold RSI, bullish reversal and decelerating drop", 0.75, fear,
                strategy=self.name, conditions=conditions,
            )
        failed = ", ".join(name for name, passed in conditions.items() if not passed)
        return self.make_signal(
            ticker, SignalType.HOLD, data, f"Mean-reversion entry conditions not met: {failed}", 0.0, fear,
            strategy=self.name, conditions=conditions,
        )

    @staticmethod
    def _is_bullish_hammer(data: pd.DataFrame) -> bool:
        candle = data.iloc[-1]
        body = abs(float(candle["close"] - candle["open"]))
        candle_range = float(candle["high"] - candle["low"])
        if candle_range <= 0 or body <= 0 or candle["close"] <= candle["open"]:
            return False
        lower_shadow = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
        upper_shadow = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
        return lower_shadow >= 2 * body and upper_shadow <= body and body / candle_range <= 0.35

    @staticmethod
    def _is_bullish_engulfing(data: pd.DataFrame) -> bool:
        if len(data) < 2:
            return False
        previous, current = data.iloc[-2], data.iloc[-1]
        return bool(
            previous["close"] < previous["open"]
            and current["close"] > current["open"]
            and current["open"] <= previous["close"]
            and current["close"] >= previous["open"]
        )

    @staticmethod
    def _positive_price_acceleration(data: pd.DataFrame) -> bool:
        if len(data) < 3:
            return False
        close = data["close"].astype(float)
        acceleration = (close.iloc[-1] - close.iloc[-2]) - (close.iloc[-2] - close.iloc[-3])
        return bool(acceleration > 0)

