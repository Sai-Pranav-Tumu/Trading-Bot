"""VWAP + EMA trend pullback strategy for intraday momentum."""

from __future__ import annotations

import pandas as pd

from indicators.fear_engine import FearState
from strategies.base_strategy import BaseStrategy, Signal, SignalType, falling_knife_reason


class VWAPMomentumStrategy(BaseStrategy):
    """Enter aligned uptrends and exit on a loss of VWAP/EMA trend structure."""

    name = "vwap_momentum"

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
            current["close"] < current["vwap"] or current["ema_9"] < current["ema_21"]
        ):
            return self.make_signal(
                ticker, SignalType.SELL, data,
                "Trend exit: price lost VWAP or EMA 9 crossed below EMA 21", 0.8, fear,
                strategy=self.name,
            )

        if fear.fear_score >= 0.65:
            return self.make_signal(
                ticker, SignalType.HOLD, data,
                "Fear score blocks new momentum entries", 0.0, fear,
                strategy=self.name,
            )
        if has_open_position:
            return self.make_signal(
                ticker, SignalType.HOLD, data, "Open position; trend exit conditions are absent", 0.0, fear,
                strategy=self.name,
            )

        knife_reason = falling_knife_reason(data, fear)
        if knife_reason:
            return self.make_signal(
                ticker, SignalType.HOLD, data, knife_reason, 0.0, fear,
                strategy=self.name,
            )

        volume_average = data["volume"].tail(20).mean()
        conditions = {
            "price_above_vwap": current["close"] > current["vwap"],
            "ema_trend": current["ema_9"] > current["ema_21"],
            "rsi_pullback": 45 <= current["rsi_14"] <= 65,
            "volume_confirmation": current["volume"] > 1.2 * volume_average,
        }
        if all(conditions.values()):
            return self.make_signal(
                ticker, SignalType.BUY, data,
                "VWAP, EMA trend, RSI pullback and volume conditions aligned", 0.8, fear,
                strategy=self.name, conditions=conditions,
            )
        failed = ", ".join(name for name, passed in conditions.items() if not passed)
        return self.make_signal(
            ticker, SignalType.HOLD, data, f"Momentum entry conditions not met: {failed}", 0.0, fear,
            strategy=self.name, conditions=conditions,
        )

