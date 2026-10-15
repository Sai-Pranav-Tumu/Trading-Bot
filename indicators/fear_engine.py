"""Composite Fear & Pullback engine for risk-aware paper trading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import pandas as pd

import config
from indicators.technicals import calculate_indicators


class FearRegime(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    HIGH_FEAR_PULLBACK = "HIGH_FEAR_PULLBACK"
    EXTREME_PANIC = "EXTREME_PANIC"


@dataclass(frozen=True)
class FearState:
    """Fear result and the sizing/stop action associated with its regime."""

    fear_score: float
    regime: FearRegime
    position_size_multiplier: float
    stop_loss_atr_multiplier: float
    vix_norm: float
    velocity_drop: float
    atr_spike: float
    drawdown_factor: float
    india_vix: float
    current_atr: float
    cooldown_candles: int = 0


class FearEngine:
    """Calculate the weighted Fear score from OHLCV, India VIX and drawdown.

    ``portfolio_drawdown_pct`` is a positive percentage drawdown from the
    portfolio high-water mark (for example, ``2.5`` means a 2.5% drawdown).
    """

    def __init__(self) -> None:
        weights = config.FEAR_WEIGHTS
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("Fear Engine weights must sum to 1.0")

    def compute(
        self,
        stock_ohlcv: pd.DataFrame,
        india_vix_data: pd.DataFrame | pd.Series | float | int | None = None,
        portfolio_drawdown_pct: float = 0.0,
    ) -> FearState:
        """Compute score, regime and corresponding position/stop multipliers."""
        if not isinstance(stock_ohlcv, pd.DataFrame) or stock_ohlcv.empty:
            raise ValueError("stock_ohlcv must be a non-empty DataFrame")
        if portfolio_drawdown_pct < 0:
            raise ValueError("portfolio_drawdown_pct must be non-negative")

        indicators = calculate_indicators(stock_ohlcv)
        last = indicators.iloc[-1]
        atr_value = last["atr_14"]
        if pd.isna(atr_value) or atr_value <= 0:
            raise ValueError("At least 14 valid candles with non-zero ATR are required")

        close = pd.to_numeric(indicators["close"], errors="coerce")
        if len(close) < 4 or close.iloc[-4] <= 0:
            raise ValueError("At least four valid candles are required for drop velocity")
        velocity = self._clamp(-(float(close.iloc[-1]) - float(close.iloc[-4])) / (3 * float(atr_value)))

        true_range = float(last["true_range"])
        atr_spike = self._clamp(true_range / float(atr_value))
        india_vix = self._latest_vix(india_vix_data)
        vix_norm = self._clamp(
            (india_vix - config.INDIA_VIX_BASELINE)
            / (config.INDIA_VIX_PANIC_LEVEL - config.INDIA_VIX_BASELINE)
        )
        drawdown_factor = self._clamp(
            portfolio_drawdown_pct / config.MAX_DRAWDOWN_PCT
            if config.MAX_DRAWDOWN_PCT > 0
            else 0.0
        )

        components = {
            "vix_norm": vix_norm,
            "velocity_drop": velocity,
            "atr_spike": atr_spike,
            "drawdown_factor": drawdown_factor,
        }
        score = self._clamp(sum(config.FEAR_WEIGHTS[key] * value for key, value in components.items()))
        regime, size_multiplier, stop_multiplier, cooldown = self._action_for_score(score)
        return FearState(
            fear_score=score,
            regime=regime,
            position_size_multiplier=size_multiplier,
            stop_loss_atr_multiplier=stop_multiplier,
            vix_norm=vix_norm,
            velocity_drop=velocity,
            atr_spike=atr_spike,
            drawdown_factor=drawdown_factor,
            india_vix=india_vix,
            current_atr=float(atr_value),
            cooldown_candles=cooldown,
        )

    @staticmethod
    def _latest_vix(data: pd.DataFrame | pd.Series | float | int | None) -> float:
        if data is None:
            return config.INDIA_VIX_BASELINE
        if isinstance(data, (int, float)):
            value = float(data)
        elif isinstance(data, pd.Series):
            clean = pd.to_numeric(data, errors="coerce").dropna()
            value = float(clean.iloc[-1]) if not clean.empty else config.INDIA_VIX_BASELINE
        elif isinstance(data, pd.DataFrame):
            column = next((name for name in ("close", "Close", "adj close", "Adj Close") if name in data.columns), None)
            if column is None and not data.empty and data.shape[1] == 1:
                values = data.iloc[:, 0]
            elif column is None:
                return config.INDIA_VIX_BASELINE
            else:
                values = data[column]
            clean = pd.to_numeric(values, errors="coerce").dropna()
            value = float(clean.iloc[-1]) if not clean.empty else config.INDIA_VIX_BASELINE
        else:
            raise TypeError("india_vix_data must be a DataFrame, Series, number or None")
        if not math.isfinite(value) or value < 0:
            raise ValueError("India VIX must be a finite non-negative number")
        return value

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _action_for_score(score: float) -> tuple[FearRegime, float, float, int]:
        if score >= config.FEAR_THRESHOLD_PANIC:
            return FearRegime.EXTREME_PANIC, 0.0, 0.0, config.PANIC_COOLDOWN_CANDLES
        if score >= config.FEAR_THRESHOLD_PULLBACK:
            return FearRegime.HIGH_FEAR_PULLBACK, 0.0, 0.75, 0
        if score >= config.FEAR_THRESHOLD_CAUTION:
            return FearRegime.CAUTION, 0.5, 1.2, 0
        return FearRegime.NORMAL, 1.0, 2.0, 0

