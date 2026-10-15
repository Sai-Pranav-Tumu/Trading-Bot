"""Verification cases for technical indicators and the fear engine."""

import numpy as np
import pandas as pd

from indicators.fear_engine import FearEngine, FearRegime
from indicators.technicals import calculate_indicators


def _ohlcv(closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.20,
            "low": closes - 0.20,
            "close": closes,
            "volume": np.full(len(closes), 10_000),
        },
        index=pd.date_range("2026-10-01 09:15", periods=len(closes), freq="15min", tz="Asia/Kolkata"),
    )


def test_calculate_indicators_adds_required_columns_and_bounded_rsi():
    closes = np.linspace(100.0, 110.0, 40)
    result = calculate_indicators(_ohlcv(closes))
    required = {"ema_9", "ema_21", "rsi_14", "atr_14", "vwap", "bb_upper", "bb_middle", "bb_lower"}
    assert required <= set(result.columns)
    assert result["rsi_14"].dropna().between(0, 100).all()
    assert result["atr_14"].dropna().gt(0).all()


def test_sharp_35_percent_drop_triggers_high_fear_or_panic():
    closes = np.full(50, 100.0)
    closes[-3:] = [99.0, 97.5, 96.5]
    state = FearEngine().compute(_ohlcv(closes), india_vix_data=30.0)
    assert state.fear_score >= 0.65
    assert state.regime in {FearRegime.HIGH_FEAR_PULLBACK, FearRegime.EXTREME_PANIC}
    assert state.position_size_multiplier == 0.0


def test_extreme_score_enters_panic_cooldown():
    closes = np.full(50, 100.0)
    closes[-3:] = [99.0, 97.5, 96.5]
    state = FearEngine().compute(
        _ohlcv(closes), india_vix_data=30.0, portfolio_drawdown_pct=5.0
    )
    assert state.regime is FearRegime.EXTREME_PANIC
    assert state.cooldown_candles > 0
