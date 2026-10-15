"""Sample-frame verification for the Part 4 strategy signals."""

import numpy as np
import pandas as pd

from indicators.fear_engine import FearRegime, FearState
from strategies.base_strategy import SignalType, falling_knife_reason
from strategies.mean_reversion import MeanReversionStrategy
from strategies.vwap_momentum import VWAPMomentumStrategy


def _fear(*, score: float = 0.20, velocity: float = 0.0) -> FearState:
    return FearState(
        fear_score=score,
        regime=FearRegime.NORMAL,
        position_size_multiplier=1.0,
        stop_loss_atr_multiplier=2.0,
        vix_norm=0.0,
        velocity_drop=velocity,
        atr_spike=0.5,
        drawdown_factor=0.0,
        india_vix=14.0,
        current_atr=0.5,
    )


def _frame(closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes.copy(),
            "high": closes + 0.2,
            "low": closes - 0.2,
            "close": closes.copy(),
            "volume": np.full(len(closes), 1000),
        }
    )


def test_momentum_strategy_emits_buy_when_all_entry_conditions_align():
    index = np.arange(50)
    closes = 100 + index * 0.04 + 0.8 * np.sin(index * 0.5)
    data = _frame(closes)
    data.loc[data.index[-1], "volume"] = 3000
    signal = VWAPMomentumStrategy().generate_signal("TEST.NS", data, fear_state=_fear())
    assert signal.action is SignalType.BUY, signal.reason
    assert signal.metadata["conditions"] == {
        "price_above_vwap": True,
        "ema_trend": True,
        "rsi_pullback": True,
        "volume_confirmation": True,
    }


def test_momentum_exit_emits_sell_for_open_position_below_vwap():
    closes = np.linspace(100.0, 105.0, 35)
    data = _frame(closes)
    data.loc[data.index[-1], ["open", "high", "low", "close"]] = [102, 102.1, 101, 101.1]
    signal = VWAPMomentumStrategy().generate_signal(
        "TEST.NS", data, fear_state=_fear(), has_open_position=True
    )
    assert signal.action is SignalType.SELL


def test_mean_reversion_requires_reversal_and_decelerating_drop():
    closes = np.linspace(100.0, 81.0, 35)
    data = _frame(closes)
    data.loc[data.index[-2], ["open", "high", "low", "close"]] = [81.7, 81.8, 80.8, 81.0]
    data.loc[data.index[-1], ["open", "high", "low", "close"]] = [81.0, 81.3, 75.0, 81.2]
    signal = MeanReversionStrategy().generate_signal("TEST.NS", data, fear_state=_fear())
    assert signal.action is SignalType.BUY, signal.reason


def test_falling_knife_filter_blocks_fast_selloff():
    data = _frame(np.linspace(100.0, 90.0, 25))
    assert "velocity" in falling_knife_reason(data, _fear(velocity=0.8))
