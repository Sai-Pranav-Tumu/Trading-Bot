"""Basic Part 6 backtester integration verification."""

import numpy as np
import pandas as pd

from backtest.backtester import Backtester


def test_backtester_returns_finite_portfolio_metrics():
    index = pd.date_range("2026-10-05 09:15", periods=80, freq="15min", tz="Asia/Kolkata")
    close = 100 + np.linspace(0, 2, len(index)) + np.sin(np.arange(len(index)) / 3)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(index), 10_000),
        },
        index=index,
    )
    result = Backtester(initial_capital=100_000).run({"TEST.NS": frame})
    assert result.initial_capital == 100_000
    assert np.isfinite(result.final_equity)
    assert np.isfinite(result.total_return_pct)
    assert 0 <= result.max_drawdown_pct <= 100
    assert result.closed_trades >= 0
