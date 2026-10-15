"""
Unit tests for Part 1: Scaffolding, Environment & Configuration.
"""

from datetime import datetime
import pytz
import pytest

import config


def test_configuration_defaults():
    """Verify that user-selected pre-flight parameters match config."""
    assert config.INITIAL_CAPITAL == 100000.0
    assert config.MAX_DRAWDOWN_PCT == 5.0
    assert config.CAPITAL_KILL_SWITCH_FLOOR == 95000.0
    assert config.DAILY_MAX_LOSS_PCT == 2.5
    assert config.RISK_PER_TRADE_PCT == 1.0


def test_graduation_gatekeeper_rules():
    """Verify Graduation Gatekeeper threshold calculations."""
    assert config.GRADUATION_RULES.profit_target_pct == 15.0
    assert config.GRADUATION_RULES.min_trades == 30
    assert config.GRADUATION_RULES.min_win_rate_pct == 50.0
    assert config.GRADUATION_RULES.max_drawdown_allowed_pct == 5.0
    assert config.GRADUATION_RULES.target_capital_floor == 115000.0


def test_universe_and_benchmarks():
    """Verify watchlist tickers and index benchmarks."""
    expected_tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "TATAMOTORS.NS",
        "LT.NS",
    ]
    assert config.WATCHLIST == expected_tickers
    assert config.NIFTY_BENCHMARK_TICKER == "^NSEI"
    assert config.INDIA_VIX_TICKER == "^INDIAVIX"


def test_fear_weights_and_thresholds():
    """Verify Fear Engine weight sum and threshold hierarchy."""
    weights = config.FEAR_WEIGHTS
    total_weight = sum(weights.values())
    assert pytest.approx(total_weight, 0.001) == 1.0

    assert config.FEAR_THRESHOLD_CAUTION == 0.40
    assert config.FEAR_THRESHOLD_PULLBACK == 0.65
    assert config.FEAR_THRESHOLD_PANIC == 0.80
    assert (
        config.FEAR_THRESHOLD_CAUTION
        < config.FEAR_THRESHOLD_PULLBACK
        < config.FEAR_THRESHOLD_PANIC
    )


def test_indian_charges_calculation():
    """Verify Indian statutory charges and brokerage for an intraday trade."""
    # Buy 100 shares of Reliance at 2500, sell at 2520 (Gross profit: Rs 2000)
    charges = config.calculate_indian_charges(
        buy_price=2500.0,
        sell_price=2520.0,
        quantity=100,
        is_intraday=True
    )

    assert charges["buy_turnover"] == 250000.0
    assert charges["sell_turnover"] == 252000.0
    assert charges["total_turnover"] == 502000.0

    # Brokerage: 0.03% capped at Rs 20 per leg -> Total Rs 40
    assert charges["brokerage"] == 40.0

    # STT: 0.025% on sell turnover (252000 * 0.00025 = 63.0)
    assert charges["stt"] == 63.0

    # Stamp duty: 0.003% on buy turnover (250000 * 0.00003 = 7.50)
    assert charges["stamp_duty"] == 7.50

    # All charges must be positive numbers
    assert charges["exchange_charges"] > 0
    assert charges["gst"] > 0
    assert charges["sebi_charges"] > 0
    assert charges["total_charges"] > (charges["brokerage"] + charges["stt"])


def test_market_time_helpers():
    """Verify market open and square-off detection logic."""
    kolkata_tz = pytz.timezone("Asia/Kolkata")

    # A typical Wednesday at 10:30 AM IST (Market is Open)
    open_time = kolkata_tz.localize(datetime(2026, 10, 14, 10, 30))
    assert config.is_market_open(open_time) is True
    assert config.is_square_off_time(open_time) is False

    # A Wednesday at 15:20 IST (Market open, but past 15:15 square-off)
    sqoff_time = kolkata_tz.localize(datetime(2026, 10, 14, 15, 20))
    assert config.is_market_open(sqoff_time) is True
    assert config.is_square_off_time(sqoff_time) is True

    # A Wednesday at 20:00 IST (Market is closed)
    night_time = kolkata_tz.localize(datetime(2026, 10, 14, 20, 0))
    assert config.is_market_open(night_time) is False

    # A Sunday at 11:00 AM IST (Weekend - Market is closed)
    sunday_time = kolkata_tz.localize(datetime(2026, 10, 18, 11, 0))
    assert config.is_market_open(sunday_time) is False
