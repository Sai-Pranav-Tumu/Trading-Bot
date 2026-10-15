"""
Indian Stock Market Algorithmic Trading Bot - Master Configuration.

This module houses all system-wide parameters, user configurations,
Indian market exchange rules (NSE/BSE), statutory tax/brokerage schedules,
and the Fear & Pullback Engine mathematical constants.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Tuple
import pytz
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

# ==============================================================================
# BASE DIRECTORIES & FILE PATHS
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
DATABASE_PATH: Path = DATA_DIR / "trading_bot.db"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# EXECUTION MODE & CAPITAL CONFIGURATION
# ==============================================================================
PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "True").lower() in ("true", "1", "yes")

# Initial Virtual Capital in INR (Default: ₹1,00,000)
INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "100000.0"))

# Risk per trade (% of current portfolio equity, default: 1.0%)
RISK_PER_TRADE_PCT: float = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))

# Maximum allowed portfolio drawdown (% from high-water mark, default: 5.0%)
# If equity drops below ₹95,000 (5% drawdown on ₹1,00,000), kill-switch is engaged.
MAX_DRAWDOWN_PCT: float = float(os.getenv("MAX_DRAWDOWN_PCT", "5.0"))
CAPITAL_KILL_SWITCH_FLOOR: float = INITIAL_CAPITAL * (1.0 - MAX_DRAWDOWN_PCT / 100.0)

# Daily Max Loss Limit (% of starting daily capital, default: 2.5% = ₹2,500)
DAILY_MAX_LOSS_PCT: float = float(os.getenv("DAILY_MAX_LOSS_PCT", "2.5"))

# ==============================================================================
# GRADUATION GATEKEEPER SPECIFICATIONS
# (Paper trading must meet these criteria before live broker trading is unlocked)
# ==============================================================================
@dataclass(frozen=True)
class GraduationConfig:
    """Rules and thresholds required to graduate from paper trading to live broker."""
    profit_target_pct: float = float(os.getenv("GRADUATION_PROFIT_TARGET_PCT", "15.0"))
    min_trades: int = int(os.getenv("GRADUATION_MIN_TRADES", "30"))
    min_win_rate_pct: float = float(os.getenv("GRADUATION_MIN_WIN_RATE_PCT", "50.0"))
    max_drawdown_allowed_pct: float = float(os.getenv("GRADUATION_MAX_DRAWDOWN_PCT", "5.0"))
    target_capital_floor: float = field(init=False)

    def __post_init__(self) -> None:
        target_capital = round(INITIAL_CAPITAL * (1.0 + self.profit_target_pct / 100.0), 2)
        object.__setattr__(self, "target_capital_floor", target_capital)

GRADUATION_RULES = GraduationConfig()

# ==============================================================================
# TARGET STOCK UNIVERSE & BENCHMARK SYMBOLS (NSE)
# ==============================================================================
# Primary Nifty 50 Liquid Leaders
WATCHLIST: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "TATAMOTORS.NS",
    "LT.NS",
]

# Benchmark index & Volatility index tickers on Yahoo Finance
NIFTY_BENCHMARK_TICKER: str = "^NSEI"      # NIFTY 50 Index
INDIA_VIX_TICKER: str = "^INDIAVIX"        # India VIX Volatility Index

# Default intraday timeframe
DEFAULT_TIMEFRAME: str = os.getenv("DATA_TIMEFRAME", "15m")

# ==============================================================================
# INDIAN MARKET HOURS (NSE/BSE) - TIMEZONE: Asia/Kolkata
# ==============================================================================
MARKET_TIMEZONE_STR: str = "Asia/Kolkata"
MARKET_TIMEZONE: pytz.BaseTzInfo = pytz.timezone(MARKET_TIMEZONE_STR)

MARKET_OPEN_TIME: time = time(9, 15)       # 09:15 IST
INTRADAY_SQUARE_OFF_TIME: time = time(15, 15)  # 15:15 IST (Auto-square-off for MIS)
MARKET_CLOSE_TIME: time = time(15, 30)      # 15:30 IST

# ==============================================================================
# DYNAMIC FEAR & PULLBACK ENGINE WEIGHTS & THRESHOLDS (F_t in [0.0, 1.0])
# F_t = w1*VIX_norm + w2*Velocity_drop + w3*ATR_spike + w4*Drawdown_factor
# ==============================================================================
FEAR_WEIGHTS: Dict[str, float] = {
    "vix_norm": 0.25,        # Macro Market Fear (India VIX relative to baseline)
    "velocity_drop": 0.35,   # Speed of price drop relative to 3*ATR14
    "atr_spike": 0.20,       # Current True Range expansion over ATR14
    "drawdown_factor": 0.20, # Current portfolio drawdown relative to max allowed
}

# Fear Threshold Regimes
FEAR_THRESHOLD_CAUTION: float = 0.40
FEAR_THRESHOLD_PULLBACK: float = 0.65
FEAR_THRESHOLD_PANIC: float = 0.80

# Emergency cooldown lock duration (number of candles)
PANIC_COOLDOWN_CANDLES: int = 5

# Baseline VIX thresholds for normalization
INDIA_VIX_BASELINE: float = 14.0
INDIA_VIX_PANIC_LEVEL: float = 24.0

# ==============================================================================
# INDIAN STATUTORY TAXES, EXCHANGE CHARGES & BROKERAGE SCHEDULE
# Based on SEBI / NSE Equity Intraday regulations
# ==============================================================================
@dataclass(frozen=True)
class IndianTaxSchedule:
    """Schedule of brokerage and regulatory taxes for Indian equity trading."""
    # Slippage simulation: 0.05% per execution
    simulated_slippage_pct: float = 0.0005
    
    # Brokerage: ₹20 or 0.03% per order (whichever is lower) for discount brokers
    brokerage_flat_cap: float = 20.0
    brokerage_rate: float = 0.0003
    
    # STT (Securities Transaction Tax):
    # Intraday: 0.025% on Sell side only
    # Delivery: 0.1% on both Buy and Sell sides
    stt_intraday_sell_rate: float = 0.00025
    stt_delivery_rate: float = 0.001
    
    # NSE cash market transaction charges + IPFT: ₹307/crore (0.00307%) each side,
    # effective 2026-03-01 (NSE circular NSE/FA/73061). SEBI fee is separate.
    nse_turnover_rate: float = 0.0000307
    
    # GST: 18% applied on (Brokerage + Exchange Charges + SEBI Charges)
    gst_rate: float = 0.18
    
    # SEBI Turnover Fee: ₹10 per crore (0.000001)
    sebi_rate: float = 0.000001
    
    # Stamp Duty: 0.003% on Buy side for intraday equity
    stamp_duty_rate: float = 0.00003

TAX_SCHEDULE = IndianTaxSchedule()

def calculate_indian_charges(
    buy_price: float,
    sell_price: float,
    quantity: int,
    is_intraday: bool = True
) -> Dict[str, float]:
    """
    Compute granular Indian statutory charges, taxes, and brokerage.

    Args:
        buy_price: Execution price on entry.
        sell_price: Execution price on exit.
        quantity: Number of shares.
        is_intraday: True for MIS (Intraday), False for CNC (Delivery).

    Returns:
        Dict with total charges, STT, brokerage, exchange charges, GST, SEBI fee, and stamp duty.
    """
    buy_turnover = float(buy_price * quantity)
    sell_turnover = float(sell_price * quantity)
    total_turnover = buy_turnover + sell_turnover

    # Brokerage calculation (per leg, capped at ₹20 per order)
    buy_brokerage = min(TAX_SCHEDULE.brokerage_flat_cap, buy_turnover * TAX_SCHEDULE.brokerage_rate)
    sell_brokerage = min(TAX_SCHEDULE.brokerage_flat_cap, sell_turnover * TAX_SCHEDULE.brokerage_rate)
    total_brokerage = buy_brokerage + sell_brokerage

    # Securities Transaction Tax (STT)
    if is_intraday:
        stt = round(sell_turnover * TAX_SCHEDULE.stt_intraday_sell_rate, 2)
    else:
        stt = round(total_turnover * TAX_SCHEDULE.stt_delivery_rate, 2)

    # Exchange Transaction Charges (NSE)
    exchange_charges = round(total_turnover * TAX_SCHEDULE.nse_turnover_rate, 2)

    # SEBI Turnover Charges
    sebi_charges = round(total_turnover * TAX_SCHEDULE.sebi_rate, 2)

    # Stamp Duty (Buy side only)
    stamp_duty = round(buy_turnover * TAX_SCHEDULE.stamp_duty_rate, 2)

    # GST on (Brokerage + Exchange Charges + SEBI Charges)
    taxable_services = total_brokerage + exchange_charges + sebi_charges
    gst = round(taxable_services * TAX_SCHEDULE.gst_rate, 2)

    total_charges = round(total_brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty, 2)

    return {
        "buy_turnover": round(buy_turnover, 2),
        "sell_turnover": round(sell_turnover, 2),
        "total_turnover": round(total_turnover, 2),
        "brokerage": round(total_brokerage, 2),
        "stt": stt,
        "exchange_charges": exchange_charges,
        "gst": gst,
        "sebi_charges": sebi_charges,
        "stamp_duty": stamp_duty,
        "total_charges": total_charges,
    }

# ==============================================================================
# MARKET SCHEDULE HELPER UTILITIES
# ==============================================================================
def get_current_ist_time() -> datetime:
    """Return the current localized datetime in Asia/Kolkata timezone."""
    return datetime.now(MARKET_TIMEZONE)

def is_market_open(dt: datetime | None = None) -> bool:
    """
    Check if the Indian stock market (NSE) is currently open for regular trading.
    Considers trading hours 09:15 to 15:30 IST on Monday through Friday.
    """
    if dt is None:
        dt = get_current_ist_time()
    else:
        if dt.tzinfo is None:
            dt = MARKET_TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(MARKET_TIMEZONE)

    # Monday is 0, Sunday is 6
    if dt.weekday() >= 5:
        return False

    current_time = dt.time()
    return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME

def is_square_off_time(dt: datetime | None = None) -> bool:
    """
    Check if the current time is at or past intraday MIS auto-square-off (15:15 IST).
    """
    if dt is None:
        dt = get_current_ist_time()
    else:
        if dt.tzinfo is None:
            dt = MARKET_TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(MARKET_TIMEZONE)

    current_time = dt.time()
    return current_time >= INTRADAY_SQUARE_OFF_TIME
