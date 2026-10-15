"""Technical indicators used by the NSE paper-trading strategies."""

from __future__ import annotations

import pandas as pd


def _require_ohlcv(frame: pd.DataFrame) -> None:
    required = {"high", "low", "close", "volume"}
    missing = required.difference(column.lower() for column in frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {', '.join(sorted(missing))}")


def calculate_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of OHLCV data with EMA, RSI, ATR, VWAP and Bollinger columns.

    Output columns are ``ema_9``, ``ema_21``, ``rsi_14``, ``atr_14``,
    ``vwap``, ``bb_upper``, ``bb_middle`` and ``bb_lower``. VWAP resets at
    each calendar session when the input index is a DatetimeIndex.
    """
    if frame.empty:
        raise ValueError("OHLCV frame must contain at least one row")
    _require_ohlcv(frame)

    result = frame.copy()
    result.columns = [str(column).lower() for column in result.columns]
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    close = pd.to_numeric(result["close"], errors="coerce")
    volume = pd.to_numeric(result["volume"], errors="coerce").fillna(0).clip(lower=0)

    result["ema_9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    result["ema_21"] = close.ewm(span=21, adjust=False, min_periods=21).mean()

    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = avg_gain / avg_loss.where(avg_loss != 0)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    result["rsi_14"] = rsi

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    result["true_range"] = true_range
    result["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    typical_price = (high + low + close) / 3
    price_volume = typical_price * volume
    if isinstance(result.index, pd.DatetimeIndex):
        sessions = pd.Series(result.index.date, index=result.index)
        cumulative_pv = price_volume.groupby(sessions).cumsum()
        cumulative_volume = volume.groupby(sessions).cumsum()
    else:
        cumulative_pv = price_volume.cumsum()
        cumulative_volume = volume.cumsum()
    result["vwap"] = cumulative_pv / cumulative_volume.where(cumulative_volume != 0)

    bb_middle = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    result["bb_middle"] = bb_middle
    result["bb_upper"] = bb_middle + 2 * bb_std
    result["bb_lower"] = bb_middle - 2 * bb_std
    return result


def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate an exponentially weighted moving average."""
    if period < 1:
        raise ValueError("period must be >= 1")
    return pd.to_numeric(series, errors="coerce").ewm(
        span=period, adjust=False, min_periods=period
    ).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder-style RSI for a numeric price series."""
    if period < 1:
        raise ValueError("period must be >= 1")
    prices = pd.to_numeric(series, errors="coerce")
    change = prices.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = -change.clip(upper=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.where(loss != 0)
    output = 100 - 100 / (1 + rs)
    output = output.mask((loss == 0) & (gain > 0), 100.0)
    return output.mask((loss == 0) & (gain == 0), 50.0)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Wilder-style Average True Range from OHLC data."""
    if period < 1:
        raise ValueError("period must be >= 1")
    _require_ohlcv(frame)
    columns = {column.lower(): column for column in frame.columns}
    high = pd.to_numeric(frame[columns["high"]], errors="coerce")
    low = pd.to_numeric(frame[columns["low"]], errors="coerce")
    close = pd.to_numeric(frame[columns["close"]], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(frame: pd.DataFrame) -> pd.Series:
    """Calculate session VWAP, resetting each date for datetime-indexed bars."""
    _require_ohlcv(frame)
    columns = {column.lower(): column for column in frame.columns}
    high = pd.to_numeric(frame[columns["high"]], errors="coerce")
    low = pd.to_numeric(frame[columns["low"]], errors="coerce")
    close = pd.to_numeric(frame[columns["close"]], errors="coerce")
    volume = pd.to_numeric(frame[columns["volume"]], errors="coerce").fillna(0).clip(lower=0)
    pv = ((high + low + close) / 3) * volume
    if isinstance(frame.index, pd.DatetimeIndex):
        sessions = pd.Series(frame.index.date, index=frame.index)
        pv = pv.groupby(sessions).cumsum()
        volume = volume.groupby(sessions).cumsum()
    else:
        pv = pv.cumsum()
        volume = volume.cumsum()
    return pv / volume.where(volume != 0)

