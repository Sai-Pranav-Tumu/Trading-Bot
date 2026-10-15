"""Yahoo Finance market data access and normalization for Indian instruments."""

from __future__ import annotations

from datetime import datetime
from typing import Final

import pandas as pd
import yfinance as yf

from config import DEFAULT_TIMEFRAME, MARKET_TIMEZONE, MARKET_TIMEZONE_STR

_OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


class MarketDataError(RuntimeError):
    """Raised when a requested market-data series cannot be retrieved or cleaned."""


class MarketFeed:
    """Fetch and clean Yahoo Finance OHLCV data for NSE symbols and indices.

    Yahoo Finance is a delayed, best-effort source. It does not guarantee live
    exchange data or that every requested intraday interval is available.
    Missing bars are left missing rather than fabricated, which also avoids
    creating candles across weekends and exchange holidays.
    """

    def __init__(self, timeout: int = 15) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def fetch_historical(
        self,
        ticker: str,
        period: str = "5d",
        interval: str = DEFAULT_TIMEFRAME,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        """Return cleaned OHLCV candles with a Kolkata-localized DatetimeIndex.

        Use either ``period`` or the ``start``/``end`` range. Yahoo may reject
        ranges longer than its retention window for intraday intervals.
        """
        symbol = ticker.strip()
        if not symbol:
            raise ValueError("ticker must not be empty")
        if (start is None) != (end is None):
            raise ValueError("start and end must be provided together")

        options: dict[str, object] = {
            "tickers": symbol,
            "interval": interval,
            "auto_adjust": False,
            "actions": False,
            "progress": False,
            "threads": False,
            "timeout": self.timeout,
        }
        if start is None:
            options["period"] = period
        else:
            options["start"] = start
            options["end"] = end

        try:
            raw = yf.download(**options)
        except Exception as exc:  # yfinance surfaces network/provider errors
            raise MarketDataError(f"Failed to fetch {symbol}: {exc}") from exc

        frame = self._select_symbol_columns(raw, symbol)
        cleaned = self.clean_ohlcv(frame)
        if cleaned.empty:
            raise MarketDataError(f"No usable OHLCV candles returned for {symbol}")
        return cleaned

    def fetch_latest(
        self, ticker: str, period: str = "5d", interval: str = DEFAULT_TIMEFRAME
    ) -> pd.DataFrame:
        """Convenience alias for fetching the most recent provider-supported data."""
        return self.fetch_historical(ticker, period=period, interval=interval)

    @staticmethod
    def clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize columns, timestamps and invalid candles without inventing data."""
        if frame.empty:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS), index=pd.DatetimeIndex([], tz=MARKET_TIMEZONE_STR, name="timestamp"))

        result = frame.copy()
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [str(column[0]) for column in result.columns]
        result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
        result = result.loc[:, ~pd.Index(result.columns).duplicated(keep="first")]

        missing = set(_OHLCV_COLUMNS).difference(result.columns)
        if missing:
            raise MarketDataError(f"OHLCV data is missing columns: {', '.join(sorted(missing))}")
        result = result.loc[:, list(_OHLCV_COLUMNS)]

        index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="coerce"))
        valid_timestamps = ~index.isna()
        result = result.loc[valid_timestamps].copy()
        index = index[valid_timestamps]
        if index.tz is None:
            # Yahoo intraday timestamps without tz metadata represent exchange-local time.
            index = index.tz_localize(MARKET_TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        else:
            index = index.tz_convert(MARKET_TIMEZONE)
        valid_timestamps = ~index.isna()
        result = result.loc[valid_timestamps].copy()
        result.index = index[valid_timestamps]
        result.index.name = "timestamp"

        for column in _OHLCV_COLUMNS:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result = result.dropna(subset=["open", "high", "low", "close"])
        result = result.loc[result["volume"].fillna(0) >= 0]
        valid_range = (
            (result["high"] >= result[["open", "close", "low"]].max(axis=1))
            & (result["low"] <= result[["open", "close", "high"]].min(axis=1))
        )
        result = result.loc[valid_range].copy()
        result["volume"] = result["volume"].fillna(0).astype("int64")
        result = result[~result.index.duplicated(keep="last")].sort_index()
        return result

    @staticmethod
    def _select_symbol_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Handle yfinance's single-symbol and MultiIndex column formats."""
        if frame.empty:
            return frame
        if not isinstance(frame.columns, pd.MultiIndex):
            return frame

        for level in range(frame.columns.nlevels):
            values = frame.columns.get_level_values(level)
            if ticker in values:
                selected = frame.xs(ticker, axis=1, level=level, drop_level=True)
                if isinstance(selected.columns, pd.MultiIndex):
                    selected.columns = selected.columns.get_level_values(0)
                return selected
        # Older yfinance versions use (Price, Ticker); when only one ticker is
        # returned, selecting that sole ticker is unambiguous.
        for level in range(frame.columns.nlevels):
            values = frame.columns.get_level_values(level).unique()
            if len(values) == 1:
                selected = frame.xs(values[0], axis=1, level=level, drop_level=True)
                if isinstance(selected.columns, pd.MultiIndex):
                    selected.columns = selected.columns.get_level_values(0)
                return selected
        raise MarketDataError(f"Could not identify {ticker} in returned market-data columns")


def fetch_ohlcv(
    ticker: str,
    period: str = "5d",
    interval: str = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """Fetch OHLCV candles using a one-shot default market feed."""
    return MarketFeed().fetch_historical(ticker, period=period, interval=interval)

