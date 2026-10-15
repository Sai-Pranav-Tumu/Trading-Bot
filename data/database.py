"""SQLite storage for paper-trading orders, positions, fear and daily PnL."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import DATABASE_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id TEXT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type TEXT NOT NULL DEFAULT 'MARKET',
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    requested_price REAL,
    fill_price REAL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    fees REAL NOT NULL DEFAULT 0,
    realized_pnl REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS positions (
    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    quantity INTEGER NOT NULL,
    average_price REAL NOT NULL,
    current_price REAL,
    stop_price REAL,
    unrealized_pnl REAL NOT NULL DEFAULT 0,
    opened_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS fear_logs (
    fear_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    fear_score REAL NOT NULL CHECK (fear_score >= 0 AND fear_score <= 1),
    regime TEXT NOT NULL,
    vix_norm REAL,
    velocity_drop REAL,
    atr_spike REAL,
    drawdown_factor REAL,
    logged_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    session_date TEXT PRIMARY KEY,
    starting_equity REAL NOT NULL,
    ending_equity REAL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    unrealized_pnl REAL NOT NULL DEFAULT 0,
    trade_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_ticker_created ON orders (ticker, created_at);
CREATE INDEX IF NOT EXISTS idx_fear_logs_ticker_logged ON fear_logs (ticker, logged_at);
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""
    path = Path(db_path) if db_path is not None else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database(db_path: str | Path | None = None) -> Path:
    """Create the local database tables and indexes; safe to call repeatedly."""
    path = Path(db_path) if db_path is not None else DATABASE_PATH
    connection = get_connection(path)
    try:
        with connection:
            connection.executescript(_SCHEMA)
    finally:
        connection.close()
    return path


@contextmanager
def database_connection(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection that commits on success and rolls back on error."""
    connection = get_connection(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()

