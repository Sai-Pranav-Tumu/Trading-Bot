"""CLI controller for historical backtests and live-feed paper trading."""

from __future__ import annotations

import argparse
import time as time_module
from datetime import datetime
from datetime import time as datetime_time, timedelta, timezone

import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import config
from backtest.backtester import BacktestResult, Backtester
from data.database import database_connection
from data.market_feed import MarketDataError, MarketFeed
from engine.gatekeeper import GraduationGatekeeper
from indicators.fear_engine import FearEngine
from risk_and_execution.broker_adapter import PaperBroker
from risk_and_execution.risk_manager import RiskManager
from strategies.base_strategy import Signal, SignalType
from strategies.mean_reversion import MeanReversionStrategy
from strategies.vwap_momentum import VWAPMomentumStrategy


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NSE paper-trading bot")
    parser.add_argument("--mode", choices=("backtest", "live"), default="live")
    parser.add_argument("--symbols", nargs="+", default=config.WATCHLIST)
    parser.add_argument("--interval", default=config.DEFAULT_TIMEFRAME)
    parser.add_argument("--period", default=None, help="Yahoo period (backtest defaults to 60d; live uses 5d)")
    parser.add_argument("--refresh-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true", help="Run one live paper-data cycle and exit")
    parser.add_argument("--capital", type=float, default=config.INITIAL_CAPITAL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.capital <= 0:
        console.print("[red]Capital must be positive.[/red]")
        return 2
    if args.refresh_seconds <= 0:
        console.print("[red]Refresh interval must be positive.[/red]")
        return 2
    if args.mode == "backtest":
        return run_backtest(args)
    if not config.PAPER_TRADING:
        console.print(
            "[bold red]This build supports paper execution only. Set PAPER_TRADING=True; "
            "no live broker adapter is enabled.[/bold red]"
        )
        return 2
    return run_live_paper(args)


def run_backtest(args: argparse.Namespace) -> int:
    period = args.period or "60d"
    feed = MarketFeed()
    candles: dict[str, pd.DataFrame] = {}
    with console.status(f"Fetching {period} of {args.interval} candles..."):
        for ticker in args.symbols:
            try:
                candles[ticker] = feed.fetch_historical(ticker, period=period, interval=args.interval)
                console.print(f"Loaded {ticker}: {len(candles[ticker])} candles")
            except (MarketDataError, ValueError) as exc:
                console.print(f"[yellow]Skipping {ticker}: {exc}[/yellow]")
        try:
            vix = feed.fetch_historical(
                config.INDIA_VIX_TICKER, period=period, interval=args.interval
            )
        except (MarketDataError, ValueError) as exc:
            console.print(f"[yellow]India VIX unavailable; using neutral baseline: {exc}[/yellow]")
            vix = None

    if not candles:
        console.print("[red]No symbol data was available for the backtest.[/red]")
        return 1
    result = Backtester(initial_capital=args.capital).run(candles, vix)
    render_backtest_result(result)
    return 0


def render_backtest_result(result: BacktestResult) -> None:
    table = Table(title="Paper Backtest Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Result", justify="right")
    table.add_row("Initial capital", f"₹{result.initial_capital:,.2f}")
    table.add_row("Final equity", f"₹{result.final_equity:,.2f}")
    table.add_row("Total return", f"{result.total_return_pct:.2f}%")
    table.add_row("Win rate", f"{result.win_rate_pct:.2f}%")
    table.add_row("Max drawdown", f"{result.max_drawdown_pct:.2f}%")
    table.add_row("Sharpe ratio", f"{result.sharpe_ratio:.2f}")
    table.add_row("Fear engine capital saved*", f"₹{result.capital_saved_by_fear:,.2f}")
    table.add_row("Closed trade fills", str(result.closed_trades))
    table.add_row("Winning / losing fills", f"{result.winning_trades} / {result.losing_trades}")
    console.print(table)
    console.print("[dim]*Estimated as positive loss avoided through the final mark after fear-driven exits.[/dim]")


def run_live_paper(args: argparse.Namespace) -> int:
    feed = MarketFeed()
    broker = PaperBroker(initial_capital=args.capital)
    risk = RiskManager(initial_capital=args.capital)
    fear_engine = FearEngine()
    momentum = VWAPMomentumStrategy(fear_engine)
    mean_reversion = MeanReversionStrategy(fear_engine)
    last_seen: dict[str, pd.Timestamp] = {}
    quote_data: dict[str, pd.DataFrame] = {}
    fears = {}
    signals: dict[str, str] = {}
    errors: dict[str, str] = {}
    vix_value: float | None = None
    session_date = None
    session_open_equity = broker.portfolio_value
    last_cooldown_timestamp = None

    try:
        with Live(console=console, refresh_per_second=2, screen=False) as live:
            while True:
                today = datetime.now(config.MARKET_TIMEZONE).date()
                if today != session_date:
                    session_date = today
                    session_open_equity = broker.portfolio_value
                    risk.reset_session()
                errors.clear()

                for ticker in args.symbols:
                    try:
                        quote_data[ticker] = feed.fetch_historical(
                            ticker, period="5d", interval=args.interval
                        )
                    except Exception as exc:
                        errors[ticker] = str(exc)
                try:
                    vix_frame = feed.fetch_historical(
                        config.INDIA_VIX_TICKER, period="5d", interval=args.interval
                    )
                    vix_value = float(vix_frame["close"].iloc[-1])
                except Exception as exc:
                    vix_frame = None
                    errors[config.INDIA_VIX_TICKER] = str(exc)

                for ticker, frame in quote_data.items():
                    if frame.empty:
                        continue
                    price = float(frame["close"].iloc[-1])
                    broker.update_market_price(ticker, price)
                daily_pnl = broker.portfolio_value - session_open_equity
                risk.enforce_daily_loss(broker, daily_pnl, session_open_equity)
                risk.enforce_portfolio_kill_switch(broker, broker.portfolio_value)

                for ticker, frame in quote_data.items():
                    if len(frame) < 21:
                        signals[ticker] = "HOLD: waiting for indicator history"
                        continue
                    timestamp = frame.index[-1]
                    price = float(frame["close"].iloc[-1])
                    try:
                        state = fear_engine.compute(
                            frame, vix_frame, max(0.0, (session_open_equity - broker.portfolio_value) / max(session_open_equity, 1) * 100.0)
                        )
                        fears[ticker] = state
                        if timestamp > last_cooldown_timestamp if last_cooldown_timestamp is not None else True:
                            risk.advance_candle()
                            last_cooldown_timestamp = timestamp

                        is_new_bar = ticker not in last_seen or timestamp > last_seen[ticker]
                        if not is_new_bar:
                            current = broker.get_position(ticker)
                            if current and current.stop_price is not None and price <= current.stop_price:
                                order = broker.sell(ticker, current.quantity, price)
                                signals[ticker] = f"SELL {order.quantity}: trailing stop filled"
                            continue
                        last_seen[ticker] = timestamp

                        current = broker.get_position(ticker)
                        if current:
                            pullback = risk.apply_fear_pullback(broker, ticker, price, state)
                            if pullback:
                                signals[ticker] = f"{pullback.side} {pullback.quantity}: {pullback.reason}"
                            current = broker.get_position(ticker)
                        if current:
                            stop = risk.ratchet_trailing_stop(price, state.current_atr, state, current.stop_price)
                            broker.set_stop_price(ticker, stop)
                            if price <= stop:
                                order = broker.sell(ticker, current.quantity, price)
                                signals[ticker] = f"SELL {order.quantity}: trailing stop filled"
                                current = None
                        if current:
                            for strategy in (momentum, mean_reversion):
                                signal = strategy.generate_signal(
                                    ticker, frame, fear_state=state, has_open_position=True
                                )
                                if signal.action is SignalType.SELL:
                                    order = broker.sell(ticker, current.quantity, price)
                                    signals[ticker] = f"SELL {order.quantity}: {signal.reason}"
                                    current = None
                                    break
                        current_session_bar = (
                            timestamp.date() == today if hasattr(timestamp, "date") else False
                        )
                        if (
                            current is None
                            and config.is_market_open()
                            and current_session_bar
                            and self_can_open(risk, broker, session_open_equity)
                        ):
                            selected = None
                            for strategy in (momentum, mean_reversion):
                                signal = strategy.generate_signal(
                                    ticker, frame, fear_state=state, has_open_position=False
                                )
                                if signal.action is SignalType.BUY:
                                    selected = signal
                                    break
                                selected = signal
                            signals[ticker] = f"{selected.action.value}: {selected.reason}" if selected else "HOLD"
                            if selected and selected.action is SignalType.BUY:
                                stop = price - state.current_atr * state.stop_loss_atr_multiplier
                                quantity = risk.calculate_quantity(
                                    broker.portfolio_value, price, stop, state
                                )
                                if quantity:
                                    order = broker.buy(ticker, quantity, price)
                                    signals[ticker] = (
                                        f"BUY {order.quantity}: {selected.reason}"
                                        if order.status == "FILLED"
                                        else f"REJECTED: {order.reason}"
                                    )
                                    if order.status == "FILLED":
                                        broker.set_stop_price(ticker, stop)
                    except Exception as exc:
                        signals[ticker] = f"ERROR: {exc}"

                daily_pnl = broker.portfolio_value - session_open_equity
                _record_daily_pnl(broker, session_date, session_open_equity, daily_pnl)
                graduation = _evaluate_graduation(broker, args.capital)
                live.update(render_dashboard(
                    args.symbols, quote_data, fears, broker, signals, errors,
                    vix_value, daily_pnl, risk, graduation.unlocked,
                ))
                if args.once:
                    break
                time_module.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        console.print("\nPaper loop stopped.")
    return 0


def self_can_open(risk: RiskManager, broker: PaperBroker, session_open_equity: float) -> bool:
    return risk.can_open_position(
        broker.portfolio_value - session_open_equity, session_open_equity
    )


def render_dashboard(
    symbols: list[str],
    quote_data: dict[str, pd.DataFrame],
    fears: dict,
    broker: PaperBroker,
    signals: dict[str, str],
    errors: dict[str, str],
    vix_value: float | None,
    daily_pnl: float,
    risk: RiskManager,
    graduation_unlocked: bool,
) -> Group:
    quotes = Table(title="NSE Watchlist", box=box.SIMPLE)
    for column in ("Symbol", "Price", "Fear", "Regime", "Signal"):
        quotes.add_column(column)
    for ticker in symbols:
        frame = quote_data.get(ticker)
        state = fears.get(ticker)
        price = f"₹{frame['close'].iloc[-1]:,.2f}" if frame is not None and not frame.empty else "—"
        fear = f"{state.fear_score:.2f}" if state else "—"
        regime = state.regime.value if state else "—"
        quotes.add_row(ticker, price, fear, regime, signals.get(ticker, "Waiting"))

    positions = Table(title="Paper Positions", box=box.SIMPLE)
    for column in ("Symbol", "Qty", "Avg Cost", "Mark", "Unrealized P&L", "Stop"):
        positions.add_column(column, justify="right" if column != "Symbol" else "left")
    for position in broker.get_positions().values():
        positions.add_row(
            position.ticker, str(position.quantity), f"₹{position.average_price:,.2f}",
            f"₹{position.current_price:,.2f}", f"₹{position.unrealized_pnl:,.2f}",
            f"₹{position.stop_price:,.2f}" if position.stop_price is not None else "—",
        )
    if not broker.get_positions():
        positions.add_row("—", "—", "—", "—", "—", "—")

    now = datetime.now(config.MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S IST")
    status = (
        f"[bold]PAPER TRADING[/bold]  •  {now}  •  India VIX: {vix_value:.2f}"
        if vix_value is not None
        else f"[bold]PAPER TRADING[/bold]  •  {now}  •  India VIX unavailable"
    )
    footer = (
        f"Cash ₹{broker.cash:,.2f}  |  Portfolio ₹{broker.portfolio_value:,.2f}  |  "
        f"Daily P&L ₹{daily_pnl:,.2f}  |  Fear cooldown {risk.cooldown_remaining} bars  |  "
        f"Daily lock {'ON' if risk.daily_lock_active else 'OFF'}  |  "
        f"Capital kill switch {'ON' if risk.capital_lock_active else 'OFF'}  |  "
        f"Graduation gate {'UNLOCKED' if graduation_unlocked else 'LOCKED'}"
    )
    error_text = "\n".join(f"{key}: {value}" for key, value in errors.items())
    panel = Panel(status + "\n" + footer + ("\n" + error_text if error_text else ""), title="Stock Bot")
    return Group(panel, quotes, positions)


def _evaluate_graduation(broker: PaperBroker, initial_capital: float):
    with database_connection(broker.database_path) as connection:
        trade_rows = connection.execute(
            "SELECT realized_pnl FROM orders WHERE side='SELL' AND status='FILLED' AND realized_pnl IS NOT NULL"
        ).fetchall()
        equity_rows = connection.execute(
            "SELECT ending_equity FROM daily_pnl WHERE ending_equity IS NOT NULL ORDER BY session_date"
        ).fetchall()
    pnl = [float(row["realized_pnl"]) for row in trade_rows]
    win_rate = (sum(value > 0 for value in pnl) / len(pnl) * 100.0) if pnl else 0.0
    equity_curve = [initial_capital] + [float(row["ending_equity"]) for row in equity_rows]
    equity_curve.append(broker.portfolio_value)
    peak = initial_capital
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return GraduationGatekeeper(initial_capital).evaluate(
        broker.portfolio_value, len(pnl), win_rate, max_dd
    )


def _record_daily_pnl(
    broker: PaperBroker, session_date, starting_equity: float, daily_pnl: float
) -> None:
    positions = broker.get_positions().values()
    unrealized = sum(position.unrealized_pnl for position in positions)
    realized = daily_pnl - unrealized
    local_start = config.MARKET_TIMEZONE.localize(datetime.combine(session_date, datetime_time.min))
    local_end = local_start + timedelta(days=1)
    utc_start = local_start.astimezone(timezone.utc).isoformat()
    utc_end = local_end.astimezone(timezone.utc).isoformat()
    with database_connection(broker.database_path) as connection:
        connection.execute(
            """INSERT INTO daily_pnl
               (session_date, starting_equity, ending_equity, realized_pnl, unrealized_pnl, trade_count, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
               ON CONFLICT(session_date) DO UPDATE SET
                 starting_equity=excluded.starting_equity, ending_equity=excluded.ending_equity,
                 realized_pnl=excluded.realized_pnl, unrealized_pnl=excluded.unrealized_pnl,
                 trade_count=excluded.trade_count, updated_at=excluded.updated_at""",
            (
                session_date.isoformat(), starting_equity, broker.portfolio_value,
                realized, unrealized,
                connection.execute(
                    "SELECT COUNT(*) FROM orders WHERE side='SELL' AND status='FILLED' AND created_at >= ? AND created_at < ?",
                    (utc_start, utc_end),
                ).fetchone()[0],
            ),
        )


if __name__ == "__main__":
    raise SystemExit(main())

