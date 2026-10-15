"""Historical event-driven paper backtester with risk and fear controls."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
from indicators.fear_engine import FearEngine, FearRegime
from risk_and_execution.broker_adapter import PaperBroker
from risk_and_execution.risk_manager import RiskManager
from strategies.base_strategy import SignalType
from strategies.mean_reversion import MeanReversionStrategy
from strategies.vwap_momentum import VWAPMomentumStrategy


@dataclass(frozen=True)
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return_pct: float
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    capital_saved_by_fear: float
    closed_trades: int
    winning_trades: int
    losing_trades: int


class Backtester:
    """Run two strategies over preloaded OHLCV frames without lookahead."""

    def __init__(
        self,
        initial_capital: float = config.INITIAL_CAPITAL,
        risk_manager: RiskManager | None = None,
        database_directory: str | Path | None = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)
        self.risk_manager = risk_manager or RiskManager(initial_capital=self.initial_capital)
        self.database_directory = Path(database_directory) if database_directory else None

    def run(
        self,
        ohlcv_by_ticker: dict[str, pd.DataFrame],
        india_vix_data: pd.DataFrame | pd.Series | None = None,
    ) -> BacktestResult:
        """Simulate available symbols in timestamp order and return portfolio metrics."""
        frames = {
            ticker: frame.sort_index()
            for ticker, frame in ohlcv_by_ticker.items()
            if isinstance(frame, pd.DataFrame) and not frame.empty
        }
        if not frames:
            raise ValueError("At least one non-empty OHLCV frame is required")

        all_events = pd.concat(
            [frame.assign(_ticker=ticker) for ticker, frame in frames.items()],
            axis=0,
        ).sort_index(kind="stable")
        fear_engine = FearEngine()
        momentum = VWAPMomentumStrategy(fear_engine)
        mean_reversion = MeanReversionStrategy(fear_engine)
        completed_pnls: list[float] = []
        fear_exits: list[tuple[str, float, int]] = []
        peak_equity = self.initial_capital
        max_drawdown = 0.0
        equity_snapshots: dict[object, float] = {}
        current_session = None
        session_open_equity = self.initial_capital
        last_cooldown_timestamp = None
        previous_regime: dict[str, FearRegime] = {}
        last_prices: dict[str, float] = {}

        def simulate(directory: Path) -> BacktestResult:
            nonlocal current_session, session_open_equity, peak_equity, max_drawdown
            nonlocal last_cooldown_timestamp
            db_path = directory / "backtest.sqlite3"
            broker = PaperBroker(self.initial_capital, db_path)
            for timestamp, row in all_events.iterrows():
                ticker = str(row["_ticker"])
                price = float(row["close"])
                last_prices[ticker] = price
                session = timestamp.date() if hasattr(timestamp, "date") else None
                if session != current_session:
                    self.risk_manager.reset_session()
                    current_session = session
                    session_open_equity = broker.portfolio_value
                    last_cooldown_timestamp = None
                if timestamp != last_cooldown_timestamp:
                    self.risk_manager.advance_candle()
                    last_cooldown_timestamp = timestamp

                for open_ticker in list(broker.get_positions()):
                    if open_ticker in last_prices:
                        broker.update_market_price(open_ticker, last_prices[open_ticker])

                daily_pnl = broker.portfolio_value - session_open_equity
                if self.risk_manager.daily_loss_triggered(daily_pnl, session_open_equity):
                    self.risk_manager.enforce_daily_loss(broker, daily_pnl, session_open_equity)
                self.risk_manager.enforce_portfolio_kill_switch(
                    broker, broker.portfolio_value
                )

                frame = frames[ticker]
                try:
                    row_index = frame.index.get_loc(timestamp)
                except KeyError:
                    continue
                if isinstance(row_index, slice):
                    row_index = row_index.stop - 1
                elif isinstance(row_index, np.ndarray):
                    row_index = int(row_index[-1])
                history = frame.iloc[max(0, int(row_index) - 99) : int(row_index) + 1]
                if len(history) < 21:
                    continue

                vix_at_time = self._vix_as_of(india_vix_data, timestamp)
                portfolio_value = broker.portfolio_value
                drawdown_pct = max(0.0, (peak_equity - portfolio_value) / peak_equity * 100.0)
                try:
                    fear = fear_engine.compute(history, vix_at_time, drawdown_pct)
                except (ValueError, KeyError):
                    continue

                position = broker.get_position(ticker)
                previous = previous_regime.get(ticker, FearRegime.NORMAL)
                if previous is not fear.regime and fear.regime is FearRegime.HIGH_FEAR_PULLBACK and position:
                    pulled = self.risk_manager.apply_fear_pullback(broker, ticker, price, fear)
                    if pulled and pulled.status == "FILLED":
                        completed_pnls.append(float(pulled.realized_pnl or 0.0))
                        fear_exits.append((ticker, float(pulled.fill_price), pulled.quantity))
                elif fear.regime is FearRegime.EXTREME_PANIC and position:
                    pulled = self.risk_manager.apply_fear_pullback(broker, ticker, price, fear)
                    if pulled and pulled.status == "FILLED":
                        completed_pnls.append(float(pulled.realized_pnl or 0.0))
                        fear_exits.append((ticker, float(pulled.fill_price), pulled.quantity))
                previous_regime[ticker] = fear.regime

                position = broker.get_position(ticker)
                if position:
                    stop = self.risk_manager.ratchet_trailing_stop(
                        price, fear.current_atr, fear, position.stop_price
                    )
                    broker.set_stop_price(ticker, stop)
                    if price <= stop:
                        exit_order = broker.sell(ticker, position.quantity, price)
                        if exit_order.status == "FILLED":
                            completed_pnls.append(float(exit_order.realized_pnl or 0.0))
                    elif broker.get_position(ticker):
                        momentum_exit = momentum.generate_signal(
                            ticker, history, fear_state=fear, has_open_position=True
                        )
                        reversion_exit = mean_reversion.generate_signal(
                            ticker, history, fear_state=fear, has_open_position=True
                        )
                        if any(signal.action is SignalType.SELL for signal in (momentum_exit, reversion_exit)):
                            exit_order = broker.sell(
                                ticker, position.quantity, price
                            )
                            if exit_order.status == "FILLED":
                                completed_pnls.append(float(exit_order.realized_pnl or 0.0))

                if not broker.get_position(ticker) and self.risk_manager.can_open_position(
                    broker.portfolio_value - session_open_equity, session_open_equity
                ):
                    momentum_signal = momentum.generate_signal(
                        ticker, history, fear_state=fear, has_open_position=False
                    )
                    reversion_signal = mean_reversion.generate_signal(
                        ticker, history, fear_state=fear, has_open_position=False
                    )
                    selected = next(
                        (signal for signal in (momentum_signal, reversion_signal) if signal.action is SignalType.BUY),
                        None,
                    )
                    if selected:
                        stop_price = price - fear.current_atr * fear.stop_loss_atr_multiplier
                        quantity = self.risk_manager.calculate_quantity(
                            broker.portfolio_value, price, stop_price, fear
                        )
                        if quantity > 0:
                            entry = broker.buy(ticker, quantity, price)
                            if entry.status == "FILLED":
                                broker.set_stop_price(ticker, stop_price)

                equity = broker.portfolio_value
                equity_snapshots[timestamp] = equity
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity * 100.0)

            for ticker, position in list(broker.get_positions().items()):
                final_price = last_prices.get(ticker, position.current_price)
                order = broker.sell(ticker, position.quantity, final_price)
                if order.status == "FILLED":
                    completed_pnls.append(float(order.realized_pnl or 0.0))

            final_equity = broker.portfolio_value
            peak_equity = max(peak_equity, final_equity)
            if peak_equity > 0:
                max_drawdown = max(max_drawdown, (peak_equity - final_equity) / peak_equity * 100.0)
            if equity_snapshots:
                equity_snapshots[max(equity_snapshots)] = final_equity
            saved = 0.0
            for ticker, sell_fill, quantity in fear_exits:
                saved += max(0.0, (sell_fill - last_prices.get(ticker, sell_fill)) * quantity)
            wins = sum(pnl > 0 for pnl in completed_pnls)
            losses = sum(pnl <= 0 for pnl in completed_pnls)
            trade_count = len(completed_pnls)
            ordered_equity = [self.initial_capital] + [
                equity_snapshots[key] for key in sorted(equity_snapshots)
            ]
            returns = np.diff(np.asarray(ordered_equity, dtype=float)) / np.maximum(
                np.asarray(ordered_equity[:-1], dtype=float), 1e-12
            )
            std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
            sharpe = (
                float(np.mean(returns) / std * math.sqrt(252 * 25))
                if std > 0
                else 0.0
            )
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_equity=final_equity,
                total_return_pct=(final_equity / self.initial_capital - 1) * 100.0,
                win_rate_pct=(wins / trade_count * 100.0) if trade_count else 0.0,
                max_drawdown_pct=max_drawdown,
                sharpe_ratio=sharpe,
                capital_saved_by_fear=saved,
                closed_trades=trade_count,
                winning_trades=wins,
                losing_trades=losses,
            )

        if self.database_directory:
            self.database_directory.mkdir(parents=True, exist_ok=True)
            return simulate(self.database_directory)
        with tempfile.TemporaryDirectory(prefix="stockbot-backtest-") as temp_dir:
            return simulate(Path(temp_dir))

    @staticmethod
    def _vix_as_of(
        data: pd.DataFrame | pd.Series | None, timestamp
    ) -> pd.DataFrame | pd.Series | None:
        if data is None or data.empty:
            return None
        return data.loc[data.index <= timestamp]

