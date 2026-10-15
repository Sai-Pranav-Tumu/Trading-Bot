"""Graduation gate for unlocking a broker integration interface."""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class GraduationStatus:
    unlocked: bool
    reasons: tuple[str, ...]
    net_return_pct: float
    closed_trades: int
    win_rate_pct: float
    max_drawdown_pct: float


class GraduationGatekeeper:
    """Evaluate paper performance against configured graduation limits."""

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)

    def evaluate(
        self,
        ending_equity: float,
        closed_trades: int,
        win_rate_pct: float,
        max_drawdown_pct: float,
    ) -> GraduationStatus:
        if ending_equity < 0 or closed_trades < 0 or not 0 <= win_rate_pct <= 100 or max_drawdown_pct < 0:
            raise ValueError("Graduation metrics are outside their valid ranges")
        net_return = (ending_equity / self.initial_capital - 1.0) * 100.0
        rules = config.GRADUATION_RULES
        reasons: list[str] = []
        if ending_equity < self.initial_capital * (1.0 + rules.profit_target_pct / 100.0):
            reasons.append(f"Net return below {rules.profit_target_pct:.1f}% target")
        if closed_trades < rules.min_trades:
            reasons.append(f"Requires at least {rules.min_trades} closed trades")
        if win_rate_pct < rules.min_win_rate_pct:
            reasons.append(f"Win rate below {rules.min_win_rate_pct:.1f}%")
        if max_drawdown_pct > rules.max_drawdown_allowed_pct:
            reasons.append(f"Drawdown above {rules.max_drawdown_allowed_pct:.1f}% limit")
        return GraduationStatus(
            unlocked=not reasons,
            reasons=tuple(reasons),
            net_return_pct=net_return,
            closed_trades=closed_trades,
            win_rate_pct=win_rate_pct,
            max_drawdown_pct=max_drawdown_pct,
        )

