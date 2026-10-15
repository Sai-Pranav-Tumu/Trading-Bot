"""
Walk-forward test with sealed folds and richer metrics.

Default behavior is development mode:
  - the latest 2 folds are sealed
  - only older folds are used for development comparisons
  - success is defined before running: beat cash and same-exposure passive
    in at least 6 of 7 development folds

Usage:
  python evolve_bots_v3.py --synthetic --folds 4 --seeds 2 --gens 24
  python evolve_bots_v3.py --mode dev --folds 9 --seeds 3
  python evolve_bots_v3.py --mode final --folds 9 --seeds 3   # only when approved
"""
import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from evolve_bots_v2 import (load_prices, build, evaluate, random_pop, breed,
                            multi_eval, HOLDOUT, RECENT, WINDOW, COST)


SEALED_FOLDS = 2
DEV_SUCCESS_FOLDS = 6
DEV_FOLDS = 7
ENSEMBLE_TOP_N = 25


def bot_daily_returns(pop, F, nxt):
    """Return daily returns, exposure, and trade counts for each bot."""
    W, thr, frac = pop
    days = F.shape[0]
    scores = np.einsum("dck,nk->ndc", F, W)
    best, bscore = scores.argmax(2), scores.max(2)
    trade = bscore > thr[:, None]
    coin_ret = nxt[np.arange(days)[None, :], best]
    daily = np.where(trade, frac[:, None] * (coin_ret - COST), 0.0)
    expo = (trade * frac[:, None]).mean(1)
    trades = trade.sum(1)
    return daily, expo, trades


def max_drawdown(daily):
    eq = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(eq)
    return float(((peak - eq) / peak).max())


def sharpe_like(daily):
    sd = daily.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(np.sqrt(365) * daily.mean() / sd)


def bootstrap_ci(daily, rng, n=1000):
    """Simple day-resampling interval. It is noisy, not a proof of skill."""
    if len(daily) == 0:
        return 0.0, 0.0
    idx = rng.integers(0, len(daily), size=(n, len(daily)))
    totals = np.prod(1 + daily[idx], axis=1) - 1
    lo, hi = np.quantile(totals, [0.05, 0.95])
    return float(lo), float(hi)


def summarize_daily(prefix, daily, exposure, trades, passive_same_exposure, rng):
    total = float(np.prod(1 + daily) - 1)
    ci_low, ci_high = bootstrap_ci(daily, rng)
    return {
        f"{prefix}_return": total,
        f"{prefix}_max_dd": max_drawdown(daily),
        f"{prefix}_sharpe": sharpe_like(daily),
        f"{prefix}_trades": float(trades),
        f"{prefix}_turnover": float(trades * exposure),
        f"{prefix}_exposure": float(exposure),
        f"{prefix}_ci05": ci_low,
        f"{prefix}_ci95": ci_high,
        f"{prefix}_beats_cash": total > 0,
        f"{prefix}_beats_same_exposure": total > passive_same_exposure,
    }


def run_fold(F, nxt, end, a, seed):
    rng = np.random.default_rng(seed)
    h0, r0 = end - HOLDOUT, end - HOLDOUT - RECENT
    his_F, his_n = F[:r0].copy(), nxt[:r0]
    rec_F, rec_n = F[r0:h0].copy(), nxt[r0:h0]
    hold_F, hold_n = F[h0:end].copy(), nxt[h0:end]
    mu, sd = his_F[:, :, :-1].mean((0, 1)), his_F[:, :, :-1].std((0, 1)) + 1e-9
    for arr in (his_F, rec_F, hold_F):
        arr[:, :, :-1] = np.clip((arr[:, :, :-1] - mu) / sd, -4, 4)

    baseline, pop = random_pop(rng), random_pop(rng)
    for g in range(a.gens):
        use_recent = (g % 12) >= 9
        PF, PN = (rec_F, rec_n) if use_recent else (his_F, his_n)
        mt, md, mn, worst = multi_eval(pop, PF, PN, rng, a.windows)
        fit = mt - 0.5 * md
        ok = (mt * 100 > a.x) & (mn >= 3) & (worst > -0.15)
        surv = np.where(ok)[0]
        if len(surv) < 50:
            surv = np.argsort(-fit)[:100]
        surv = surv[np.argsort(-fit[surv])]
        pop = breed(pop, surv, rng)

    tr_F, tr_n = np.concatenate([his_F, rec_F]), np.concatenate([his_n, rec_n])
    t, d, _ = evaluate(pop, tr_F, tr_n)
    fitness = t - 0.5 * d
    champ = int(np.argmax(fitness))
    ensemble_idx = np.argsort(-fitness)[:a.ensemble_top]

    bot_daily, bot_expo, bot_trades = bot_daily_returns(pop, hold_F, hold_n)
    random_daily, random_expo, random_trades = bot_daily_returns(baseline, hold_F, hold_n)
    h_t = np.prod(1 + bot_daily, axis=1) - 1
    r_t = np.prod(1 + random_daily, axis=1) - 1
    evolved_idx = int(np.argsort(h_t)[len(h_t) // 2])
    random_idx = int(np.argsort(r_t)[len(r_t) // 2])

    ensemble_daily = bot_daily[ensemble_idx].mean(0)
    ensemble_expo = float(bot_expo[ensemble_idx].mean())
    ensemble_trades = float(bot_trades[ensemble_idx].mean())
    ew_hold = float(np.prod(1 + hold_n.mean(1)) - 1)

    out = {}
    out.update(summarize_daily("champion", bot_daily[champ], bot_expo[champ],
                               bot_trades[champ], ew_hold * bot_expo[champ], rng))
    out.update(summarize_daily("evolved", bot_daily[evolved_idx], bot_expo[evolved_idx],
                               bot_trades[evolved_idx], ew_hold * bot_expo[evolved_idx], rng))
    out.update(summarize_daily("ensemble", ensemble_daily, ensemble_expo,
                               ensemble_trades, ew_hold * ensemble_expo, rng))
    out.update({
        "random_return": float(r_t[random_idx]),
        "random_exposure": float(random_expo[random_idx]),
        "random_trades": float(random_trades[random_idx]),
    })
    return out


def fold_indices(total_folds, mode, sealed_folds):
    if mode == "all":
        return list(range(total_folds))
    if mode == "final":
        return list(range(min(sealed_folds, total_folds)))
    start = min(sealed_folds, total_folds)
    stop = min(total_folds, start + DEV_FOLDS)
    return list(range(start, stop))


def write_changelog(path):
    if os.path.exists(path):
        return
    text = """# Changelog

## 2026-10-14 - Phase 1 protocol started

- Added a sealed-fold protocol for walk-forward testing.
- Latest 2 folds are reserved for one final evaluation and must not guide development.
- Development comparisons use the older 7 folds only.
- Success is defined in advance: a change counts only if it beats cash and the same-exposure passive benchmark in at least 6 of 7 development folds.
- Phase 1 is limited to at most 3 experiments.
- Experiment A: metrics, timestamped results, and this changelog.
- Experiment B: predefined top-25 ensemble vs champion.
- Experiments C and D are intentionally not started yet.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_results(df, a, dates):
    os.makedirs(a.results_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(a.results_dir, f"walkforward_{a.mode}_{ts}.csv")
    summary_path = os.path.join(a.results_dir, f"summary_{a.mode}_{ts}.txt")
    df.to_csv(csv_path, index=False)

    ensemble_ok = int(df.ensemble_success.sum())
    champion_ok = int(df.champion_success.sum())
    evolved_ok = int(df.evolved_success.sum())
    verdict = "PASS" if a.mode == "dev" and ensemble_ok >= DEV_SUCCESS_FOLDS else "FAIL"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"mode: {a.mode}\n")
        f.write(f"date_range: {dates[0].date()}..{dates[-1].date()}\n")
        f.write(f"sealed_latest_folds: {a.sealed_folds}\n")
        f.write(f"development_success_rule: {DEV_SUCCESS_FOLDS}/{DEV_FOLDS}\n")
        f.write(f"champion_success_folds: {champion_ok}/{len(df)}\n")
        f.write(f"evolved_success_folds: {evolved_ok}/{len(df)}\n")
        f.write(f"ensemble_success_folds: {ensemble_ok}/{len(df)}\n")
        f.write(f"ensemble_verdict: {verdict}\n")
    return csv_path, summary_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--x", type=float, default=2.0)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=9)
    ap.add_argument("--mode", choices=["dev", "final", "all"], default="dev")
    ap.add_argument("--sealed-folds", type=int, default=SEALED_FOLDS)
    ap.add_argument("--ensemble-top", type=int, default=ENSEMBLE_TOP_N)
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()

    if a.mode == "all":
        print("WARNING: --mode all includes sealed folds. Do not use it for development decisions.")
    if a.mode == "final":
        print("FINAL MODE: using sealed folds only. Run this only after development is frozen.")

    write_changelog("CHANGELOG.md")
    px = load_prices(a.synthetic)
    F, nxt, dates = build(px)
    total_folds = min(a.folds, (len(F) - RECENT - 90) // HOLDOUT)
    selected_folds = fold_indices(total_folds, a.mode, a.sealed_folds)
    if not selected_folds:
        raise RuntimeError("No folds selected. Download more history or lower --sealed-folds.")

    out = []
    for k in selected_folds:
        end = len(F) - k * HOLDOUT
        if end - HOLDOUT - RECENT < 90:
            print(f"stopping at fold {k}: not enough earlier data (download more history)")
            break
        runs = [run_fold(F, nxt, end, a, s) for s in range(1, a.seeds + 1)]
        hn = nxt[end - HOLDOUT:end]
        btc_hold = float(np.prod(1 + hn[:, 0]) - 1)
        ew_hold = float(np.prod(1 + hn.mean(1)) - 1)
        r = pd.DataFrame(runs).median(numeric_only=True)
        row = dict(period=f"{dates[end-HOLDOUT].date()}..{dates[end-1].date()}",
                   fold_index=k, btc_hold=btc_hold, ew_hold=ew_hold)
        row.update(r.to_dict())
        row["champion_same_exposure"] = ew_hold * row["champion_exposure"]
        row["evolved_same_exposure"] = ew_hold * row["evolved_exposure"]
        row["ensemble_same_exposure"] = ew_hold * row["ensemble_exposure"]
        row["champion_beats_cash"] = row["champion_return"] > 0
        row["evolved_beats_cash"] = row["evolved_return"] > 0
        row["ensemble_beats_cash"] = row["ensemble_return"] > 0
        row["champion_beats_same_exposure"] = row["champion_return"] > row["champion_same_exposure"]
        row["evolved_beats_same_exposure"] = row["evolved_return"] > row["evolved_same_exposure"]
        row["ensemble_beats_same_exposure"] = row["ensemble_return"] > row["ensemble_same_exposure"]
        row["champion_success"] = bool(row["champion_beats_cash"] and row["champion_beats_same_exposure"])
        row["evolved_success"] = bool(row["evolved_beats_cash"] and row["evolved_beats_same_exposure"])
        row["ensemble_success"] = bool(row["ensemble_beats_cash"] and row["ensemble_beats_same_exposure"])
        out.append(row)
        print(f"fold {k} done: {row['period']}")

    df = pd.DataFrame(out)
    csv_path, summary_path = save_results(df, a, dates)

    show_cols = [
        "period", "champion_return", "evolved_return", "ensemble_return",
        "random_return", "ensemble_exposure", "ensemble_same_exposure",
        "ensemble_max_dd", "ensemble_sharpe", "ensemble_success",
    ]
    show = df[show_cols].copy()
    for c in show.columns:
        if c != "period" and c != "ensemble_success":
            show[c] = (show[c] * 100).round(1) if c != "ensemble_sharpe" else show[c].round(2)

    print("\n=== WALK-FORWARD DEVELOPMENT REPORT ===")
    print("All return, exposure, and drawdown numbers are percentages unless noted.")
    print(show.to_string(index=False))

    for label in ("champion", "evolved", "ensemble"):
        ok = int(df[f"{label}_success"].sum())
        print(f"{label:9s} beats cash and same-exposure passive: {ok}/{len(df)}")
    print(f"\nSuccess rule for a Phase 1 change: {DEV_SUCCESS_FOLDS}/{DEV_FOLDS} development folds.")
    print(f"Saved results: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
