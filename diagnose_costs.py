"""
Diagnostic only: gross vs net walk-forward returns on development folds.

This does not change the trading strategy. It repeats the existing v3 training
process and reports how much fees/slippage appear to matter.
"""
import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from evolve_bots_v2 import (load_prices, build, evaluate, random_pop, breed,
                            multi_eval, HOLDOUT, RECENT, COST)
from evolve_bots_v3 import SEALED_FOLDS, DEV_FOLDS, ENSEMBLE_TOP_N, fold_indices


def bot_daily(pop, F, nxt, cost):
    W, thr, frac = pop
    days = F.shape[0]
    scores = np.einsum("dck,nk->ndc", F, W)
    best, bscore = scores.argmax(2), scores.max(2)
    trade = bscore > thr[:, None]
    coin_ret = nxt[np.arange(days)[None, :], best]
    daily = np.where(trade, frac[:, None] * (coin_ret - cost), 0.0)
    exposure = (trade * frac[:, None]).mean(1)
    trades = trade.sum(1)
    estimated_cost = (trade * frac[:, None] * COST).sum(1)
    return daily, exposure, trades, estimated_cost


def run_fold(F, nxt, end, args, seed):
    rng = np.random.default_rng(seed)
    h0, r0 = end - HOLDOUT, end - HOLDOUT - RECENT
    his_F, his_n = F[:r0].copy(), nxt[:r0]
    rec_F, rec_n = F[r0:h0].copy(), nxt[r0:h0]
    hold_F, hold_n = F[h0:end].copy(), nxt[h0:end]
    mu, sd = his_F[:, :, :-1].mean((0, 1)), his_F[:, :, :-1].std((0, 1)) + 1e-9
    for arr in (his_F, rec_F, hold_F):
        arr[:, :, :-1] = np.clip((arr[:, :, :-1] - mu) / sd, -4, 4)

    random_pop(rng)  # keep the same RNG path as evolve_bots_v3.py
    pop = random_pop(rng)
    for g in range(args.gens):
        use_recent = (g % 12) >= 9
        PF, PN = (rec_F, rec_n) if use_recent else (his_F, his_n)
        mt, md, mn, worst = multi_eval(pop, PF, PN, rng, args.windows)
        fit = mt - 0.5 * md
        ok = (mt * 100 > args.x) & (mn >= 3) & (worst > -0.15)
        surv = np.where(ok)[0]
        if len(surv) < 50:
            surv = np.argsort(-fit)[:100]
        surv = surv[np.argsort(-fit[surv])]
        pop = breed(pop, surv, rng)

    tr_F, tr_n = np.concatenate([his_F, rec_F]), np.concatenate([his_n, rec_n])
    train_t, train_d, _ = evaluate(pop, tr_F, tr_n)
    fitness = train_t - 0.5 * train_d
    champion = int(np.argmax(fitness))
    ensemble_idx = np.argsort(-fitness)[:args.ensemble_top]

    net_daily, _, net_trades, net_cost = bot_daily(pop, hold_F, hold_n, COST)
    gross_daily, _, _, _ = bot_daily(pop, hold_F, hold_n, 0.0)
    net_total = np.prod(1 + net_daily, axis=1) - 1
    median_idx = int(np.argsort(net_total)[len(net_total) // 2])

    def row(label, idxs):
        if np.isscalar(idxs):
            gross = gross_daily[int(idxs)]
            net = net_daily[int(idxs)]
            trades = float(net_trades[int(idxs)])
            cost = float(net_cost[int(idxs)])
        else:
            gross = gross_daily[idxs].mean(0)
            net = net_daily[idxs].mean(0)
            trades = float(net_trades[idxs].mean())
            cost = float(net_cost[idxs].mean())
        return {
            f"{label}_gross_return": float(np.prod(1 + gross) - 1),
            f"{label}_net_return": float(np.prod(1 + net) - 1),
            f"{label}_avg_trades": trades,
            f"{label}_estimated_cost": cost,
        }

    out = {}
    out.update(row("champion", champion))
    out.update(row("evolved", median_idx))
    out.update(row("ensemble", ensemble_idx))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--gens", type=int, default=60)
    parser.add_argument("--x", type=float, default=2.0)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--folds", type=int, default=9)
    parser.add_argument("--sealed-folds", type=int, default=SEALED_FOLDS)
    parser.add_argument("--ensemble-top", type=int, default=ENSEMBLE_TOP_N)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    px = load_prices(args.synthetic)
    F, nxt, dates = build(px)
    total_folds = min(args.folds, (len(F) - RECENT - 90) // HOLDOUT)
    selected = fold_indices(total_folds, "dev", args.sealed_folds)
    out = []
    for k in selected:
        end = len(F) - k * HOLDOUT
        runs = [run_fold(F, nxt, end, args, seed) for seed in range(1, args.seeds + 1)]
        row = {
            "period": f"{dates[end-HOLDOUT].date()}..{dates[end-1].date()}",
            "fold_index": k,
        }
        row.update(pd.DataFrame(runs).median(numeric_only=True).to_dict())
        out.append(row)
        print(f"fold {k} done: {row['period']}")

    df = pd.DataFrame(out)
    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.results_dir, f"cost_diagnostic_dev_{ts}.csv")
    df.to_csv(path, index=False)

    show = df.copy()
    for col in show.columns:
        if col.endswith("_return") or col.endswith("_estimated_cost"):
            show[col] = (show[col] * 100).round(2)
        elif col.endswith("_avg_trades"):
            show[col] = show[col].round(1)
    print("\n=== COST DIAGNOSTIC: DEV FOLDS ONLY ===")
    print("Returns and estimated costs are percentages of starting capital.")
    print(show.to_string(index=False))
    print(f"\nSaved diagnostic: {path}")


if __name__ == "__main__":
    main()
