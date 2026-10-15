"""
Pre-registered Phase 2 feature experiments.

Do not run this until the user confirms. Final folds stay sealed by default.

Experiments:
  baseline    = price-only, identical seeds/settings
  funding     = mean funding settlements in the 24h ending at T - 1h
  fear_greed  = previous available Fear & Greed value / 100, max age 3 days
  controls    = same feature values shuffled across days

Both use only rows with available_at_utc <= T - 1h.
"""
import argparse
import os
from datetime import timezone

import numpy as np
import pandas as pd

from evolve_bots_v2 import (load_prices, evaluate, random_pop, breed, multi_eval,
                            HOLDOUT, RECENT, COST, FEATS)
from evolve_bots_v3 import (SEALED_FOLDS, DEV_FOLDS, DEV_SUCCESS_FOLDS,
                            ENSEMBLE_TOP_N, fold_indices, max_drawdown,
                            sharpe_like, bootstrap_ci)


SAFETY_LAG = pd.Timedelta(hours=1)
FEAR_GREED_MAX_AGE = pd.Timedelta(days=3)
FUNDING_MIN_SETTLEMENTS = 2
FEAR_GREED_NEUTRAL_RAW = 0.5
FUNDING_NEUTRAL_RAW = 0.0


def as_utc(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(timezone.utc)
    return ts.tz_convert(timezone.utc)


def price_row_decision_time(price_row_date, hour_utc):
    """Binance daily candle D closes at D+1 00:00 UTC; decide after that close."""
    return as_utc(pd.Timestamp(price_row_date).normalize() +
                  pd.Timedelta(days=1, hours=hour_utc))


def fear_greed_effective_available(value_date):
    return as_utc(pd.Timestamp(value_date).normalize() + pd.Timedelta(days=1))


def load_funding_feature(dates, coins, data_dir, decision_hour_utc):
    path = os.path.join(data_dir, "funding_rates.csv")
    out = pd.DataFrame(FUNDING_NEUTRAL_RAW, index=dates, columns=coins)
    if not os.path.exists(path):
        return out
    df = pd.read_csv(path, parse_dates=["funding_time_utc", "available_at_utc"])
    df["coin"] = df["symbol"].str.replace("USDT", "", regex=False)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    for date in dates:
        cutoff = price_row_decision_time(date, decision_hour_utc) - SAFETY_LAG
        start = cutoff - pd.Timedelta(hours=24)
        usable = df[(df["available_at_utc"] <= cutoff) &
                    (df["funding_time_utc"] > start) &
                    (df["funding_time_utc"] <= cutoff)]
        grouped = usable.groupby("coin")["funding_rate"]
        means = grouped.mean()
        counts = grouped.count()
        for coin in coins:
            if (coin in means.index and pd.notna(means.loc[coin]) and
                    counts.loc[coin] >= FUNDING_MIN_SETTLEMENTS):
                out.loc[date, coin] = float(means.loc[coin])
    return out


def shuffle_feature_by_day(feature, seed):
    rng = np.random.default_rng(seed)
    shuffled = feature.copy()
    order = rng.permutation(len(shuffled))
    shuffled.iloc[:, :] = shuffled.iloc[order].to_numpy()
    return shuffled


def load_fear_greed_feature(dates, coins, data_dir, decision_hour_utc):
    path = os.path.join(data_dir, "fear_greed.csv")
    out = pd.DataFrame(FEAR_GREED_NEUTRAL_RAW, index=dates, columns=coins)
    if not os.path.exists(path):
        return out
    df = pd.read_csv(path, parse_dates=["value_date_utc", "available_at_utc"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce") / 100.0
    df["effective_available_at_utc"] = df["value_date_utc"].apply(fear_greed_effective_available)
    for date in dates:
        cutoff = price_row_decision_time(date, decision_hour_utc) - SAFETY_LAG
        usable = df[(df["effective_available_at_utc"] <= cutoff) &
                    (df["value_date_utc"] < cutoff)]
        if not usable.empty:
            chosen = usable.sort_values(["value_date_utc", "effective_available_at_utc"]).iloc[-1]
            age = cutoff - chosen["value_date_utc"]
            if age <= FEAR_GREED_MAX_AGE:
                out.loc[date, :] = float(chosen["value"])
    return out


def build_with_external(px, experiment, data_dir, decision_hour_utc, shuffle_seed=777):
    r1 = px.pct_change()
    if experiment == "baseline":
        external = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    elif experiment in ("funding", "funding_shuffled"):
        external = load_funding_feature(px.index, px.columns, data_dir, decision_hour_utc)
        if experiment == "funding_shuffled":
            external = shuffle_feature_by_day(external, shuffle_seed)
    elif experiment in ("fear_greed", "fear_greed_shuffled"):
        external = load_fear_greed_feature(px.index, px.columns, data_dir, decision_hour_utc)
        if experiment == "fear_greed_shuffled":
            external = shuffle_feature_by_day(external, shuffle_seed)
    else:
        raise ValueError(f"unknown experiment: {experiment}")
    features = {
        "ret1": r1,
        "ret3": px.pct_change(3),
        "ret7": px.pct_change(7),
        "ret30": px.pct_change(30),
        "vol7": r1.rolling(7).std(),
        "sentiment": external,
    }
    F = np.stack([features[k].values for k in FEATS], axis=2)
    nxt = (px.shift(-1) / px - 1).values
    ok = ~(np.isnan(F).any((1, 2)) | np.isnan(nxt).any(1))
    return F[ok], nxt[ok], px.index[ok]


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


def summarize(label, gross_daily, net_daily, exposure, trades, estimated_cost,
              same_exposure_passive, rng):
    net_return = float(np.prod(1 + net_daily) - 1)
    ci_low, ci_high = bootstrap_ci(net_daily, rng)
    return {
        f"{label}_gross_return": float(np.prod(1 + gross_daily) - 1),
        f"{label}_net_return": net_return,
        f"{label}_estimated_cost": float(estimated_cost),
        f"{label}_trades": float(trades),
        f"{label}_exposure": float(exposure),
        f"{label}_same_exposure": float(same_exposure_passive),
        f"{label}_max_dd": max_drawdown(net_daily),
        f"{label}_sharpe": sharpe_like(net_daily),
        f"{label}_ci05": ci_low,
        f"{label}_ci95": ci_high,
        f"{label}_beats_cash": net_return > 0,
        f"{label}_beats_same_exposure": net_return > same_exposure_passive,
        f"{label}_success": net_return > 0 and net_return > same_exposure_passive,
    }


def run_fold(F, nxt, end, args, seed):
    rng = np.random.default_rng(seed)
    h0, r0 = end - HOLDOUT, end - HOLDOUT - RECENT
    his_F, his_n = F[:r0].copy(), nxt[:r0]
    rec_F, rec_n = F[r0:h0].copy(), nxt[r0:h0]
    hold_F, hold_n = F[h0:end].copy(), nxt[h0:end]

    # Standardize on historical training data only. Recent and holdout/test
    # periods never influence feature means or standard deviations.
    mu, sd = his_F.mean((0, 1)), his_F.std((0, 1)) + 1e-9
    for arr in (his_F, rec_F, hold_F):
        arr[:] = np.clip((arr - mu) / sd, -4, 4)

    random_pop(rng)  # match v3 RNG path
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

    net_daily, exposure, trades, estimated_cost = bot_daily(pop, hold_F, hold_n, COST)
    gross_daily, _, _, _ = bot_daily(pop, hold_F, hold_n, 0.0)
    net_total = np.prod(1 + net_daily, axis=1) - 1
    evolved = int(np.argsort(net_total)[len(net_total) // 2])
    ew_hold = float(np.prod(1 + hold_n.mean(1)) - 1)

    def one_bot(label, idx):
        return summarize(label, gross_daily[idx], net_daily[idx], exposure[idx],
                         trades[idx], estimated_cost[idx], ew_hold * exposure[idx], rng)

    out = {}
    out.update(one_bot("champion", champion))
    out.update(one_bot("evolved", evolved))
    out.update(summarize(
        "ensemble",
        gross_daily[ensemble_idx].mean(0),
        net_daily[ensemble_idx].mean(0),
        float(exposure[ensemble_idx].mean()),
        float(trades[ensemble_idx].mean()),
        float(estimated_cost[ensemble_idx].mean()),
        ew_hold * float(exposure[ensemble_idx].mean()),
        rng,
    ))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=[
        "baseline", "funding", "fear_greed", "funding_shuffled", "fear_greed_shuffled",
    ], required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--gens", type=int, default=60)
    parser.add_argument("--x", type=float, default=2.0)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--folds", type=int, default=9)
    parser.add_argument("--sealed-folds", type=int, default=SEALED_FOLDS)
    parser.add_argument("--ensemble-top", type=int, default=ENSEMBLE_TOP_N)
    parser.add_argument("--decision-hour-utc", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--shuffle-seed", type=int, default=777)
    args = parser.parse_args()

    px = load_prices(args.synthetic)
    F, nxt, dates = build_with_external(px, args.experiment, args.data_dir,
                                        args.decision_hour_utc, args.shuffle_seed)
    total_folds = min(args.folds, (len(F) - RECENT - 90) // HOLDOUT)
    selected = fold_indices(total_folds, "dev", args.sealed_folds)
    out = []
    for k in selected:
        end = len(F) - k * HOLDOUT
        runs = [run_fold(F, nxt, end, args, seed) for seed in range(1, args.seeds + 1)]
        row = {
            "experiment": args.experiment,
            "period": f"{dates[end-HOLDOUT].date()}..{dates[end-1].date()}",
            "fold_index": k,
        }
        row.update(pd.DataFrame(runs).median(numeric_only=True).to_dict())
        out.append(row)
        print(f"fold {k} done: {row['period']}")

    df = pd.DataFrame(out)
    os.makedirs(args.results_dir, exist_ok=True)
    suffix = f"_shuffle{args.shuffle_seed}" if args.experiment.endswith("_shuffled") else ""
    path = os.path.join(args.results_dir, f"phase2_{args.experiment}{suffix}_dev.csv")
    df.to_csv(path, index=False)
    successes = int(df["ensemble_success"].sum())
    show_cols = [
        "period", "ensemble_gross_return", "ensemble_net_return",
        "ensemble_ci05", "ensemble_ci95", "ensemble_same_exposure",
        "ensemble_success",
    ]
    show = df[show_cols].copy()
    for col in show.columns:
        if col != "period" and col != "ensemble_success":
            show[col] = (show[col] * 100).round(2)
    print(show.to_string(index=False))
    print(f"ensemble success folds: {successes}/{len(df)}")
    print(f"success rule: {DEV_SUCCESS_FOLDS}/{DEV_FOLDS}")
    print("primary_metric=top25_ensemble_net_return")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
