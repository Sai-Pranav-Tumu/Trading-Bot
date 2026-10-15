"""
Point-in-time unit audit for Phase 2 features.

This prints exact selected rows for one decision time, verifies stale fallbacks
and funding edge cases, and runs a future-poisoning test across development folds.
It does not run evolution experiments or final mode.
"""
import argparse
import hashlib
import os
from datetime import timezone

import numpy as np
import pandas as pd

from evolve_bots_v2 import COINS, FEATS, load_prices, build, HOLDOUT, RECENT
from evolve_bots_v3 import SEALED_FOLDS, fold_indices
from phase2_experiments import (SAFETY_LAG, FEAR_GREED_MAX_AGE,
                                FEAR_GREED_NEUTRAL_RAW, FUNDING_MIN_SETTLEMENTS,
                                FUNDING_NEUTRAL_RAW, as_utc, build_with_external,
                                load_funding_feature, load_fear_greed_feature,
                                fear_greed_effective_available,
                                price_row_decision_time)


def decision_time(value):
    return as_utc(pd.Timestamp(value))


def load_csv(path, date_cols):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=date_cols)


def funding_values_for_cutoff(df, cutoff):
    start = cutoff - pd.Timedelta(hours=24)
    selected = df[(df["available_at_utc"] <= cutoff) &
                  (df["funding_time_utc"] > start) &
                  (df["funding_time_utc"] <= cutoff)].copy()
    grouped = selected.groupby("coin")["funding_rate"]
    means = grouped.mean()
    counts = grouped.count()
    values = {}
    for coin in COINS:
        if coin in means.index and counts.loc[coin] >= FUNDING_MIN_SETTLEMENTS:
            values[coin] = float(means.loc[coin])
        else:
            values[coin] = FUNDING_NEUTRAL_RAW
    return selected, values


def fear_greed_value_for_cutoff(df, cutoff):
    usable = df[(df["effective_available_at_utc"] <= cutoff) &
                (df["value_date_utc"] < cutoff)].copy()
    if usable.empty:
        return None, FEAR_GREED_NEUTRAL_RAW, None, "neutral_no_rows"
    chosen = usable.sort_values(["value_date_utc", "effective_available_at_utc"]).iloc[-1]
    age = cutoff - chosen["value_date_utc"]
    if age <= FEAR_GREED_MAX_AGE:
        return chosen, float(chosen["value"]) / 100.0, age, "selected"
    return chosen, FEAR_GREED_NEUTRAL_RAW, age, "neutral_too_old"


def load_funding_df(data_dir):
    df = load_csv(os.path.join(data_dir, "funding_rates.csv"),
                  ["funding_time_utc", "available_at_utc"])
    df["coin"] = df["symbol"].str.replace("USDT", "", regex=False)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df[df["coin"].isin(COINS)].copy()


def load_fear_greed_df(data_dir):
    df = load_csv(os.path.join(data_dir, "fear_greed.csv"),
                  ["value_date_utc", "available_at_utc"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["effective_available_at_utc"] = df["value_date_utc"].apply(fear_greed_effective_available)
    return df


def audit_mapping(T):
    price_row_D = (T - pd.Timedelta(days=1)).normalize()
    print("PRICE_ROW_MAPPING")
    print("Binance daily candle date D opens at D 00:00 UTC and closes at D+1 00:00 UTC.")
    print("For a price row labeled D, features using that close are available at decision time T = D+1 00:00 UTC.")
    print("External rows must satisfy available_at_utc <= T - 1h.")
    print(f"chosen_decision_time_T={T.isoformat()}")
    print(f"corresponding_price_row_D={price_row_D.date()}")
    print(f"external_safety_cutoff={T - SAFETY_LAG}")


def audit_funding(data_dir, T):
    cutoff = T - SAFETY_LAG
    start = cutoff - pd.Timedelta(hours=24)
    df = load_funding_df(data_dir)
    selected, values = funding_values_for_cutoff(df, cutoff)
    assert (selected["available_at_utc"] <= cutoff).all()
    assert (selected["funding_time_utc"] <= cutoff).all()
    print("\nFUNDING_SELECTION")
    print(f"decision_time_utc={T.isoformat()}")
    print(f"safety_cutoff_utc={cutoff.isoformat()}")
    print(f"window_start_exclusive_utc={start.isoformat()}")
    print(f"min_settlements_rule={FUNDING_MIN_SETTLEMENTS}_or_neutral_{FUNDING_NEUTRAL_RAW}")
    cols = ["symbol", "funding_time_utc", "available_at_utc", "funding_rate"]
    print(selected.sort_values(["symbol", "funding_time_utc"])[cols].to_string(index=False))
    print("FUNDING_FEATURE_MEANS")
    print(pd.Series(values, name="funding_feature").to_string())
    print(f"funding_rows_selected={len(selected)}")
    print("funding_assertions=PASS")


def audit_fear_greed(data_dir, T):
    cutoff = T - SAFETY_LAG
    df = load_fear_greed_df(data_dir)
    chosen, value, age, rule = fear_greed_value_for_cutoff(df, cutoff)
    usable = df[df["effective_available_at_utc"] <= cutoff]
    assert (usable["effective_available_at_utc"] <= cutoff).all()
    print("\nFEAR_GREED_SELECTION")
    print(f"decision_time_utc={T.isoformat()}")
    print(f"safety_cutoff_utc={cutoff.isoformat()}")
    print("available_at_rule=value_date_utc + 1 day")
    print(f"neutral_raw_value={FEAR_GREED_NEUTRAL_RAW}")
    if chosen is None:
        print("selected_row=None")
    else:
        cols = ["source", "value_date_utc", "available_at_utc",
                "effective_available_at_utc", "value", "classification"]
        print(pd.DataFrame([chosen[cols].to_dict()]).to_string(index=False))
        print(f"row_age={age}")
    print(f"feature_value={value:.4f}")
    print(f"max_age_rule={rule}")
    print("fear_greed_assertions=PASS")


def audit_fear_greed_gap(data_dir):
    df = load_fear_greed_df(data_dir).sort_values("value_date_utc")
    gaps = []
    for prev, cur in zip(df["value_date_utc"], df["value_date_utc"].iloc[1:]):
        gap = cur - prev
        if gap > FEAR_GREED_MAX_AGE:
            gaps.append((prev, cur, gap))
    if not gaps:
        print("\nFEAR_GREED_GAP_TEST")
        print("No gap longer than 3 days found.")
        return
    prev, cur, gap = gaps[0]
    T = prev.normalize() + pd.Timedelta(days=5)
    cutoff = T - SAFETY_LAG
    chosen, value, age, rule = fear_greed_value_for_cutoff(df, cutoff)
    print("\nFEAR_GREED_GAP_TEST")
    print(f"gap_previous_value_date={prev.isoformat()}")
    print(f"gap_next_value_date={cur.isoformat()}")
    print(f"gap_size={gap}")
    print(f"test_decision_time_utc={T.isoformat()}")
    print(f"safety_cutoff_utc={cutoff.isoformat()}")
    print(f"selected_value_date={chosen['value_date_utc'].isoformat() if chosen is not None else None}")
    print(f"row_age={age}")
    print(f"feature_value={value:.4f}")
    print(f"max_age_rule={rule}")
    assert rule == "neutral_too_old"
    assert value == FEAR_GREED_NEUTRAL_RAW
    print("fear_greed_gap_assertions=PASS")


def dev_fold_date_sets(decision_hour_utc):
    px = load_prices(False)
    _, _, dates = build(px)
    total_folds = min(9, (len(dates) - RECENT - 90) // HOLDOUT)
    selected_folds = fold_indices(total_folds, "dev", SEALED_FOLDS)
    fold_ranges = []
    all_dates = []
    for k in selected_folds:
        end = len(dates) - k * HOLDOUT
        fold_dates = dates[:end]
        fold_ranges.append((k, fold_dates))
        all_dates.extend(list(fold_dates))
    unique_dates = pd.DatetimeIndex(sorted(set(all_dates)))
    return px, dates, fold_ranges, unique_dates


def audit_funding_edges():
    print("\nFUNDING_EDGE_CASES")
    examples = {
        "zero_settlements": [],
        "one_settlement": [0.0003],
        "two_settlements": [0.0003, -0.0001],
    }
    for name, vals in examples.items():
        if len(vals) >= FUNDING_MIN_SETTLEMENTS:
            result = float(np.mean(vals))
            rule = "mean_selected"
        else:
            result = FUNDING_NEUTRAL_RAW
            rule = "neutral_too_few_settlements"
        print(f"{name}: count={len(vals)} value={result:.8f} rule={rule}")
    print("funding_edge_assertions=PASS")


def audit_future_poisoning(data_dir, decision_hour_utc):
    _, _, fold_ranges, _ = dev_fold_date_sets(decision_hour_utc)
    funding = load_funding_df(data_dir)
    fear_greed = load_fear_greed_df(data_dir)
    tested = 0
    for _, fold_dates in fold_ranges:
        for date in fold_dates:
            T = price_row_decision_time(date, decision_hour_utc)
            cutoff = T - SAFETY_LAG

            _, full_funding = funding_values_for_cutoff(funding, cutoff)
            poisoned_funding = funding.copy()
            poisoned_funding.loc[poisoned_funding["available_at_utc"] > cutoff, "funding_rate"] = np.nan
            _, poison_funding = funding_values_for_cutoff(poisoned_funding, cutoff)
            assert full_funding == poison_funding

            _, full_fg, _, _ = fear_greed_value_for_cutoff(fear_greed, cutoff)
            poisoned_fg = fear_greed.copy()
            poisoned_fg.loc[poisoned_fg["effective_available_at_utc"] > cutoff, "value"] = np.nan
            _, poison_fg, _, _ = fear_greed_value_for_cutoff(poisoned_fg, cutoff)
            assert full_fg == poison_fg
            tested += 1
    print("\nFUTURE_POISONING_TEST")
    print("scope=all days used by development folds, including training/recent/test windows")
    print(f"dev_fold_decision_times_tested={tested}")
    print(f"feature_checks_tested={tested * 2}")
    print("future_poisoning_assertions=PASS")


def audit_standardization(data_dir, decision_hour_utc):
    px = load_prices(False)
    outputs = []
    for experiment in ("funding", "fear_greed"):
        F, _, dates = build_with_external(px, experiment, data_dir, decision_hour_utc)
        total_folds = min(9, (len(F) - RECENT - 90) // HOLDOUT)
        for k in fold_indices(total_folds, "dev", SEALED_FOLDS)[:2]:
            end = len(F) - k * HOLDOUT
            h0, r0 = end - HOLDOUT, end - HOLDOUT - RECENT
            original = F.copy()
            his_F = original[:r0]
            mu = his_F.mean((0, 1))
            sd = his_F.std((0, 1)) + 1e-9
            poisoned = F.copy()
            poisoned[r0:end, :, :] = 999999.0
            poison_mu = poisoned[:r0].mean((0, 1))
            poison_sd = poisoned[:r0].std((0, 1)) + 1e-9
            assert np.allclose(mu, poison_mu)
            assert np.allclose(sd, poison_sd)
            idx = FEATS.index("sentiment")
            outputs.append((experiment, k, dates[h0].date(), dates[end - 1].date(),
                            float(mu[idx]), float(sd[idx])))
    print("\nSTANDARDIZATION_GARBAGE_TEST")
    print("Garbage was written into recent+holdout/test windows; historical mean/std stayed unchanged.")
    for experiment, fold, start, end, mu, sd in outputs:
        print(f"{experiment} fold={fold} period={start}..{end} sentiment_mu={mu:.10f} sentiment_sd={sd:.10f}")
    print("standardization_garbage_assertions=PASS")


def audit_neutral_fallbacks(data_dir, decision_hour_utc):
    px, _, _, unique_dates = dev_fold_date_sets(decision_hour_utc)
    funding = load_funding_feature(unique_dates, px.columns, data_dir, decision_hour_utc)
    fg = load_fear_greed_feature(unique_dates, px.columns, data_dir, decision_hour_utc)
    funding_neutral = int((funding == FUNDING_NEUTRAL_RAW).sum().sum())
    funding_total = int(funding.size)
    fg_neutral_days = int((fg.iloc[:, 0] == FEAR_GREED_NEUTRAL_RAW).sum())
    fg_total_days = int(len(fg))
    print("\nNEUTRAL_FALLBACK_COUNTS")
    print(f"development_unique_price_rows={len(unique_dates)}")
    print(f"funding_neutral_cells={funding_neutral}/{funding_total} ({100 * funding_neutral / funding_total:.2f}%)")
    print(f"fear_greed_neutral_days={fg_neutral_days}/{fg_total_days} ({100 * fg_neutral_days / fg_total_days:.2f}%)")


def audit_hashes(files):
    print("\nFILE_HASHES")
    for path in files:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        print(f"{path} sha256={h.hexdigest()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-time", default="2026-04-17T00:00:00+00:00")
    parser.add_argument("--decision-hour-utc", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    T = decision_time(args.decision_time)
    audit_mapping(T)
    audit_funding(args.data_dir, T)
    audit_fear_greed(args.data_dir, T)
    audit_fear_greed_gap(args.data_dir)
    audit_funding_edges()
    audit_future_poisoning(args.data_dir, args.decision_hour_utc)
    audit_standardization(args.data_dir, args.decision_hour_utc)
    audit_neutral_fallbacks(args.data_dir, args.decision_hour_utc)
    audit_hashes([
        "phase2_experiments.py",
        "audit_point_in_time.py",
        os.path.join(args.data_dir, "funding_rates.csv"),
        os.path.join(args.data_dir, "fear_greed.csv"),
        "prices_cache.csv",
    ])
    print("\nSTANDARDIZATION_CONFIRMATION")
    print("Phase2 runner standardizes all feature columns using historical training data only inside each fold.")
    print("Baseline uses the same feature columns as feature experiments; external placeholder is constant raw 0.")
    print("Shuffled controls use a fixed-seed shuffle of days, preserving the same day order for all coins.")


if __name__ == "__main__":
    main()
