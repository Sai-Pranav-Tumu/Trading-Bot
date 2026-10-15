"""
Run the frozen Phase 2 development-fold suite once.

Final folds are not used. This script runs:
  baseline, funding, fear_greed,
  funding_shuffled with 5 shuffle seeds,
  fear_greed_shuffled with 5 shuffle seeds.
"""
import argparse
import os
import subprocess

import pandas as pd


PYTHON = r"C:\Users\saipr\AppData\Local\Programs\Python\Python314\python.exe"
SHUFFLE_SEEDS = [101, 202, 303, 404, 505]


def run_one(experiment, shuffle_seed=None):
    cmd = [
        PYTHON, "phase2_experiments.py",
        "--experiment", experiment,
        "--folds", "9",
        "--seeds", "3",
        "--gens", "60",
        "--windows", "8",
    ]
    if shuffle_seed is not None:
        cmd += ["--shuffle-seed", str(shuffle_seed)]
    print("\nRUNNING", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)
    suffix = f"_shuffle{shuffle_seed}" if experiment.endswith("_shuffled") else ""
    return os.path.join("results", f"phase2_{experiment}{suffix}_dev.csv")


def summarize(path, label):
    df = pd.read_csv(path)
    successes = int(df["ensemble_success"].sum())
    return {
        "label": label,
        "path": path,
        "successes": successes,
        "mean_net": float(df["ensemble_net_return"].mean()),
        "median_net": float(df["ensemble_net_return"].median()),
        "min_net": float(df["ensemble_net_return"].min()),
        "max_net": float(df["ensemble_net_return"].max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rows = []
    for exp in ("baseline", "funding", "fear_greed"):
        path = run_one(exp)
        rows.append(summarize(path, exp))
    for exp in ("funding_shuffled", "fear_greed_shuffled"):
        control_rows = []
        for seed in SHUFFLE_SEEDS:
            path = run_one(exp, seed)
            row = summarize(path, f"{exp}_{seed}")
            control_rows.append(row)
            rows.append(row)
        success_values = [r["successes"] for r in control_rows]
        mean_values = [r["mean_net"] for r in control_rows]
        print(f"\nCONTROL_RANGE {exp}")
        print(f"success_folds_range={min(success_values)}..{max(success_values)}")
        print(f"mean_net_return_range_pct={100 * min(mean_values):.2f}..{100 * max(mean_values):.2f}")
    summary = pd.DataFrame(rows)
    summary_path = os.path.join("results", "phase2_suite_summary_dev.csv")
    summary.to_csv(summary_path, index=False)
    show = summary.copy()
    for col in ("mean_net", "median_net", "min_net", "max_net"):
        show[col] = (show[col] * 100).round(2)
    print("\nPHASE2_SUITE_SUMMARY")
    print(show.to_string(index=False))
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
