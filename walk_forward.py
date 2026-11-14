#!/usr/bin/env python3
"""Walk-forward test: for each year, train on everything before it and test on that year.

The single 2023 split gives one 472-match test. This repeats the test for every year
from FIRST_TEST_YEAR, so you can see whether the result holds across years or came
from one lucky stretch. The Elo speed K is picked again for every year, using only
matches before that year.

Usage:
    python walk_forward.py --data data --out results
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from train_evaluate import PLAYER_FEATURES, SEED, add_squad_strength, fit_logistic, paired_logloss_gain

FIRST_TEST_YEAR = 2016
K_WINDOW_FROM = "2005-01-01"   # ignore the first years of Elo, which are still settling

MODELS = {
    "Higher Elo wins": None,
    "Logistic: Elo only": ["elo_diff"],
    "Logistic: Elo + home advantage": ["elo_diff", "home_a"],
    "Logistic: all features": ["elo_diff", "home_a", "form_diff", "venue_exp_diff",
                               "venue_wr_diff", "squad_exp_diff", "played_diff",
                               "toss_a", "batfirst_a"],
    "Logistic: all features + player strength": ["elo_diff", "home_a", "form_diff", "venue_exp_diff",
                                                 "venue_wr_diff", "squad_exp_diff", "played_diff",
                                                 "toss_a", "batfirst_a"] + PLAYER_FEATURES,
    "Logistic: Elo + home + player strength": ["elo_diff", "home_a"] + PLAYER_FEATURES,
}


def pick_k_before(df, cutoff):
    sel = df[(df.date >= K_WINDOW_FROM) & (df.date < cutoff)]
    best, best_ll = None, 9
    for col in [c for c in df.columns if c.startswith("elo_diff_k")]:
        ll = log_loss(sel.a_won, 1 / (1 + 10 ** (-sel[col] / 400)))
        if ll < best_ll:
            best, best_ll = col, ll
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(os.path.join(args.data, "odi_matches.csv"))
    df = add_squad_strength(df, args.data)
    df["year"] = df.date.str[:4].astype(int)
    if not all(c in df.columns for c in PLAYER_FEATURES):
        for k in [k for k in MODELS if "player strength" in k]:
            del MODELS[k]

    rows, pooled = [], {m: [] for m in MODELS}
    pooled_y = []
    for year in range(FIRST_TEST_YEAR, df.year.max() + 1):
        cutoff = f"{year}-01-01"
        train, test = df[df.date < cutoff].copy(), df[df.year == year].copy()
        if len(test) < 20:
            continue
        k = pick_k_before(df, cutoff)
        train["elo_diff"], test["elo_diff"] = train[k], test[k]
        y = test.a_won.values
        row = {"year": year, "n": len(test), "elo_k": int(k.replace("elo_diff_k", ""))}
        for name, cols in MODELS.items():
            if cols is None:
                p = 1 / (1 + 10 ** (-test.elo_diff / 400))
            else:
                p, _ = fit_logistic(train[cols], train.a_won.values, test[cols])
            p = np.asarray(p)
            row[name] = float(((p >= 0.5).astype(int) == y).mean())
            pooled[name].append(p)
        pooled_y.append(y)
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    yearly = pd.DataFrame(rows)
    yearly.to_csv(os.path.join(args.out, "walk_forward_by_year.csv"), index=False)

    y = np.concatenate(pooled_y)
    summary = []
    for name in MODELS:
        p = np.concatenate(pooled[name])
        acc = ((p >= 0.5).astype(int) == y)
        boots = [acc[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)]
        summary.append({"model": name, "n": len(y), "accuracy": acc.mean(),
                        "acc_ci_low": np.percentile(boots, 2.5), "acc_ci_high": np.percentile(boots, 97.5),
                        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))})
    base = np.concatenate(pooled["Logistic: Elo only"])
    for r in summary:
        if r["model"] in ("Logistic: Elo only", "Higher Elo wins"):
            r.update(gain=np.nan, gain_low=np.nan, gain_high=np.nan)
            continue
        g, lo, hi = paired_logloss_gain(y, base, np.concatenate(pooled[r["model"]]), rng)
        r.update(gain=g, gain_low=lo, gain_high=hi)
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out, "walk_forward_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\nAll test years pooled ({yearly.year.min()} to {yearly.year.max()}, {len(y)} matches):")
    print(summary.round(3).to_string(index=False))
    better = (yearly["Logistic: all features"] > yearly["Higher Elo wins"]).sum()
    print(f"\nAll-features beat 'higher Elo wins' in {better} of {len(yearly)} years.")


if __name__ == "__main__":
    main()
