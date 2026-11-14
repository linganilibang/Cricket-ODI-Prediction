#!/usr/bin/env python3
"""Fit the small model behind the dashboard's "Predict a match" page and save it as JSON.

The page can only know things you can pick in a menu, so this model uses five clues:
team strength (Elo), home advantage, matches played, who won the toss
and who bats first. Recent form is left out because its weight is tiny and slightly negative (noise).
Squad experience and ground record are left out because they need a named XI and
a named ground.

Fit on matches before SPLIT_DATE and tested on the matches since, the same way as
train_evaluate.py. Current team numbers come from every match in the data.

Usage:
    python export_picker.py --raw path/to/odis_json --data data --out results
"""
import argparse
import json
import math
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import build_dataset as bd
from train_evaluate import SPLIT_DATE, fit_logistic, pick_k

BASE = ["elo_diff", "home_a", "played_diff"]
TOSS = ["toss_a", "batfirst_a"]
ACTIVE_FROM = "2023-01-01"   # only list teams that have played since this date
MIN_PLAYED = 20              # and that have a real track record


def current_team_state(raw_dir, k):
    elo = defaultdict(lambda: bd.START_ELO)
    form = defaultdict(lambda: deque(maxlen=bd.FORM_WINDOW))
    played, last = defaultdict(int), {}
    for m in bd.load_matches(raw_dir):
        a, b = m["teams"]
        res = 1 if m["winner"] == a else 0
        ea = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
        elo[a] += k * (res - ea)
        elo[b] -= k * (res - ea)
        for t, won in ((a, res), (b, 1 - res)):
            form[t].append(won)
            played[t] += 1
            last[t] = m["date"]
    teams = {}
    for t in played:
        if last[t] >= ACTIVE_FROM and played[t] >= MIN_PLAYED:
            teams[t] = {"elo": round(elo[t], 1), "played": played[t], "last": last[t]}
    return teams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data, "odi_matches.csv"))
    best_k, _ = pick_k(df)
    k = int(best_k.replace("elo_diff_k", ""))
    df["elo_diff"] = df[best_k]
    train, test = df[df.date < SPLIT_DATE], df[df.date >= SPLIT_DATE]

    out = {"k": k, "tests": {}}
    for label, cols in (("without toss", BASE), ("with toss", BASE + TOSS)):
        p, _ = fit_logistic(train[cols], train.a_won.values, test[cols])
        acc = float(((p >= 0.5).astype(int) == test.a_won.values).mean())
        out["tests"][label] = {"accuracy": acc, "n": len(test)}
        print(f"{label}: {acc:.3f} on {len(test)} test matches")

    cols = BASE + TOSS
    scale = train[cols].std(axis=0).replace(0, 1)
    _, coefs = fit_logistic(train[cols], train.a_won.values, test[cols])
    # weights on the raw (unscaled) features, so the page just adds them up
    out["weights"] = {c: float(coefs[c] / scale[c]) for c in cols}
    out["teams"] = current_team_state(args.raw, k)
    with open(os.path.join(args.out, "picker_model.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("weights:", {c: round(v, 4) for c, v in out["weights"].items()})
    print(len(out["teams"]), "active teams")


if __name__ == "__main__":
    main()
