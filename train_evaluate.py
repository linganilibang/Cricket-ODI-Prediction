#!/usr/bin/env python3
"""Train ODI win predictors and test them honestly on later matches only.

Train on matches before SPLIT_DATE, test on matches from SPLIT_DATE onward.
The Elo speed (K) is chosen using pre-split matches only, never the test set.

Usage:
    python train_evaluate.py --data data --out results
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

SPLIT_DATE = "2023-01-01"
FULL_MEMBERS = {
    "Afghanistan", "Australia", "Bangladesh", "England", "India", "Ireland",
    "New Zealand", "Pakistan", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe",
}
SEED = 7


PLAYER_FEATURES = ["bat_diff", "bowl_diff", "wkt_diff"]


def add_squad_strength(df, data_dir):
    """Add how strong each side's XI looked from its players' EARLIER matches.

    Reads data/squad_strength.csv (made by build_squad_strength.py). Differences are
    A minus B, except bowling runs per ball, where lower is better, so it is B minus A.
    Matches with no squad data get 0 (no difference).
    """
    path = os.path.join(data_dir, "squad_strength.csv")
    if not os.path.exists(path):
        return df
    sq = pd.read_csv(path).drop_duplicates(["match_id", "team"]).set_index(["match_id", "team"])
    def get(team_col, col):
        idx = pd.MultiIndex.from_arrays([df.match_id, df[team_col]])
        return sq[col].reindex(idx).values
    for new, col, sign in (("bat_diff", "bat_rate", 1), ("bowl_diff", "bowl_rate", -1), ("wkt_diff", "wkt_rate", 1)):
        d = sign * (get("team_a", col) - get("team_b", col))
        df[new] = np.nan_to_num(d, nan=0.0)
    return df


def pick_k(df):
    """Choose the Elo K that gives the best log loss on 2012 up to the split date."""
    sel = df[(df.date >= "2012-01-01") & (df.date < SPLIT_DATE)]
    scores = {}
    for col in [c for c in df.columns if c.startswith("elo_diff_k")]:
        p = 1 / (1 + 10 ** (-sel[col] / 400))
        scores[col] = log_loss(sel.a_won, p)
    best = min(scores, key=scores.get)
    return best, scores


def fit_logistic(Xtr, ytr, Xte):
    # Features are A-minus-B differences, so we scale without centring and use
    # no intercept: swapping the two teams then flips the prediction exactly.
    scale = Xtr.std(axis=0).replace(0, 1)
    model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
    model.fit(Xtr / scale, ytr)
    coefs = dict(zip(Xtr.columns, model.coef_[0]))
    return model.predict_proba(Xte / scale)[:, 1], coefs


def fit_boosting(Xtr, ytr, Xte):
    # Add a mirrored copy of every training match (team A <-> team B) and
    # average both directions when predicting, so the model is symmetric too.
    X = pd.concat([Xtr, -Xtr])
    y = np.concatenate([ytr, 1 - ytr])
    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150,
        l2_regularization=2.0, min_samples_leaf=40, random_state=SEED)
    model.fit(X, y)
    p = (model.predict_proba(Xte)[:, 1] + 1 - model.predict_proba(-Xte)[:, 1]) / 2
    return p


def paired_logloss_gain(y, p_base, p_new, rng, n_boot=2000):
    """How much better (lower) is p_new's log loss than p_base's, on the same matches?

    Positive means the new model is better. Re-sampling whole matches keeps the two
    models paired, so the range shows whether the gain could be luck.
    """
    y = np.asarray(y)
    p_base = np.clip(np.asarray(p_base), 1e-6, 1 - 1e-6)
    p_new = np.clip(np.asarray(p_new), 1e-6, 1 - 1e-6)
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    per_match = ll(p_base) - ll(p_new)
    boots = [per_match[rng.integers(0, len(y), len(y))].mean() for _ in range(n_boot)]
    return per_match.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def summarise(name, subset, y, p=None, pred=None, rng=None):
    y = np.asarray(y)
    if pred is None:
        pred = (np.asarray(p) >= 0.5).astype(int)
    pred = np.asarray(pred)
    accs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        accs.append((pred[idx] == y[idx]).mean())
    row = {
        "model": name, "subset": subset, "n": len(y),
        "accuracy": (pred == y).mean(),
        "acc_ci_low": np.percentile(accs, 2.5),
        "acc_ci_high": np.percentile(accs, 97.5),
        "log_loss": np.nan, "brier": np.nan,
    }
    if p is not None:
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        row["log_loss"] = log_loss(y, p, labels=[0, 1])
        row["brier"] = brier_score_loss(y, p)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(os.path.join(args.data, "odi_matches.csv"))
    df = add_squad_strength(df, args.data)
    have_players = all(c in df.columns for c in PLAYER_FEATURES)
    best_k, k_scores = pick_k(df)
    print("Elo K selection (log loss, 2012 to split, lower is better):")
    for c, s in k_scores.items():
        print(f"  {c}: {s:.4f}{'  <- chosen' if c == best_k else ''}")

    df["elo_diff"] = df[best_k]
    elo_only = ["elo_diff"]
    no_toss = ["elo_diff", "form_diff", "venue_exp_diff", "venue_wr_diff",
               "squad_exp_diff", "played_diff"]
    with_home = no_toss + ["home_a"]
    full = with_home + ["toss_a", "batfirst_a"]

    train = df[df.date < SPLIT_DATE]
    test = df[df.date >= SPLIT_DATE].copy()
    ytr, yte = train.a_won.values, test.a_won.values
    print(f"\nTrain: {len(train)} matches before {SPLIT_DATE}. Test: {len(test)} matches since.")

    preds, coefs = {}, {}
    for label, cols in (("Logistic: Elo only", elo_only),
                        ("Logistic: Elo + home advantage", ["elo_diff", "home_a"]),
                        ("Logistic: Elo + form/venue/squad (no toss)", no_toss),
                        ("Logistic: all features (incl. toss)", full)):
        preds[label], coefs[label] = fit_logistic(train[cols], ytr, test[cols])
    preds["Boosting: all features (incl. toss)"] = fit_boosting(train[full], ytr, test[full])
    if have_players:
        label = "Logistic: all features + player strength"
        preds[label], coefs[label] = fit_logistic(train[full + PLAYER_FEATURES], ytr, test[full + PLAYER_FEATURES])
        label = "Logistic: Elo + home + player strength"
        cols = ["elo_diff", "home_a"] + PLAYER_FEATURES
        preds[label], coefs[label] = fit_logistic(train[cols], ytr, test[cols])

    both_full = (test.team_a.isin(FULL_MEMBERS) & test.team_b.isin(FULL_MEMBERS)).values
    subsets = {"all test matches": np.ones(len(test), bool),
               "full member vs full member": both_full}

    rows = []
    for sname, mask in subsets.items():
        y = yte[mask]
        t = test[mask]
        coin = summarise("Coin flip (expected)", sname, y, p=np.full(len(y), 0.5), pred=np.zeros(len(y), int), rng=rng)
        coin.update(accuracy=0.5, acc_ci_low=np.nan, acc_ci_high=np.nan)
        rows.append(coin)
        rows.append(summarise("Rule: higher Elo wins", sname, y, p=1 / (1 + 10 ** (-t.elo_diff / 400)), pred=(t.elo_diff > 0).astype(int), rng=rng))
        rows.append(summarise("Rule: team batting second wins", sname, y, pred=(t.batfirst_a == -1).astype(int), rng=rng))
        rows.append(summarise("Rule: better recent form wins", sname, y, pred=(t.form_diff > 0).astype(int), rng=rng))
        for label, p in preds.items():
            rows.append(summarise(label, sname, y, p=p[mask], rng=rng))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(args.out, "metrics.csv"), index=False)

    pd.set_option("display.width", 200)
    for sname in subsets:
        print(f"\n=== {sname} ===")
        show = metrics[metrics.subset == sname].drop(columns="subset")
        print(show.round(3).to_string(index=False))

    # Is each extra set of clues better than Elo alone, or is the gap just luck?
    gains = []
    for label in [k for k in preds if k.startswith("Logistic") and k != "Logistic: Elo only"] + ["Boosting: all features (incl. toss)"]:
        g, lo, hi = paired_logloss_gain(yte, preds["Logistic: Elo only"], preds[label], rng)
        gains.append({"model": label, "vs": "Logistic: Elo only", "logloss_gain": g, "ci_low": lo, "ci_high": hi})
    gains = pd.DataFrame(gains)
    gains.to_csv(os.path.join(args.out, "logloss_gain.csv"), index=False)
    print("\nLog-loss gain over Elo only (positive = better; range is a 95% bootstrap):")
    print(gains.round(4).to_string(index=False))

    if have_players:
        print("\nStandardised coefficients, all features + player strength:")
        print(pd.Series(coefs["Logistic: all features + player strength"]).round(3).to_string())

    # what the logistic model leans on (standardised coefficients)
    c = pd.Series(coefs["Logistic: all features (incl. toss)"]).sort_values(key=abs, ascending=False)
    c.to_csv(os.path.join(args.out, "coefficients.csv"), header=["standardised_coefficient"])
    print("\nStandardised coefficients, all-features logistic:")
    print(c.round(3).to_string())

    # calibration: when the model is X% sure, how often is it right?
    best = "Logistic: all features (incl. toss)"
    p = preds[best]
    conf = np.maximum(p, 1 - p)
    right = ((p >= 0.5).astype(int) == yte)
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]
    cal = pd.DataFrame({"conf": conf, "right": right})
    cal["bin"] = pd.cut(cal.conf, bins, include_lowest=True)
    cal = cal.groupby("bin", observed=True).agg(matches=("right", "size"),
                                                avg_confidence=("conf", "mean"),
                                                actual_accuracy=("right", "mean")).reset_index()
    cal["bin"] = cal["bin"].astype(str)
    cal.to_csv(os.path.join(args.out, "calibration.csv"), index=False)
    print("\nCalibration (all-features logistic, test set):")
    print(cal.round(3).to_string(index=False))

    out = test[["date", "team_a", "team_b", "venue", "a_won"]].copy()
    out["p_team_a_wins"] = preds[best].round(3)
    out.to_csv(os.path.join(args.out, "test_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
