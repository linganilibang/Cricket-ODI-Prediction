#!/usr/bin/env python3
"""Turn Cricsheet ODI JSON files into one row per men's ODI.

Every feature is calculated from matches played BEFORE that match, so nothing
from the future leaks into the data the model learns from.

Usage:
    python build_dataset.py --raw path/to/odis_json --out data
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict, deque

import pandas as pd

from venues import canonical_venue, host_team

K_VALUES = [10, 20, 30, 40]   # Elo update speeds; train_evaluate.py picks one
START_ELO = 1500.0
FORM_WINDOW = 10               # last N matches count as "recent form"


def load_matches(raw_dir):
    """Read men's ODIs that had a winner (ties and no-results are skipped)."""
    matches = []
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        match_id = int(os.path.splitext(os.path.basename(path))[0])
        with open(path) as f:
            info = json.load(f)["info"]
        if info.get("gender") != "male":
            continue
        winner = info.get("outcome", {}).get("winner")
        if winner is None:
            continue
        teams = info["teams"]
        if len(teams) != 2:
            continue
        people = info.get("registry", {}).get("people", {})
        squads = {
            t: [people.get(p, p) for p in info.get("players", {}).get(t, [])]
            for t in teams
        }
        toss = info.get("toss", {})
        matches.append({
            "match_id": match_id,
            "date": info["dates"][0],
            "teams": teams,
            "winner": winner,
            "venue": info.get("venue", "unknown"),
            "city": info.get("city", ""),
            "overs": info.get("overs", 50),
            "toss_winner": toss.get("winner"),
            "toss_decision": toss.get("decision"),
            "squads": squads,
        })
    matches.sort(key=lambda m: (m["date"], m["match_id"]))
    return matches


def build(matches):
    elo = {k: defaultdict(lambda: START_ELO) for k in K_VALUES}
    form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    played = defaultdict(int)
    last_date = {}
    venue_played = defaultdict(int)
    venue_wins = defaultdict(int)
    appearances = defaultdict(int)
    rows = []

    for m in matches:
        t = m["teams"]
        # Team A / B order is decided by match id parity, not by listing order,
        # so the model cannot learn anything from who is written first.
        a, b = (t[0], t[1]) if m["match_id"] % 2 == 0 else (t[1], t[0])
        venue = canonical_venue(m["venue"])   # one name per ground
        host = host_team(m["venue"], m["city"])
        home_a = 1 if host == a else (-1 if host == b else 0)   # 0 = neutral ground

        def feats(team):
            hist = form[team]
            form_val = (sum(hist) + 2.5) / (len(hist) + 5)
            vp = venue_played[(team, venue)]
            vwr = (venue_wins[(team, venue)] + 2) / (vp + 4)
            squad = m["squads"][team]
            exp = sum(appearances[p] for p in squad) / len(squad) if squad else 0.0
            return form_val, math.log1p(vp), vwr, math.log1p(exp), math.log1p(played[team])

        fa, fb = feats(a), feats(b)

        tw = m["toss_winner"]
        toss_a = 0 if tw is None else (1 if tw == a else -1)
        if tw is None or m["toss_decision"] is None:
            batfirst_a = 0
        else:
            bat_first = tw if m["toss_decision"] == "bat" else (b if tw == a else a)
            batfirst_a = 1 if bat_first == a else -1

        row = {
            "match_id": m["match_id"],
            "date": m["date"],
            "team_a": a,
            "team_b": b,
            "venue": m["venue"],
            "ground": venue,
            "city": m["city"],
            "overs": m["overs"],
            "a_won": int(m["winner"] == a),
            "form_diff": fa[0] - fb[0],
            "venue_exp_diff": fa[1] - fb[1],
            "venue_wr_diff": fa[2] - fb[2],
            "squad_exp_diff": fa[3] - fb[3],
            "played_diff": fa[4] - fb[4],
            "toss_a": toss_a,
            "batfirst_a": batfirst_a,
            "home_a": home_a,
        }
        for k in K_VALUES:
            row[f"elo_diff_k{k}"] = elo[k][a] - elo[k][b]
        rows.append(row)

        # ---- update state AFTER the row is recorded ----
        res_a = row["a_won"]
        for k in K_VALUES:
            ea = 1 / (1 + 10 ** ((elo[k][b] - elo[k][a]) / 400))
            elo[k][a] += k * (res_a - ea)
            elo[k][b] -= k * (res_a - ea)
        for team, won in ((a, res_a), (b, 1 - res_a)):
            form[team].append(won)
            played[team] += 1
            last_date[team] = m["date"]
            venue_played[(team, venue)] += 1
            venue_wins[(team, venue)] += won
            for p in m["squads"][team]:
                appearances[p] += 1

    ratings = pd.DataFrame([
        {"team": team, "matches": played[team], "last_match": last_date[team],
         **{f"elo_k{k}": round(elo[k][team], 1) for k in K_VALUES}}
        for team in played
    ]).sort_values("elo_k20", ascending=False)
    return pd.DataFrame(rows), ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder containing the Cricsheet *.json files")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    matches = load_matches(args.raw)
    df, ratings = build(matches)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "odi_matches.csv"), index=False)
    ratings.to_csv(os.path.join(args.out, "final_ratings.csv"), index=False)
    print(f"{len(df)} men's ODI matches with a winner, {df.date.min()} to {df.date.max()}")
    print(ratings.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
