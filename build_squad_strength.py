#!/usr/bin/env python3
"""Rate each team's playing XI from what its players did in EARLIER matches.

For every men's ODI this writes, per team: the XI's batting rate (runs per ball),
bowling rate (runs conceded per ball) and wicket rate (wickets per ball). Each
player's own history is blended with the all-matches average (PRIOR_BALLS balls of
average cricket), so a debutant starts at average and a veteran counts for his record.

Nothing from the match itself is used: the numbers are read first, then the match
is added to the players' histories. Matches are processed in date order.

Usage:
    python build_squad_strength.py --raw path/to/odis_json --out data
"""
import argparse
import csv
import os
from collections import defaultdict

from build_player_stats import read_matches

PRIOR_BALLS = 300          # about 50 overs of average cricket per player
DEFAULT = {"bat": 0.80, "bowl": 0.80, "wkt": 0.025}   # used before any history exists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    matches = sorted(read_matches(args.raw), key=lambda m: (m[1], m[0]))
    P = defaultdict(lambda: {"runs": 0, "balls": 0, "bb": 0, "br": 0, "w": 0})
    G = {"runs": 0, "balls": 0, "bb": 0, "br": 0, "w": 0}
    rows = []
    for mid, date, squads, names, inns in matches:
        mu = G["runs"] / G["balls"] if G["balls"] else DEFAULT["bat"]
        nu = G["br"] / G["bb"] if G["bb"] else DEFAULT["bowl"]
        om = G["w"] / G["bb"] if G["bb"] else DEFAULT["wkt"]
        for team, xi in squads.items():
            if not xi:
                continue
            n = len(xi)
            rb = sum(P[p]["runs"] for p in xi) + n * PRIOR_BALLS * mu
            bb = sum(P[p]["balls"] for p in xi) + n * PRIOR_BALLS
            rows.append({
                "match_id": mid, "team": team, "players": n,
                "bat_rate": rb / bb,
                "bowl_rate": (sum(P[p]["br"] for p in xi) + n * PRIOR_BALLS * nu)
                             / (sum(P[p]["bb"] for p in xi) + n * PRIOR_BALLS),
                "wkt_rate": (sum(P[p]["w"] for p in xi) + n * PRIOR_BALLS * om)
                            / (sum(P[p]["bb"] for p in xi) + n * PRIOR_BALLS),
            })
        # only now add this match to the histories
        for order, bat, bowl in inns:
            for p, s in bat.items():
                P[p]["runs"] += s["runs"]; P[p]["balls"] += s["balls"]
                G["runs"] += s["runs"]; G["balls"] += s["balls"]
            for p, s in bowl.items():
                P[p]["bb"] += s["balls"]; P[p]["br"] += s["runs"]; P[p]["w"] += s["wkts"]
                G["bb"] += s["balls"]; G["br"] += s["runs"]; G["w"] += s["wkts"]

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "squad_strength.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} team-matches written to {path}")


if __name__ == "__main__":
    main()
