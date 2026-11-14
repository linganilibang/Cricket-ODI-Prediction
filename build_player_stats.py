#!/usr/bin/env python3
"""Work out player stats from Cricsheet ball-by-ball ODI files (men's matches only).

Writes two files: one for matches since RECENT_FROM, one for every match in the data.

Usage:
    python build_player_stats.py --raw path/to/odis_json --out data
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

RECENT_FROM = "2023-01-01"
# Credited to the bowler. Run outs and retirements are not.
BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}
# Not counted as a dismissal for a batting average.
NOT_OUT_KINDS = {"retired hurt", "retired not out"}
# A player needs this much cricket in the period to appear.
PROFILES = {
    "recent": {"min_inn": 10, "min_balls": 120},   # 10 innings, 20 overs
    "all": {"min_inn": 30, "min_balls": 600},      # 30 innings, 100 overs
}
BOWL_BALLS_PER_MATCH = 18   # a bowler averages 3+ overs in every match they play (not a part-timer)
POWER_HITTER_PERCENTILE = 0.75  # top-order batters scoring faster than 3 in 4 of their peers
TOP_ORDER_MAX_POS = 3.5     # opens or bats three: candidates for power hitters
ANCHOR_MAX_POS = 4.5        # bats in the top four
MIDDLE_MAX_POS = 7.5        # bats in the top seven
ALLROUNDER_MAX_POS = 7.5


def new_player():
    return {"teams": Counter(), "matches": 0, "bat_inn": 0, "runs": 0, "balls": 0, "outs": 0,
            "fours": 0, "sixes": 0, "bound_runs": 0, "pos_sum": 0,
            "bowl_balls": 0, "bowl_runs": 0, "wkts": 0, "dots": 0}


def read_matches(raw_dir):
    """Yield (match_id, date, squads, names, innings_stats) for each men's match."""
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        with open(path) as f:
            j = json.load(f)
        info = j["info"]
        if info.get("gender") != "male":
            continue
        ids = info.get("registry", {}).get("people", {})
        pid = lambda name: ids.get(name, name)
        names = {}
        squads = {}
        for team, plist in info.get("players", {}).items():
            squads[team] = [pid(p) for p in plist]
            for p in plist:
                names[pid(p)] = p
        inns_out = []
        for inn in j.get("innings", []):
            if inn.get("super_over"):
                continue
            order = []
            bat = defaultdict(lambda: {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "bound": 0, "out": False})
            bowl = defaultdict(lambda: {"balls": 0, "runs": 0, "wkts": 0, "dots": 0})
            for over in inn.get("overs", []):
                for d in over["deliveries"]:
                    b, n, w = pid(d["batter"]), pid(d["non_striker"]), pid(d["bowler"])
                    for p in (b, n):
                        if p not in order:
                            order.append(p)
                    runs = d["runs"]
                    ex = d.get("extras", {})
                    wide, noball = "wides" in ex, "noballs" in ex
                    br = runs["batter"]
                    if not wide:
                        s = bat[b]
                        s["balls"] += 1
                        s["runs"] += br
                        if br in (4, 6) and not runs.get("non_boundary"):
                            s["bound"] += br
                            s["fours" if br == 4 else "sixes"] += 1
                    bw = bowl[w]
                    bw["runs"] += br + ex.get("wides", 0) + ex.get("noballs", 0)
                    if not wide and not noball:
                        bw["balls"] += 1
                        if br == 0:
                            bw["dots"] += 1
                    for wk in d.get("wickets", []):
                        kind = wk["kind"]
                        if kind not in NOT_OUT_KINDS:
                            bat[pid(wk["player_out"])]["out"] = True
                        if kind in BOWLER_WICKETS:
                            bw["wkts"] += 1
            inns_out.append((order, dict(bat), dict(bowl)))
        yield int(os.path.splitext(os.path.basename(path))[0]), info["dates"][0], squads, names, inns_out


def aggregate(raw_dir):
    out = {"recent": defaultdict(new_player), "all": defaultdict(new_player)}
    names_all = {}
    for _mid, date, squads, names, inns in read_matches(raw_dir):
        names_all.update(names)
        keys = ["all"] + (["recent"] if date >= RECENT_FROM else [])
        for key in keys:
            P = out[key]
            for team, plist in squads.items():
                for p in plist:
                    P[p]["matches"] += 1
                    P[p]["teams"][team] += 1
            for order, bat, bowl in inns:
                for pos, p in enumerate(order, start=1):
                    s = bat.get(p)
                    if s is None or (s["balls"] == 0 and not s["out"]):
                        continue
                    r = P[p]
                    r["bat_inn"] += 1
                    r["runs"] += s["runs"]; r["balls"] += s["balls"]; r["outs"] += int(s["out"])
                    r["fours"] += s["fours"]; r["sixes"] += s["sixes"]; r["bound_runs"] += s["bound"]
                    r["pos_sum"] += pos
                for p, s in bowl.items():
                    r = P[p]
                    r["bowl_balls"] += s["balls"]; r["bowl_runs"] += s["runs"]
                    r["wkts"] += s["wkts"]; r["dots"] += s["dots"]
    return out, names_all


def finish(players, names, cfg):
    """Add averages and a role to every qualifying player."""
    rows = []
    for p, r in players.items():
        bat_q = r["bat_inn"] >= cfg["min_inn"]
        bowl_q = (r["bowl_balls"] >= cfg["min_balls"]
                  and r["bowl_balls"] >= BOWL_BALLS_PER_MATCH * r["matches"])
        if not (bat_q or bowl_q):
            continue
        row = {"id": p, "name": re.sub(r"\s\(\d+\)$", "", names.get(p, p)), "team": r["teams"].most_common(1)[0][0], "matches": r["matches"],
               "bat_inn": r["bat_inn"], "runs": r["runs"], "balls": r["balls"], "outs": r["outs"],
               "fours": r["fours"], "sixes": r["sixes"], "bound_runs": r["bound_runs"],
               "bowl_balls": r["bowl_balls"], "bowl_runs": r["bowl_runs"], "wkts": r["wkts"], "dots": r["dots"]}
        row["avg_pos"] = round(r["pos_sum"] / r["bat_inn"], 2) if r["bat_inn"] else ""
        row["avg"] = round(r["runs"] / r["outs"], 2) if r["outs"] else ""
        row["sr"] = round(100 * r["runs"] / r["balls"], 2) if r["balls"] else ""
        row["boundary_pct"] = round(100 * r["bound_runs"] / r["runs"], 2) if r["runs"] else ""
        row["econ"] = round(6 * r["bowl_runs"] / r["bowl_balls"], 2) if r["bowl_balls"] else ""
        row["bowl_avg"] = round(r["bowl_runs"] / r["wkts"], 2) if r["wkts"] else ""
        row["bowl_sr"] = round(r["bowl_balls"] / r["wkts"], 1) if r["wkts"] else ""
        row["dot_pct"] = round(100 * r["dots"] / r["bowl_balls"], 2) if r["bowl_balls"] else ""
        row["_bat_q"], row["_bowl_q"] = bat_q, bowl_q
        rows.append(row)

    # Roles, by rule. All-rounder first, then bowlers, then batters by where they bat.
    for row in rows:
        pos = row["avg_pos"] if row["avg_pos"] != "" else 99
        if row["_bat_q"] and row["_bowl_q"] and pos <= ALLROUNDER_MAX_POS:
            row["role"] = "allrounder"
        elif row["_bowl_q"]:
            row["role"] = "bowler"
        elif row["_bat_q"] and pos <= MIDDLE_MAX_POS:
            row["role"] = "batter"
        else:
            row["role"] = ""
    top = [r["sr"] for r in rows if r["role"] == "batter" and r["avg_pos"] <= TOP_ORDER_MAX_POS]
    cutoff = round(statistics.quantiles(top, n=100, method="inclusive")[int(POWER_HITTER_PERCENTILE * 100) - 1], 1)
    for row in rows:
        if row["role"] == "batter":
            pos = row["avg_pos"]
            if pos <= TOP_ORDER_MAX_POS and row["sr"] >= cutoff:
                row["role"] = "power"
            elif pos <= ANCHOR_MAX_POS:
                row["role"] = "anchor"
            else:
                row["role"] = "finisher"
    rows = [r for r in rows if r["role"]]
    for r in rows:
        r.pop("_bat_q"); r.pop("_bowl_q")
    rows.sort(key=lambda r: -r["runs"])
    return rows, cutoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder containing the Cricsheet *.json files")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    agg, names = aggregate(args.raw)
    meta = {"recent_from": RECENT_FROM}
    for key, cfg in PROFILES.items():
        rows, cutoff = finish(agg[key], names, cfg)
        path = os.path.join(args.out, f"player_stats_{key}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        meta[key] = {**cfg, "power_hitter_sr_cutoff": cutoff, "players": len(rows),
                     "roles": dict(Counter(r["role"] for r in rows))}
        print(key, meta[key])
    with open(os.path.join(args.out, "player_roles_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
