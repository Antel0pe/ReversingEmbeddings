"""Summarise runs/simple/*.json: passes per sheet group, at 4,225 dots and at the dots each sheet needs.

Run: .venv/bin/python -m unfolding_suite.experiments.simple_table main [more run names]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from unfolding_suite.sheets import SHEETS

RUNS = Path(__file__).resolve().parent.parent / "runs" / "simple"
GROUPS = ("design", "holdout", "audit")


def table(names):
    rows = []
    for n in names:
        rows += json.loads((RUNS / f"{n}.json").read_text())
    by = defaultdict(lambda: defaultdict(list))         # (method, noise) -> (sheet, grid) -> [passed]
    secs = defaultdict(list)
    for r in rows:
        by[(r["method"], r["noise"])][(r["sheet"], r["grid"])].append(bool(r["passed"]))
        if r["grid"] == 65:
            secs[(r["method"], r["noise"])].append(r["seconds"] or 0)
    out = {}
    for key, res in by.items():
        at65 = {s: all(v) for (s, g), v in res.items() if g == 65}
        best_grid = {}
        for (s, g), v in res.items():
            if s not in best_grid or g > best_grid[s][0]:
                best_grid[s] = (g, all(v))
        enough = {s: p for s, (g, p) in best_grid.items()}
        summary = {}
        for label, res_ in (("4225", at65), ("enough", enough)):
            summary[label] = {grp: [sum(res_.get(s, False) for s in SHEETS if SHEETS[s].group == grp),
                                    sum(s in res_ for s in SHEETS if SHEETS[s].group == grp)] for grp in GROUPS}
            summary[label]["failed"] = sorted(s for s, p in res_.items() if not p)
        summary["dense_grids"] = {s: g for s, (g, _) in best_grid.items() if g > 65}
        summary["median_seconds_4225"] = float(np.median(secs[key])) if secs[key] else None
        out[f"{key[0]}|noise={key[1]}"] = summary
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ["main"]
    out = table(names)
    for k, v in out.items():
        f = lambda d: " ".join(f"{g[:4]} {d[g][0]}/{d[g][1]}" for g in GROUPS)
        print(f"{k:34s} 4225: {f(v['4225'])} | enough: {f(v['enough'])} | {v['median_seconds_4225']:.1f}s")
        print(f"{'':34s} fails at enough: {v['enough']['failed']}")
    (RUNS / f"table_{'_'.join(names)}.json").write_text(json.dumps(out, indent=1))
