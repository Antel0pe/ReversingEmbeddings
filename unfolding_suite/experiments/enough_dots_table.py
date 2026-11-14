"""Re-read every saved flattening run and score each method on the sheets that have enough dots.

"Enough dots" comes from runs/enough_dots/density.json (made by enough_dots.py): at the usual
4,225 dots, 18 sheets have their layers clearly farther apart than neighbouring dots. The two
rolls need more (h_roll5 16,641, tight_roll 66,049), and the four crumples cannot be measured
reliably (see the README section). A method "passes" a sheet when every saved seed passed.

Run: .venv/bin/python -m unfolding_suite.experiments.enough_dots_table
"""
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
ROLLS = ["h_roll5", "tight_roll"]
CRUMPLES = ["crumple", "hard_crumple", "h_crumple17", "h_crumple29"]


def enough_sheets():
    d = json.loads((RUNS / "enough_dots" / "density.json").read_text())
    return [s for s, r in d.items() if s not in ROLLS + CRUMPLES]


def collect():
    res = defaultdict(lambda: defaultdict(list))          # method -> sheet -> [passed per seed]

    def suite(run, rename=None):
        p = RUNS / run / "results.json"
        if not p.exists():
            return
        for key, t in json.loads(p.read_text())["tests"].items():
            s, m = key.split(" / ")
            res[(rename or {}).get(m, m)][s].append(bool(t["passed"]))

    def keyed(path, pick=None):
        for key, seeds in json.loads((RUNS / path).read_text()).items():
            s, m = key.split(" / ")
            if pick and m not in pick:
                continue
            for r in seeds.values():
                res[m][s].append(bool(r.get("passed")))

    suite("current")
    suite("pull_k10")
    suite("mesh_recipe_v1")
    keyed("negative_space/scores.json")
    keyed("height_split/scores.json")
    keyed("up_loops/score.json")
    keyed("fold_topology/flatten.json")
    for r in json.loads((RUNS / "carry_frame" / "main.json").read_text()):
        if r["links"] != "true":
            for v in ("A_tree", "B_relax", "C_stretch"):
                if v in r and r[v]:
                    res[f"carry_{v}_{r['links']}"][r["sheet"]].append(bool(r[v]["passed"]))
    for r in json.loads((RUNS / "press_flat" / "main_seeds100.json").read_text()):
        if r["push"] == 0 and r["links"] != "true":
            res[f"press_{r['links']}"][r["sheet"]].append(bool(r["passed"]))
    for r in json.loads((RUNS / "patch_stitch" / "seeds100.json").read_text()):
        if r["links"] != "true":
            res[f"grow_join_{r['links']}_{r['n_seeds']}_{'settle' if r['settle'] else 'rigid'}"][r["sheet"]].append(bool(r["passed"]))
    for r in json.loads((RUNS / "press_unfold" / "results.json").read_text()):
        if r["links"] != "true":
            res[f"press_flip_{r['links']}"][r["sheet"]].append(bool(r["unfolded_settled"]["passed"]))
    return res


if __name__ == "__main__":
    enough = enough_sheets()
    res = collect()
    rows = {}
    for m, by in sorted(res.items()):
        tried = [s for s in enough if s in by]
        passed = [s for s in tried if all(by[s])]
        rows[m] = {"enough_tried": len(tried), "enough_passed": len(passed),
                   "enough_failed": sorted(set(tried) - set(passed)),
                   "seeds": max(len(v) for v in by.values()),
                   "rolls": {s: all(by[s]) for s in ROLLS if s in by},
                   "crumples": {s: all(by[s]) for s in CRUMPLES if s in by}}
        print(f"{m:34s} {len(passed):2d}/{len(tried):2d} seeds {rows[m]['seeds']}  fails {rows[m]['enough_failed']}")
    (RUNS / "enough_dots" / "table.json").write_text(json.dumps({"enough_sheets": enough, "methods": rows}, indent=1))
