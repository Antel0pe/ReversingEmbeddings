"""Run methods against sheets, score them, explain failures, compare with a baseline.

    python -m unfolding_suite --list                              # every sheet and method
    python -m unfolding_suite                                     # default methods, design sheets
    python -m unfolding_suite --methods isomap_k10 --sheets s_curl roll
    python -m unfolding_suite --sheets holdout --first-seed 100 --seeds 3 --name fair_check
    python -m unfolding_suite --plugin my_idea.py --methods my_idea --against baseline:isomap_k10

--sheets also takes the group names design, holdout and all. Writes
<out-root>/<name>/: results.json, layouts.npz, scorecard.png, gallery_*.png, failures/*.png.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from .diagnose import diagnose
from .methods import METHOD_NOTES, METHODS
from .scoring import TEAR_TOLERANCE, score_layout, to_grid_order
from .sheets import SHEETS, make_instance
from .verify import summary_words, verify

ROOT = Path(__file__).resolve().parent / "runs"
DEFAULT_METHODS = ["best_fit_plane", "tangent_plane", "isomap_k10", "check_k10", "convex_edge"]
GROUPS = ("design", "holdout", "all")


def expand_sheets(names):
    """Sheet names, with the group words design / holdout / all expanded in registry order."""
    out = []
    for name in names:
        if name in GROUPS:
            out += [s for s, v in SHEETS.items() if name == "all" or v.group == name]
        else:
            out.append(name)
    return list(dict.fromkeys(out))


def list_everything():
    print("SHEETS (name, family, group: note)")
    for s in SHEETS.values():
        print(f"  {s.name:15s} {s.family:9s} {s.group:7s}  {s.note}")
    print("\nMETHODS (name: what it does)")
    for name in METHODS:
        print(f"  {name:15s} {METHOD_NOTES.get(name, '')}")


def run_one(fn, instance, seed):
    """One method on one instance. A method that crashes is a failed test, not a crash."""
    start = time.time()
    try:
        answer = fn(instance.points.copy(), 2, np.random.default_rng(seed + 1000))
        flat, info = answer if isinstance(answer, tuple) else (answer, {})
        flat = to_grid_order(flat, instance)
        if info.get("edges") is not None:                 # method's indices -> paper order
            info = {**info, "edges": instance.order[np.asarray(info["edges"])]}
    except Exception as e:                                # noqa: BLE001 - any failure is a result
        return {"result": {"passed": False, "verdict": "crashed", "error": f"{type(e).__name__}: {e}",
                           "seconds": time.time() - start},
                "flat": None, "info": {}, "diagnosis": {"causes": [{"cause": "the method crashed", "plain": str(e)}]}}
    seconds = time.time() - start
    result = {**score_layout(flat, instance.uv, instance.xyz, instance.grid, instance.sheet.family),
              "seconds": seconds}
    seen = np.empty_like(instance.points)                 # the dots the method saw, in paper order
    seen[instance.order] = instance.points
    layers = verify(seen, flat)                           # no truth: dots and flat result only
    result["double_layer"] = {k: v for k, v in layers.items() if k != "per_dot"}
    result["double_layer"]["words"] = summary_words(layers)
    return {"result": result, "flat": flat, "info": info, "diagnosis": diagnose(flat, instance, result, info)}


def failure_words(r):
    """The cell text for a scorecard: what went wrong, most serious first."""
    if r["verdict"] == "crashed":
        return "crashed"
    if r["passed"]:
        return "pass"
    if r["verdict"] == "folded":
        return f"folded {100 * max(r['fold_share'], r['overlap_share']):.0f}%"
    if max(r["tear_share"], r["crush_share"]) > TEAR_TOLERANCE:
        return "torn"
    if r["square"] is False:
        return f"not square {r['square_error_median']:.0f}%"
    return "rough"


def worst_of(rows):
    """A test passes only if it passes on every seed. Report the worst seed."""
    def badness(row):
        r = row["result"]
        return (r["passed"], -r.get("fold_share", 1), -r.get("square_error_median", 1e9))

    return min(rows, key=badness)


def compare(summary, baseline, against_method=None):
    """What did this run fix and what did it break, against a saved run (or one of its methods)?"""
    fixed, broken, still_failing = [], [], []
    for test, row in summary.items():
        old = baseline.get(f"{row['sheet']} / {against_method}" if against_method else test)
        if old is None:
            continue
        if row["passed"] and not old["passed"]:
            fixed.append(test)
        elif old["passed"] and not row["passed"]:
            broken.append(test)
        elif not row["passed"]:
            still_failing.append(test)
    return {"fixed": fixed, "broken": broken, "still_failing": still_failing}


def draw_galleries(detail, summary, methods, sheets, out, gallery):
    for family in ("bent", "stretched"):
        gallery(detail, summary, methods, [s for s in sheets if SHEETS[s].family == family], family,
                out / f"gallery_{family}.png")


def redraw(out, gallery):
    """Redraw the galleries from saved layouts, without running any method."""
    saved = json.loads((out / "results.json").read_text())
    summary, settings = saved["tests"], saved["settings"]
    layouts = np.load(out / "layouts.npz")
    stored = {key.rsplit("|", 1)[0]: key for key in layouts.files}
    detail = {}
    for test in summary:
        sheet_name = test.split(" / ")[0]
        seed = int(stored[test].rsplit("|", 1)[1]) if test in stored else 0
        detail[test] = {"flat": layouts[stored[test]] if test in stored else None,
                        "instance": make_instance(SHEETS[sheet_name], settings["grid"], seed, settings["noise"])}
    draw_galleries(detail, summary, settings["methods"], settings["sheets"], out, gallery)
    print(f"Redrew {out}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--sheets", nargs="*", default=["design"], help="sheet names, or design / holdout / all")
    ap.add_argument("--grid", type=int, default=65, help="nodes per side; grid-1 must divide by 4")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=0, help="start here; use fresh seeds for a fair final check")
    ap.add_argument("--noise", type=float, default=0.0, help="3-D noise, in grid spacings")
    ap.add_argument("--name", default="latest", help="folder under --out-root for this run")
    ap.add_argument("--out-root", type=Path, default=ROOT, help="where runs are saved (default: unfolding_suite/runs)")
    ap.add_argument("--list", action="store_true", help="list every sheet and method, then stop")
    ap.add_argument("--plugin", help="a .py file that registers more methods or sheets")
    ap.add_argument("--redraw", action="store_true", help="redraw the galleries of run --name from its saved layouts")
    ap.add_argument("--against", help="saved run to compare with, as NAME or NAME:METHOD")
    a = ap.parse_args(argv)

    if a.plugin:
        spec = importlib.util.spec_from_file_location("plugin", a.plugin)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    from .report import failure_figure, gallery, scorecard  # after plugins, and only when needed

    if a.list:
        return list_everything()
    if a.redraw:
        return redraw(a.out_root / a.name, gallery)
    a.sheets = expand_sheets(a.sheets)

    unknown = [m for m in a.methods if m not in METHODS] + [s for s in a.sheets if s not in SHEETS]
    if unknown:
        sys.exit(f"unknown: {unknown}. Methods: {sorted(METHODS)}. Sheets: {sorted(SHEETS)}.")
    out = a.out_root / a.name
    (out / "failures").mkdir(parents=True, exist_ok=True)
    for old in (out / "failures").glob("*.png"):
        old.unlink()

    summary, detail = {}, {}
    for sheet_name in a.sheets:
        s = SHEETS[sheet_name]
        for method_name in a.methods:
            runs = []
            for seed in range(a.first_seed, a.first_seed + a.seeds):
                inst = make_instance(s, a.grid, seed, a.noise)
                runs.append({**run_one(METHODS[method_name], inst, seed), "instance": inst, "seed": seed})
            worst = worst_of(runs)
            test = f"{sheet_name} / {method_name}"
            r = worst["result"]
            passes = sum(x["result"]["passed"] for x in runs)
            summary[test] = {"sheet": sheet_name, "method": method_name, "family": s.family,
                             "passed": passes == a.seeds, "seeds_passed": passes, "seeds": a.seeds,
                             "words": failure_words(r), "worst": r,
                             "causes": worst["diagnosis"]["causes"]}
            detail[test] = worst
            layers = r.get("double_layer", {}).get("words", "-")
            print(f"{'PASS' if summary[test]['passed'] else 'FAIL'}  {test:42s} {failure_words(r):16s} "
                  f"layers: {layers:24s} {r['seconds']:5.1f}s" + ("" if summary[test]["passed"] else
                                             f"  <- {worst['diagnosis']['causes'][0]['cause']}"
                                             if worst["diagnosis"]["causes"] else ""), flush=True)
            if not summary[test]["passed"] and worst["flat"] is not None:
                failure_figure(test, worst, out / "failures" / f"{sheet_name}__{method_name}.png")

    settings = {k: v for k, v in vars(a).items() if k not in ("plugin", "redraw", "list", "out_root")}
    (out / "results.json").write_text(json.dumps({"settings": settings, "tests": summary}, indent=1))
    np.savez_compressed(out / "layouts.npz", **{f"{test}|{row['seed']}": row["flat"] for test, row in detail.items()
                                                if row["flat"] is not None})
    scorecard(summary, a.methods, a.sheets, a, out / "scorecard.png")
    draw_galleries(detail, summary, a.methods, a.sheets, out, gallery)
    for method_name in a.methods:
        rows = [v for v in summary.values() if v["method"] == method_name]
        print(f"{method_name:18s} {sum(r['passed'] for r in rows):2d} of {len(rows)} tests pass")
    if a.against:
        run_name, _, against_method = a.against.partition(":")
        base = json.loads((a.out_root / run_name / "results.json").read_text())["tests"]
        diff = compare(summary, base, against_method or None)
        print(f"\nAgainst {a.against}:")
        for label in ("fixed", "broken", "still_failing"):
            print(f"  {label}: {len(diff[label])}" + "".join(f"\n    {t}" for t in diff[label]))
        (out / "comparison.json").write_text(json.dumps({"against": a.against, **diff}, indent=1))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
