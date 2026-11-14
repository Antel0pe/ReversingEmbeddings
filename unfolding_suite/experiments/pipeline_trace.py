"""Trace every pipeline failure to the decision that leaked (reads runs/pipeline/*.json).

For each sheet and unfold method, a seed's failure is blamed on the first decision whose perfect
version fixes it:
  neighbours   the pipeline fails, the same steps with the true neighbours pass
  edge         (edge unfold) true neighbours still fail, true neighbours + true edge pass
  crossing     true neighbours + true edge fail, the true non-crossing mesh + true edge in order passes
  none fixes   fails even with every perfect input: the unfold / flip / settle steps themselves
Also counts what the check-and-flip steps did on their own: flips that made things worse, folds the
face-down detector missed, folds it found but the flip did not fix, and double-layer false alarms.

Run: .venv/bin/python -m unfolding_suite.experiments.pipeline_trace [main dense]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs" / "pipeline"


def load(parts):
    rows = []
    for p in parts:
        f = RUNS / f"{p}.json"
        if f.exists():
            rows += json.loads(f.read_text())
    return rows


def passed(r, which):
    return bool(r.get(which, {}).get("passed")) if "error" not in r else False


def blame(by, which):
    own = by.get("own")
    if own is None:
        return None
    if passed(own, which):
        return "ok"
    for control, cause in (("true_links", "neighbours"), ("true_links_edge", "edge"),
                           ("true_mesh_circle", "crossing")):
        if control in by and passed(by[control], which):
            return cause
    return "none fixes"


def main(parts):
    rows = load(parts)
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["sheet"], r["grid"], r["unfold"], r["seed"])][r["control"]] = r
    table = defaultdict(lambda: {"naive": Counter(), "guarded": Counter()})
    for (sheet, grid, unfold, seed), by in groups.items():
        for which in ("final", "guarded"):
            b = blame(by, which)
            if b:
                table[(sheet, grid, unfold)]["naive" if which == "final" else "guarded"][b] += 1
    out = {"by_sheet": {f"{s}|{g}|{u}": {k: dict(v) for k, v in t.items()} for (s, g, u), t in sorted(table.items())}}

    steps = Counter()
    examples = defaultdict(list)
    for r in rows:
        if r["control"] != "own" or "error" in r:
            continue
        u = r["after_unfold"]
        tag = f"{r['sheet']}|{r['grid']}|{r['unfold']}|{r['seed']}"
        steps["runs"] += 1
        if not u["passed"]:
            steps["unfold folded or failed"] += 1
            if u["fold"] > 0.002 and r["face_down_found"] < 0.002:
                steps["fold present, detector found no face-down patch"] += 1
                examples["missed"].append(tag)
            if r["flipped"] and not r["after_flip"]["passed"]:
                steps["flipped but still failing"] += 1
                examples["flip did not fix"].append((tag, r.get("crease_bend_spacings")))
            if r["flipped"] and r["after_flip"]["passed"] and not r["after_settle"]["passed"]:
                steps["flip fixed it, settling broke it"] += 1
                examples["settle broke"].append(tag)
        else:
            if r["flipped"]:
                steps["unfold passed but detector still flipped"] += 1
                if not r["final"]["passed"]:
                    steps["...and the naive pipeline then failed"] += 1
                    examples["false flip"].append(tag)
        if u["passed"] and u["check"] == "stacked":
            steps[f"check false alarm ({r['family']})"] += 1
        if not u["passed"] and u["fold"] > 0.002 and u["check"] in ("clean", "specks"):
            steps["check missed a fold"] += 1
            examples["check missed"].append(tag)
        if r["flipped"] and r["final"]["passed"] and not u["passed"]:
            steps["flip rescued it (naive)"] += 1
            examples["rescued"].append(tag)
        if r.get("guarded", {}).get("passed") and not u["passed"]:
            steps["guarded rescued it"] += 1
    out["steps"] = dict(steps)
    out["examples"] = {k: v for k, v in examples.items()}
    for k, v in sorted(out["by_sheet"].items()):
        if v["guarded"].get("ok", 0) < sum(v["guarded"].values()) or v["naive"].get("ok", 0) < sum(v["naive"].values()):
            print(f"{k:32s} naive {v['naive']}  guarded {v['guarded']}")
    print(json.dumps(out["steps"], indent=1))
    for k, v in out["examples"].items():
        print(k, v[:12])
    (RUNS / f"trace_{'_'.join(parts)}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["main"])
