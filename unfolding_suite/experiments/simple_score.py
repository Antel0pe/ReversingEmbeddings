"""Score-only runner for choosing the best simple method (2026-11-15). No failure pages, 4 processes.

Same instances, scorer and pass rule as the suite (seeds 100-102). Each sheet is run at 4,225 dots, and
again at the dot count it needs (runs/enough_dots/density.json) when that is higher.

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.simple_score NAME [--noise X] METHOD...
Writes unfolding_suite/runs/simple/NAME.json.
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import unfolding_suite.experiments.mesh_recipe  # noqa: F401 - registers mesh_full (the graphics recipe)
import unfolding_suite.experiments.up_loops  # noqa: F401 - registers up_0.0_surface
from unfolding_suite.methods import METHODS
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order
from unfolding_suite.sheets import SHEETS, make_instance
from unfolding_suite.verify import verify

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = RUNS / "simple"
SEEDS = (100, 101, 102)
DENSE_OK = {"simple_unfold", "simple_vote", "convex_edge", "simple_vote_no_checks", "simple_vote_tilt_only",
            "simple_vote_facing_only", "best_simple", "mesh_full"}   # scale past 9,000 dots


def needed_grid(name):
    if name == "tight_roll":
        return 257                             # 1.97 dot spacings at 257 x 257: "just enough" (see README)
    d = json.loads((RUNS / "enough_dots" / "density.json").read_text()).get(name, {})
    return d.get("enough_grid") or 65


def one(job):
    sheet, method_name, seed, grid, noise = job
    inst = make_instance(SHEETS[sheet], grid, seed, noise)
    start = time.time()
    try:
        answer = METHODS[method_name](inst.points.copy(), 2, np.random.default_rng(seed + 1000))
        flat = to_grid_order(answer[0] if isinstance(answer, tuple) else answer, inst)
        r = {**score_layout(flat, inst.uv, inst.xyz, inst.grid, inst.sheet.family), "seconds": time.time() - start}
        seen = np.empty_like(inst.points)
        seen[inst.order] = inst.points
        r["layers"] = verify(seen, flat)["reading"]
    except Exception as e:                     # noqa: BLE001
        r = {"passed": False, "verdict": "crashed", "error": f"{type(e).__name__}: {e}", "seconds": time.time() - start}
    keep = {k: r.get(k) for k in ("passed", "verdict", "fold_share", "overlap_share", "square_error_median",
                                  "tear_share", "crush_share", "roughness", "seconds", "error", "layers")}
    keep["words"] = failure_words(r) if "error" not in r else "crashed"
    return job, keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--sheets", nargs="*", default=None)
    a = ap.parse_args()
    sheets = a.sheets or list(SHEETS)
    jobs = []
    for s in sheets:
        grids = sorted({65, needed_grid(s) if needed_grid(s) != 257 else 257})
        for m in a.methods:
            for g in grids:
                if g > 65 and m not in DENSE_OK:
                    continue
                for seed in (SEEDS if g <= 129 else SEEDS[:1]):
                    jobs.append((s, m, seed, g, a.noise))
    jobs.sort(key=lambda j: j[3])                  # small first
    results = []
    with Pool(4) as pool:
        for job, r in pool.imap_unordered(one, jobs):
            results.append({"sheet": job[0], "method": job[1], "seed": job[2], "grid": job[3], "noise": job[4], **r})
            print(f"{'PASS' if r['passed'] else 'FAIL'} {job[0]:15s} {job[1]:20s} {job[3]:3d} {job[2]} {r['words']:16s} "
                  f"{r['seconds']:.1f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}.json").write_text(json.dumps(results, indent=1))
    print("wrote", OUT / f"{a.name}.json")
