"""Ceiling for mesh_recipe: steps 3-4 given the paper's TRUE triangles (a cheat control).

Separates "the woven mesh was wrong" from "the flattening steps were wrong".
Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.mesh_recipe_truth
Writes runs/mesh_recipe/truth_mesh.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import unfolding_suite.holdout  # noqa: F401  registers the held-out sheets
from unfolding_suite.experiments import mesh_recipe as M
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order, triangles
from unfolding_suite.sheets import SHEETS, make_instance

OUT = Path(__file__).resolve().parent.parent / "runs" / "mesh_recipe"


def job(args):
    name, seed, full = args
    inst = make_instance(SHEETS[name], 65, seed)
    n = len(inst.points)
    where = np.empty(n, int)
    where[inst.order] = np.arange(n)
    T = M.orient(where[triangles(65, 1)])
    outer = max(M.boundary_loops(T), key=len)
    flat = M.flatten_mesh(inst.points, T, outer, n, full)
    r = score_layout(to_grid_order(flat[:n], inst), inst.uv, inst.xyz, 65, inst.sheet.family)
    return {"sheet": name, "seed": seed, "full": full, "passed": r["passed"], "words": failure_words(r),
            "square_error_median": r["square_error_median"], "roughness": r["roughness"]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, seed, full) for s in SHEETS for seed in (100, 101, 102) for full in (False, True)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        rows = p.map(job, jobs)
    (OUT / "truth_mesh.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['sheet']:14s} {r['seed']} {'full ' if r['full'] else 'tutte'} {r['words']}")
