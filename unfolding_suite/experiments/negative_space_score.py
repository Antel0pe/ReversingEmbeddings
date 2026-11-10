"""Score-only runner for the negative_space methods: no failure pages, four processes.

Same instances, scorer and pass rule as the suite (seeds 100-102, like runs/current).
Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.negative_space_score
Writes unfolding_suite/runs/negative_space/scores.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import unfolding_suite.experiments.negative_space  # noqa: F401 - registers the methods
from unfolding_suite.methods import METHODS
import time

import numpy as np

from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score_layout, to_grid_order
from unfolding_suite.sheets import SHEETS, make_instance

NEW = ["nearby_only", "push_random", "push_random_cold", "push_targeted", "air_probe", "air_probe_check",
       "star_tutte", "ring_circle", "ring_circle_star"]
SEEDS = (100, 101, 102)
OUT = Path(__file__).resolve().parent.parent / "runs" / "negative_space"


def one(job):
    sheet_name, method_name, seed = job
    inst = make_instance(SHEETS[sheet_name], 65, seed)
    start = time.time()
    try:                                       # the suite's run_one, minus the slow diagnosis
        answer = METHODS[method_name](inst.points.copy(), 2, np.random.default_rng(seed + 1000))
        flat = to_grid_order(answer[0] if isinstance(answer, tuple) else answer, inst)
        r = {**score_layout(flat, inst.uv, inst.xyz, inst.grid, inst.sheet.family), "seconds": time.time() - start}
    except Exception as e:                     # noqa: BLE001
        r = {"passed": False, "verdict": "crashed", "error": f"{type(e).__name__}: {e}"}
    keep = {k: r.get(k) for k in ("passed", "verdict", "fold_share", "overlap_share", "square_error_median",
                                  "square_error_worst", "roughness", "tear_share", "crush_share", "seconds", "error")}
    keep["words"] = failure_words(r)
    return job, keep


if __name__ == "__main__":
    methods = sys.argv[1:] or NEW
    jobs = [(s, m, seed) for s in SHEETS for m in methods for seed in SEEDS]
    results = {}
    with Pool(4) as pool:
        for (s, m, seed), r in pool.imap_unordered(one, jobs):
            results.setdefault(f"{s} / {m}", {})[seed] = r
            print(f"{'PASS' if r['passed'] else 'FAIL'} {s:15s} {m:18s} {seed} {r['words']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scores.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({k: {str(a): b for a, b in v.items()} for k, v in results.items()})
    path.write_text(json.dumps(old, indent=1))
    print("wrote", path)
