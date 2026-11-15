"""Is each sheet solvable at all? (2026-11-15)

A sheet stays in the suite only if SOME method can pass it when handed perfect information, so that a
failure is the method's fault and not the input's. Three perfect-information layouts, each given the
paper's true neighbours (never available to a real method):

  walk_true   walking along the true 10 nearest paper neighbours, then the best flat layout
              (from 1,500 landmark dots above 4,225 dots, to keep memory small)
  vote_true   carry two directions + every connection votes, on the true 10 nearest paper neighbours
  tutte_true  the paper's own triangle mesh (no crossings), its true edge in order on a circle, every
              inside dot at the average of its neighbours (Tutte: cannot fold). Never the square, so it
              can only pass sheets judged on one-to-one and smoothness (stretched / non-square paper)

Each sheet runs at the dot count it needs (runs/enough_dots/density.json; tight_roll 257 x 257).

Run: OMP_NUM_THREADS=1 .venv/bin/python -m unfolding_suite.experiments.solvable
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from unfolding_suite.experiments.pipeline import true_circle, true_mesh
from unfolding_suite.experiments.press_flat import true_links
from unfolding_suite.experiments.simple_score import needed_grid
from unfolding_suite.methods import layout_from_links
from unfolding_suite.run import failure_words
from unfolding_suite.scoring import score
from unfolding_suite.sheets import SHEETS, make_instance
from unfolding_suite.simple import carry_vote_layout, landmark_layout

OUT = Path(__file__).resolve().parent.parent / "runs" / "solvable"
SEEDS = (100, 101, 102)


def one(job):
    sheet, seed, grid, how = job
    inst = make_instance(SHEETS[sheet], grid, seed)
    links = np.unique(np.sort(true_links(inst), 1), axis=0)
    try:
        if how == "walk_true":
            flat = layout_from_links(inst.points, links, 2)[0] if grid <= 65 else \
                landmark_layout(inst.points, links, 2, np.random.default_rng(seed))
        elif how == "vote_true":
            flat = carry_vote_layout(inst.points, links, 2)
        else:
            flat = true_circle(inst, true_mesh(inst))
        r = score(flat, inst)
        return job, {"passed": bool(r["passed"]), "words": failure_words(r)}
    except Exception as e:                                       # noqa: BLE001
        return job, {"passed": False, "words": f"crashed: {e}"}


if __name__ == "__main__":
    jobs = []
    for s in SHEETS:
        g = needed_grid(s)
        for how in ("walk_true", "vote_true", "tutte_true"):
            for seed in (SEEDS if g <= 129 else SEEDS[:1]):
                jobs.append((s, seed, g, how))
    rows = []
    with Pool(4) as pool:
        for (s, seed, g, how), r in pool.imap_unordered(one, jobs):
            rows.append({"sheet": s, "seed": seed, "grid": g, "control": how, **r})
            print(f"{'PASS' if r['passed'] else 'FAIL'} {s:15s} {g} {seed} {how:10s} {r['words']}", flush=True)
    out = {}
    for s in SHEETS:
        by = {h: all(r["passed"] for r in rows if r["sheet"] == s and r["control"] == h)
              for h in ("walk_true", "vote_true", "tutte_true")}
        out[s] = {"solvable": any(by.values()), **by}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "solvable.json").write_text(json.dumps({"rows": rows, "by_sheet": out}, indent=1))
    print({s: v for s, v in out.items() if not v["solvable"]})
