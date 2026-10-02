"""Control survey: does a simple height detour help across varied generated 1s?"""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_lean_shortcut import length, smooth_route
from grey_ones import RANGES


def main():
    rng = np.random.default_rng(20260930)
    records = []
    for _ in range(100):
        base = RANGES[:, 0] + rng.random(5) * (RANGES[:, 1] - RANGES[:, 0])
        start, end = base.copy(), base.copy()
        start[4] = rng.uniform(-10, 0)
        end[4] = rng.uniform(25, 35)
        headroom = start[2] - RANGES[2, 0]
        fractions = np.linspace(0, 1, 9)
        lengths = np.array([length(smooth_route(start, end, headroom * f, 0, 512)) for f in fractions])
        best = int(np.argmin(lengths))
        records.append({
            "start": start.tolist(), "end": end.tolist(),
            "height_headroom": float(headroom), "best_fraction_of_headroom": float(fractions[best]),
            "direct_length_512": float(lengths[0]), "detour_length_512": float(lengths[best]),
            "gain_percent_512": float(100 * (1 - lengths[best] / lengths[0])),
        })
    gains = np.array([r["gain_percent_512"] for r in records])
    summary = {
        "case_count": len(records),
        "selection": "All non-lean controls random in allowed range; starting lean uniform [-10,0], ending lean uniform [25,35] degrees",
        "candidate_height_depths": "Nine fractions from 0 to all available headroom above minimum height 19",
        "gain_percent_quantiles": dict(zip(["min", "p10", "median", "p90", "max"], np.quantile(gains, [0,.1,.5,.9,1]).tolist())),
        "cases_gain_over_1_percent": int(np.count_nonzero(gains > 1)),
        "cases_gain_over_5_percent": int(np.count_nonzero(gains > 5)),
        "cases_no_gain_at_grid_resolution": int(np.count_nonzero(gains <= 1e-9)),
        "records": records,
    }
    output = Path(__file__).with_name("shortcut_control_results.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="records"}, indent=2))


if __name__ == "__main__":
    main()
