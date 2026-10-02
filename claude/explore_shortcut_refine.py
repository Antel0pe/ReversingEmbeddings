"""Re-evaluate the 100 valid height-detour routes with a finer path grid."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_lean_shortcut import length, smooth_route


def main():
    original = json.loads(Path(__file__).with_name("shortcut_control_results.json").read_text())
    records = []
    for row in original["records"]:
        start = np.asarray(row["start"])
        end = np.asarray(row["end"])
        depth = row["height_headroom"] * row["best_fraction_of_headroom"]
        direct = length(smooth_route(start, end, 0, 0, 4096))
        detour = length(smooth_route(start, end, depth, 0, 4096))
        records.append({**row, "direct_length_4096": direct, "detour_length_4096": detour,
                        "gain_percent_4096": 100 * (1 - detour / direct)})
    gains = np.array([r["gain_percent_4096"] for r in records])
    summary = {
        "case_count": len(records),
        "gain_percent_quantiles_4096": dict(zip(["min", "p10", "median", "p90", "max"], np.quantile(gains, [0,.1,.5,.9,1]).tolist())),
        "cases_gain_over_1_percent": int(np.count_nonzero(gains > 1)),
        "cases_gain_over_5_percent": int(np.count_nonzero(gains > 5)),
        "cases_with_positive_gain": int(np.count_nonzero(gains > 0)),
        "records": records,
    }
    destination = Path(__file__).with_name("shortcut_refined_results.json")
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="records"}, indent=2))


if __name__ == "__main__":
    main()
