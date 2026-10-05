"""Separate finite-difference and integration effects from the fitted edge rule."""
import json
import numpy as np
from run import (OUT, BOUNDS, WIDTH, LEAN, edge_basis, grid, field_scores,
                 images, integrate, measured_field, path_scores, settings_from_slice)


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    fitted = metrics["models"]["piecewise_edge_rule"]["fitted_scale"]
    exact = np.pi / 180
    corners = np.asarray([(w, l) for w in BOUNDS[WIDTH] for l in BOUNDS[LEAN]])
    points = np.vstack((grid(19, 37, .371), corners))
    feature = edge_basis(settings_from_slice(points))
    report = {"finite_difference_checks": [], "integration_checks": [],
              "scope": "The same 707 evaluation states; one integration path at width 2.7324 px",
              "fitted_coefficient_relative_difference_from_pi_over_180": float(fitted/exact-1),
              "limits": "The analytic coefficient uses known geometry; decreasing numerical steps does not establish exact image membership."}
    for delta in (.05, .025, .0125, .00625):
        observed = measured_field(points, delta)
        report["finite_difference_checks"].append({
            "step_degrees": delta,
            "fitted_coefficient": field_scores(fitted*feature, observed)[0],
            "analytic_coefficient": field_scores(exact*feature, observed)[0],
        })
    for intervals in (900, 1800, 3600):
        angles = np.linspace(*BOUNDS[LEAN], intervals+1)
        path_points = np.column_stack((np.full(len(angles), 2.7324), angles))
        actual = images(settings_from_slice(path_points))
        feature = edge_basis(settings_from_slice(path_points))
        report["integration_checks"].append({
            "step_degrees": float(np.diff(angles)[0]),
            "sample_count": len(angles),
            "fitted_coefficient": path_scores(integrate(fitted*feature, angles, actual[0]), actual),
            "analytic_coefficient": path_scores(integrate(exact*feature, angles, actual[0]), actual),
        })
    (OUT / "numerical_checks.json").write_text(json.dumps(report, indent=2)+"\n")
    for item in report["finite_difference_checks"]:
        print("Derivative measurement step:", item["step_degrees"],
              "fitted mean arrow error:", item["fitted_coefficient"]["relative_arrow_error"]["mean"])
    for item in report["integration_checks"]:
        print("Integration step:", item["step_degrees"], "fitted mean image error:",
              item["fitted_coefficient"]["image_l2_error"]["mean"])


if __name__ == "__main__":
    main()
