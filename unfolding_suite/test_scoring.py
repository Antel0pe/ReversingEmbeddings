"""Do the suite's own checks work? Each test builds a layout whose flaw is known exactly.

Run from the repo root: .venv/bin/python -m unfolding_suite.test_scoring
"""
import numpy as np

from .diagnose import diagnose
from .methods import METHODS
from .run import run_one
from .scoring import score, score_layout, signed_area, triangles
from .sheets import SHEETS, make_instance

GRID = 65
I = make_instance(SHEETS["roll"], GRID, 0)
UV, XYZ = I.uv, I.xyz


def check(flat, family="bent"):
    return score_layout(flat, UV, XYZ, GRID, family)


def test_truth_is_certified():
    r = check(UV)
    assert r["verdict"] == "certified" and r["square"] and r["smooth"] and r["passed"]
    assert r["square_error_worst"] < 1e-9 and r["overlap_share"] == 0


def test_turning_mirroring_and_scaling_do_not_matter():
    a = 1.1
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([1, -1])
    r = check(3.7 * UV @ rot + 5)
    assert r["passed"] and r["square_error_worst"] < 1e-9 and abs(r["scale_to_paper"] - 1 / 3.7) < 1e-9


def test_answer_in_the_shuffled_order_is_put_back():
    assert score(UV[I.order], I)["passed"]
    assert score(UV, I)["verdict"] == "folded"          # right points, wrong order


def test_a_fold_is_caught():
    flat = UV.copy()
    flat[flat[:, 0] > 0.5, 0] = 1.0 - flat[flat[:, 0] > 0.5, 0]
    r = check(flat)
    assert r["verdict"] == "folded" and not r["passed"]
    assert abs(r["fold_share"] - 0.25) < 0.03 and r["overlap_share"] > 0.15


def test_wrapping_over_itself_with_no_flipped_triangle_is_caught():
    theta = (UV[:, 0] + 1) * np.pi * 1.25
    r_ = 2 + 0.9 * UV[:, 1] + 0.12 * theta
    flat = np.c_[r_ * np.cos(theta), r_ * np.sin(theta)]
    area = signed_area(flat, triangles(GRID))
    assert (area > 0).all() or (area < 0).all(), "this layout must have no flips at all"
    r = check(flat)
    assert r["verdict"] == "folded" and r["outer_edge_crossings"] > 0 and r["overlap_share"] > 0.05


def test_a_distant_overlap_is_caught():
    flat = UV.copy()
    flat[flat[:, 0] > 0, 0] -= 1.0
    r = check(flat)
    assert r["verdict"] == "folded" and r["overlap_share"] > 0.4


def test_a_tear_is_not_a_fold_but_still_fails():
    flat = UV.copy()
    right_half = np.arange(GRID * GRID) % GRID > GRID // 2      # by grid column, so the seam is clean
    flat[right_half, 0] += 1.5
    r = check(flat)
    assert r["verdict"] == "certified" and r["tear_share"] > 0 and not r["smooth"] and not r["passed"]


def test_collapsing_the_sheet_is_caught():
    flat = UV.copy()
    flat[:, 1] = 0
    r = check(flat)
    assert r["verdict"] == "folded" and not r["passed"]


def test_swapping_two_far_points_is_caught():
    flat = UV.copy()
    flat[[100, 3000]] = flat[[3000, 100]]
    r = check(flat)
    assert r["verdict"] == "folded" and r["tear_share"] > 0 and not r["passed"]


def test_small_wobble_is_not_called_a_fold():
    cell = 2 / (GRID - 1)
    rng = np.random.default_rng(1)
    assert check(UV + rng.normal(0, 0.01 * cell, UV.shape))["verdict"] == "certified"
    r = check(UV + rng.normal(0, 0.2 * cell, UV.shape))
    assert r["verdict"] == "wobble only"


def test_wobble_large_enough_to_flip_coarse_triangles_is_a_fold():
    r = check(UV + np.random.default_rng(2).normal(0, 3.0 * 2 / (GRID - 1), UV.shape))
    assert r["verdict"] == "folded"


def test_a_stretched_sheet_is_not_marked_on_squareness():
    assert check(UV * [1, 3], "stretched")["square"] is None
    assert check(UV * [1, 3])["square"] is False


def test_every_sheet_is_valid():
    for name, s in SHEETS.items():
        inst = make_instance(s, GRID, 3)
        T = triangles(GRID)
        assert (signed_area(inst.uv, T) > 0).all(), name
        assert score_layout(inst.uv, inst.uv, inst.xyz, GRID, s.family)["passed"], name
        e = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
        ratio = np.linalg.norm(inst.xyz[e[:, 0]] - inst.xyz[e[:, 1]], axis=1) \
            / np.linalg.norm(inst.uv[e[:, 0]] - inst.uv[e[:, 1]], axis=1)
        if s.family == "bent":
            assert 0.97 < ratio.min() and ratio.max() < 1.001, f"{name} stretches: {ratio.min()} {ratio.max()}"
        else:
            assert ratio.max() > 1.05, f"{name} is labelled stretched but is not"


def test_the_method_cannot_see_the_layout():
    assert not np.allclose(I.points[:, :2], I.xyz[I.order][:, :2])
    assert not np.array_equal(I.order, np.arange(len(I.order)))


def test_a_broken_method_is_recorded_not_fatal():
    def crash(points, dim, rng):
        raise RuntimeError("boom")

    def wrong_shape(points, dim, rng):
        return np.zeros((3, 2))

    def nans(points, dim, rng):
        return np.full((len(points), 2), np.nan)

    for fn, text in [(crash, "boom"), (wrong_shape, "expected shape"), (nans, "NaN")]:
        out = run_one(fn, I, 0)
        assert out["result"]["passed"] is False and text in out["result"]["error"], out["result"]


def test_diagnosis_names_the_right_cause():
    r = check(UV)
    far = np.array([[0, GRID * GRID - 1], [5, 6]])
    d = diagnose(UV, I, {**r, "verdict": "folded", "passed": False}, {"edges": far})
    assert d["hops_jumping"] == 1 and d["causes"][0]["cause"] == "hops jumped between layers"
    flat = UV.copy()
    flat[flat[:, 0] > 0.5, 0] = 1.0 - flat[flat[:, 0] > 0.5, 0]
    d = diagnose(flat, I, check(flat), {})
    assert d["example"]["paper_distance"] > 0.3 and abs(d["folded_paper"] - 0.25) < 0.03
    assert d["causes"][0]["cause"] == "unexplained"


def test_the_named_methods_exist():
    assert {"tangent_plane", "best_fit_plane", "isomap_k10"} <= set(METHODS)


if __name__ == "__main__":
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:  # noqa: BLE001 - report every test, even one that crashes
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    raise SystemExit(1 if failed else 0)
