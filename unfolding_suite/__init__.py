"""Unfolding test suite: score methods that flatten a 2-D sheet crumpled in 3-D.

Start with README.md in this folder. The pieces most code needs:

    from unfolding_suite import SHEETS, METHODS, make_instance, score, method, sheet

    inst = make_instance(SHEETS["s_curl"], grid=65, seed=0)   # inst.points is all a method sees
    flat, info = METHODS["check_k10"](inst.points, 2, numpy.random.default_rng(0))
    result = score(flat, inst)                                # result["passed"], ["verdict"], ...
"""
from .diagnose import diagnose
from .methods import METHOD_NOTES, METHODS, layout_from_links, method
from .scoring import score, score_layout
from .sheets import SHEETS, Instance, Sheet, make_instance, sheet

__all__ = ["SHEETS", "METHODS", "METHOD_NOTES", "Sheet", "Instance", "sheet", "method", "make_instance",
           "score", "score_layout", "diagnose", "layout_from_links"]
