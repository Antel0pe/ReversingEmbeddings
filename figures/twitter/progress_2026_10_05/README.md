# Generated-one progress: September 28 to October 5, 2026

Attach the three numbered PNG files to the draft in `post.txt`. It has 261 ASCII characters excluding the final file newline, within the standard 280-character limit. The figures use synthetic 28-by-28 ink-coverage images, not real handwriting. No post was published.

## The three picks

1. **Use another knob to make the route cheaper.** Temporarily shorten a stroke while tilting, then restore its height. Same endpoints, 5.0515% less accumulated pixel-L2 change in the example. A survey of 100 large-tilt cases had a 2.9155% median improvement. This is a feasible shorter route, not a proof of a shortest path.
2. **Joint changes need an interaction correction.** From the midpoint state, width +1 px and lean +15 degrees give a 3.4601 pixel-L2 difference between the true joint image and the sum of separate image changes. The sum puts 26 pixels above full coverage. The precise equation for that comparison is wider image + tilted image - starting image. The full-box control, involving all five knobs, has 34.1% median relative joint-change error across 2,048 starts.
3. **Known edge geometry gives a compact, accurate tilt-response rule.** Mean relative response error is 0.6529%, versus 55.6235% for a degree-5 polynomial, evaluated on 703 unseen width/lean states and four corners. The edge rule is supplied with known geometry and fits one scalar; it does not discover the generator. The polynomial uses 16,464 coefficients. A separate 256-state five-knob control gives 0.7619% edge-rule error. These scores concern finite-difference pixel responses, not percent-correct images.

## Evidence

- `claude/GENERATED_GEOMETRY_FOLLOWUP.md`, `claude/lean_shortcut_results.json`, `claude/shortcut_control_results.json`: shorter routes and controls; committed October 1.
- `figures/generated_one_principal_curves/followup/README.md`: direct width/lean interaction example; committed October 4.
- `figures/generated_knob_metric/README.md`: five-knob addition control; committed September 30.
- `experiments/lean_field_fit_20261004/README.md` and `results/metrics.json`: edge-rule comparison and full-space control; committed October 4.

New curve-fitting results were also reviewed. High captured variation alone does not establish an exact coordinate system, so these three concrete mechanisms make a clearer progress post.

## Reproduce

From the repository root, run:

```sh
.venv/bin/python figures/twitter/progress_2026_10_05/make_figures.py
```

The script re-renders 8,193 states per shortcut route, checks equal endpoints and allowed settings, recomputes the mixed interaction, and re-evaluates the saved tilt-field models without refitting. `verification.json` records results. Source pixels and response maps share fixed scales; invalid coverage is explicitly marked. `preview.png` is only a local contact sheet for inspection.

## Image descriptions for accessibility

1. Two rows show the same synthetic stroke tilting from -10 to 35 degrees. In the second row its height temporarily falls from 20.5 to 19.03 pixels before recovering. A height-versus-tilt chart makes the dip visible. Accumulated pixel change falls from 30.73 to 29.18, a 5.05% saving.
2. Images show a starting stroke, a wider stroke, and a more tilted stroke. Below them, the actual joint change is compared with adding separate changes. Pink marks 26 impossible coverage values in the sum. An orange-and-blue correction map reveals the missing interaction.
3. A stroke and three pixel-response maps show how its pixels change with tilt. The smooth polynomial spreads the response; the geometry rule follows the actual edge pattern. A bar chart shows mean response errors of 55.62% for the degree-5 polynomial and 0.65% for the edge rule on unseen width/lean settings and corners.
