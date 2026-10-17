# Generated-1 discovery with data-only inputs

The [report](../../results/generated_one_data_only/REPORT.md) records a failed
whole-family reconstruction and successful local transition learning. No
original-knob recovery is required. Do not confuse local numerical success
with a compact generative explanation of the whole family.

Discovery uses generic spatial moments and piecewise polynomial regression.
It imports no renderer, original inverse, ranges, or stroke geometry. The
optional coordinate condition receives unnamed five-number coordinate rows.
See the [base information contract](PROTOCOL.md) and
[local follow-up contract](HINGE_PROTOCOL.md), including oracle-side sampling.

Run from the repository root with the existing environment:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_data_only/run.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_data_only/hinge_run.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_data_only/check.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_data_only/figure.py
```

The independent checker requires NumPy and SciPy, uses the saved arrays, and
imports neither the discovery code nor the oracle renderer. Its passing
result verifies reproduction of both scientific failures and local successes.
