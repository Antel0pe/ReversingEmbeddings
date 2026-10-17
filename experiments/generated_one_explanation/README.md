# One focused generative-explanation experiment

Read the [report](../../results/generated_one_explanation/REPORT.md) and
[pre-fit protocol](PROTOCOL.md) for identifiability, information contracts,
supplied structural knowledge, and limits.

Run from the repository root with the existing environment:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_explanation/run.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_explanation/check.py --oracle
```

`discover.py` uses NumPy and accepts images plus anonymous intervention links.
It imports no generator, range box, parameter values, or oracle. `run.py` alone
constructs observations using the actual renderer and holds settings for
independent evaluation. `check.py` separately implements encoding/decoding;
without `--oracle`, it needs only NumPy and the saved result arrays.

The reported outcome is conditional symbolic program recovery, not blind
recovery of an unrestricted renderer. Image-only coordinate semantics remain
unidentifiable. The exact support box is evaluator-supplied. Stored-image
collisions explicitly limit exact physical inversion and future prediction.
