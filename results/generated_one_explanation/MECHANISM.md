# The recovered mechanism, on one page

**We recovered a five-coordinate symbolic explanation within a declared
straight-strip and raster-program grammar. Images alone do not uniquely
identify the original controls.** Anonymous single-control intervention links
selected the control families; they supplied neither control names nor values.
The exact allowed range box remains evaluator-supplied.

An image is a 28x28 array of ink coverage. There is one straight stroke,
with horizontal caps and constant horizontal width. First read these geometric
measurements from row sums and cumulative row ink:

- `t,b`: top and bottom vertical coordinates, measured downward in pixels.
- `w`: width; a complete row's total ink equals this width.
- `a,s`: line intercept and slope, with centerline `center(y)=a-s*y`.

If `r0,r1` are first and last inked rows and `m_r` is a row's total ink,

```
w = median complete-row total ink
t = r0 + 1 - m_r0/w
b = r1 + m_r1/w
```

In each of two complete rows, choose an integer column boundary `k` with
cumulative ink `C(k)` closest to `w/2`. Its row-midpoint center is
`k-C(k)+w/2`. Those two centers determine `a,s`. The report proves this
boundary lies inside all relevant strip cross-sections over the default box.

The intervention links selected these small expressions:

```
z0 = t+b                   # twice vertical center
z1 = w                     # horizontal width
z2 = s                     # tan(lean angle)
z3 = a-s*(t+b)/2           # horizontal center
z4 = t-b                   # negative height
```

Their inverse is `t=(z0+z4)/2`, `b=(z0-z4)/2`, `w=z1`, `s=z2`, and
`a=z3+z0*z2/2`. Only permutation, independent rescaling/sign, and the
explicit slope-to-angle function are needed to match the original knobs.
An unrestricted fitted mixing was not used for comparison.

Changing `z0` moves both caps together; changing `z4` changes their separation;
`z1` separates the two sides; `z3` moves the centerline horizontally;
`z2` tilts it about the stroke's vertical center. The effect on pixels depends
on the current state, but each control's meaning remains fixed.

The decoder selected **16 subrows** from the supplied raster candidates.
For each subrow, take its vertical overlap with `[t,b]`. At its midpoint `y`,
intersect the horizontal stroke interval
`[a-s*y-w/2, a-s*y+w/2]` with each pixel column. Multiply the two overlap
lengths, then sum subrows within each pixel row. This short piecewise program
explains every pixel and automatically handles clipping-branch transitions.
It differs measurably from exact area integration of an ideal parallelogram.

On **11,908 held-out states**, including lattice corners and exact/near
rendering transitions:

| Observations | Maximum pixel error | Maximum Euclidean image error |
|---|---:|---:|
| float64 | 1.222e-14 | 4.990e-14 |
| float32 | 9.231e-8 | 3.217e-7 |

A separately implemented checker also passed fresh oracle comparisons,
unseen donor-control copies, full joint reconstruction, and collision checks.
Action-order and joint-interpolation tests passed their predefined thresholds.
These maxima are empirical. Algebra establishes the conditional decoder
identity and full-domain correspondence; it does not certify these particular
floating-point maxima across the continuous domain.

Three limits remain central:

1. Five inputs imply at most five continuous dimensions here. A valid global
   read-off supplies the additional evidence for dimension five. It does not
   identify a preferred meaning for the controls.
2. Invertible mixtures reconstruct exactly the same images, even with dense
   observations. Pure, globally consistent intervention links exclude mixing
   under the stated regularity/coverage assumptions, leaving independent
   scalar reparameterization. They are not proved to be minimum information.
3. Two physical settings can have identical float32 images yet different
   outputs after the same physical increment. Exact unique inversion and
   universal bitwise future prediction from the stored image are impossible.
   Bounded-error reconstruction remains meaningful.

The structural grammar, intervention purity, and original support boundaries
are supplied assumptions. Truly blind recovery without such assumptions is
unsupported. The full [report](REPORT.md) gives proofs, failure cases,
precision bounds, information separation, and what might transfer to real data.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_explanation/run.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/generated_one_explanation/check.py --oracle
```
