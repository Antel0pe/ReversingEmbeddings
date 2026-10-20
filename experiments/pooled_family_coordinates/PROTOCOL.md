# Common perpendicular measurements with controlled tolerance

Protocol written before fitting this experiment. The question is whether pooled
families admit useful scalar readings with small unwanted responses, and whether
state-dependent readings improve on one fixed pixel-weight vector.

## Inputs and isolation

Use only anonymous image/vector observations from the previous blind experiment,
its initial distance/angle families, and newly acquired image-only small patches.
No original coordinates, shape readers, region markers, renderer equations or
analytic derivatives enter discovery. Acquisition can use the simulator to obtain
images; its five-control sampling design is an explicit prior. Five readings are
a requested target. New original settings stay in a private evaluation archive.
Discovery is an isolated process with read access checks. Freeze sources, inputs,
models and blind results before physical evaluation. Do not change discovery
after physical evaluation. Researcher knowledge of this dataset is acknowledged.

## Scopes

Reuse the previous branch and broad observations. Add image-distance balls of
radii 0.005, 0.05, 0.25 and 1 around the canonical root and one predetermined
source (the first broad training-frame state in the previous acquisition).
Each patch has 1,536 image training observations, 256 validation observations,
256 untouched checking observations, and 96/32/32 independent fitting/validation/
checking frames. Each new frame has 192 equal-image-distance neighbors at radius
0.0001, and both ordinary SVD and centered PCA fits. Root neighborhoods are
two-sided and can extend outside the default position bounds. The predetermined
second source is not selected for smoothness or favorable results.

## Methods

All family assignments and signs use images and vectors only. The five readings
are solved simultaneously from all fitting-state constraints. A chosen reading's
mean response to its assigned family is normalized to one; constant/zero
measurements cannot pass. No physical label enters the weights.

1. Fixed pooled common-nullspace measurement: minimize squared pixel-gradient
   norm, subject to unit mean own-family response and an RMS response bound to
   the other four families. Try 1%, 3%, 5%, 10%, 20%, 30%, 50%, 100% of the
   calibrated unit response. Spectral inversion computes the solution directly;
   report infeasible requested bounds and held-out violations honestly.
2. Fixed pooled derivative least squares: fit all five local response matrices
   to the identity, with a pixel-gradient penalty. This controls both unwanted
   responses and variation of own-family response.
3. State-dependent scalar functions, using exactly the same derivative least
   squares. Compare raw linear pixels plus generic quadratic features of 12
   global image PCs; cubic features of 8 PCs; and 64 Gaussian kernel features
   of 12 PCs, with bandwidth multipliers 0.75 and 1.5. Each scalar function has
   an explicit gradient, not an independently assigned vector at every state.
   Thus the learned readings themselves have path-independent values.
4. Baseline: five ordinary global image PCs. A pointwise dual-frame calculation
   is only a local-response control, not a global coordinate method.

Gradient penalties are in original pixel units, including all derivatives of
nonlinear features; no whitening of rare pixel channels is used. Retain all
data-varying pixel columns. Only nonlinear dictionaries use a PC projection.
Derivative ridge parameters: 1e-5, 1e-4, 0.001, 0.01, 0.1, 1. No original labels
are used in hyperparameter selection.

## Selection and checks

Select each encoder on validation-only image reconstruction with a fixed local
linear decoder (48 neighbors), recording response error and local rank. Then
select final decoder degree/neighbors from (1,48), (2,64), (2,128), (3,128),
(3,192) using image validation only. This is an interpolation diagnostic, not
an automatically discovered exact renderer. All failures and alternatives stay
in the results. Evaluate untouched images and frames. Reuse broad paths/corners/
faces as extra checks. Report model size and regularization choice.

Report off-family RMS response, own-response variability, sign reversals,
gradient norm, local five-coordinate rank, held-out reconstruction and finite
loop circulation of the inferred local measurement fields. Distinguish scalar
readings (path independent by construction) from a moving raw SVD frame (which
need not integrate to coordinates). Requested percentages are response errors,
not retained variance, pixel errors, or a guarantee on every state.

A simple analytic curved-coordinate control uses state (u, v+u^2): the second
coordinate v is y-x^2, so a quadratic reading can be exact while no nonzero
constant gradient is perpendicular to all first-family directions. This control
is separate from the generated-one experiment and cannot supply features to it.

## Post-freeze physical comparison

Only after models and selections are frozen, fit original-coordinate readouts
from the discovered readings (global polynomial degrees 1/2/3 selected on
validation, plus a local quadratic control). Evaluate the known reverse-derived
readers separately as a benchmark, never a fitting target. Their failure to be
exclusive to raw SVD families is informative: a good coordinate system may mix
those families. Repeat read-access and finite-difference checks and verify hashes
before and after physical evaluation. No global exactness claim from finite fits.
