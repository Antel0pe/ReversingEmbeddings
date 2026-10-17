# Frozen protocol: conditional mechanism recovery from generated-1 observations

Written before fitting this experiment. Scope: one family, one compact program,
one extra observation type, no neural models or benchmark sweep.

## Claims kept separate

1. Five real inputs imply an upper bound of five on continuous intrinsic
   dimension for this locally Lipschitz renderer. They do not imply rank five,
   injectivity, unique semantics, or a five-dimensional finite stored set.
2. Known inputs plus their generating program already provide reconstructive
   coordinates. Recovering such a program from observations is a separate task.
3. Image-only semantics are not identifiable under invertible coordinate mixing.
   We accept permutation, sign, units, and simple independent invertible scalar
   functions; reject mixing when claiming recovery of individual physical knobs.

## Information contract

The discovery module receives raw coverage images and, in the added-information
condition, pairs and stable anonymous IDs: exactly one hidden factor changed,
and the same ID has the same meaning at every base state. No hidden values,
step sizes, action names, renderer import, source code, or range box enter it.
Nearby images alone do not supply these IDs or the single-factor guarantee.
Evaluation source/donor copy tasks ask to replace one factor with the donor's
value. Their target arrays are hidden from prediction. Copy semantics are an
explicit test specification, not silently inferred from anonymous pairs.

## Explicit structural priors (substantial)

Noise-free untransformed 28x28 nonnegative ink coverage; one unclipped straight
constant-horizontal-width strip, horizontal caps, enough full rows, and an
integer boundary common to the strip across each selected full row. Pixel grid
and coverage units are known. Candidate raster programs use exact vertical
interval overlap, a uniform midpoint grid, horizontal interval overlap, and
summation. Candidate subrow counts are {1,2,4,8,16,32,64}; true count is not
supplied to discovery. This is selection inside a supplied program grammar,
not unrestricted recovery of rendering physics. Sensor and geometry priors
were motivated by inspection of this repository; they are renderer knowledge
and must never be described as learned from pixels alone.

Geometric feature coordinates are top, bottom, horizontal centerline intercept,
width, and signed slope. Anonymous-factor invariants are sought as degree-one
then degree-two polynomials in these features. Choose the smallest admissible
degree, require a unique numerical null direction, and snap coefficients to
small rationals (denominators <=16, relative tolerance 2e-5). These are explicit
symbolic simplicity assumptions; failure rejects this route.

## Success fixed before fitting

- Float64 reconstruction: max pixel <=5e-11; image L2 <=5e-10.
- Float32 reconstruction: max pixel <=2e-6; image L2 <=1e-5.
  These are empirical acceptance thresholds, not certified universal bounds.
- Same thresholds for withheld source/donor interventions, simultaneous copies,
  joint interpolation, and both action orders; reject any valid-state failures.
- Hold out random joint states, a full lattice including corners, near-faces,
  exact/near subrow cap and side-column transitions, cumulative-boundary ties.
- Select subrow count using training images only; freeze the symbolic program
  before opening test arrays or ground-truth settings.
- Compare controls using only a permutation, independent affine calibration,
  and the declared scalar angle<->slope conversion. No multivariate alignment.
- Test complete-domain coverage in both directions by a mathematical argument
  for the conditional program. Do not turn finite testing into that argument.
  The original allowed coordinate box is supplied to the evaluator only;
  finite observations cannot identify an exact continuous support boundary.
- Search for exact stored-image collisions, including divergent outputs after
  the same physical increment. Unique physical inversion is not required for
  indistinguishable float32 observations. Bitwise action prediction can fail.
- Diagnose finite-scale tangent residuals, cap kinks, float32 quantization, and
  raster-model error separately. Record worst errors and offending states.

The added pair information can identify factor foliations under global purity,
coverage, invertibility, and regularity assumptions. It leaves permutation and
independent scalar reparameterization. We do not claim it is minimum necessary;
finite pairs and this polynomial search do not prove that general theorem's
hypotheses from data.
