# Blind discovery from estimated direction families

Written before fitting. Discovery receives finite pixel arrays, five estimated
SVD/PCA vectors per local state, and anonymous initial family assignments from
the earlier distance/angle branch experiment. Five coordinates are a target,
not a dimension inferred by this experiment.

The exporter can use the generator to collect observations. Original settings,
derivatives, parameter names/ranges, inverse functions, prior shape readers,
cap/side formulas, and previous intermediate coordinates are not learner inputs.
Known simulator controls also affect the neighborhood sampling distribution;
this is an explicit acquisition prior, not intervention-free real data.

## Fixed discovery process

1. Transfer the existing anonymous families to fresh frames using nearest
   image plus one-to-one absolute pixel cosine. No physical labels are used.
   Across a broad sample, extend families through a nearest-neighbor spanning
   tree. Measure agreement on edges outside that tree.
2. Search for scalar image functions with strong directional response to one
   family and weak response to the other four. Use a generalized eigenproblem
   on their measured response energies, with regularization selected on new
   images/frames only. No target physical directions enter this objective.
3. Compare two generic function classes: weighted pixels; weighted pixels plus
   weighted pixels divided by total image intensity. The latter is a generic
   nonnegative-data normalization, not a supplied edge/center/shape formula.
   Both classes first reduce/whiten training-image features to 48 or 96 retained
   components (or available numerical rank). This is a candidate-function
   dictionary, not a claim these components parameterize the manifold.
4. For each family retain the three best eigenfunctions and select one per
   family using validation leakage plus a coordinate-dependence penalty. Try
   regularization 0.001, 0.01, 0.1 of mean response energy. Preserve all attempts.
5. Compare ordinary SVD and centered PCA, a shuffled-family control, and five
   ordinary global image PCs. Select the encoder using blind validation
   independence and family leakage. The shuffled control is descriptive, not
   a multi-repeat significance test. Select the encoder before decoding.
6. Fit a generic local polynomial image decoder, trying degree/neighbors
   (1,48), (2,64), (2,128), (3,128), (3,192) on validation only. Also fit a global
   cubic as a compactness control. Store model size. This is an interpolator,
   not an established exact manifold equation.

## Scope and fixed checks

First use the existing branch region and new independently perturbed states
around it. Separately run across the default family range. Include independent
held-out images, fresh local frames, joint paths, default-box corners and faces,
and a transfer check from the branch model to the broad sample. There are no
training images at held-out query states. Report errors in actual pixel units,
local five-coordinate rank/conditioning, family response leakage, loop-edge
assignment agreement, and family-vector compression on held-out frames.

Five useful measurements require both local independence and predictive image
reconstruction. Low response leakage alone is not success. Thresholds for an
exact numerical reconstruction claim are max pixel error <= 2e-6 and max image
L2 error <= 1e-5; failures remain visible. Finite checks cannot certify absence
of global coordinate collisions or mathematical exactness.

## Freeze before physical evaluation

Discovery runs in a separate process with file-access checks. It can read only
this learner's source, its approved image/vector archive, and installed numerical
libraries, and writes its own outputs. It cannot read the generator, previous
coordinate code, or the private settings archive. After all model/decoder
selection, write model files plus SHA256 hashes of learner source, inputs and
outputs. Only then run a separate physical evaluator; do not change discovery
after inspecting physical results. The experiment designer already knows this
dataset, so software isolation is not a claim of human-level blind design.

The evaluator fits maps in both directions using training labels only, selects
polynomial degrees 1/2/3 on validation, then checks untouched labels. Also report
a local quadratic mapping as an empirical, less compact control. Physical
evaluation never changes which five measurements were discovered.
