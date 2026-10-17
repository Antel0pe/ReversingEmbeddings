# Focused follow-up after the initial interpolation failure

The base results showed accurate small-patch cubic fits and persistent error
across a cap transition. This follow-up extends the SAME piecewise-polynomial
route with one generic learned crease; it is not a new full-family benchmark.

The learner receives only training image arrays and anonymous supplied five-
coordinate rows. A local cube is selected by the evaluator, which knows the
renderer. Selecting a transition-centered cube is additional sampling design
in the evaluator, explicitly reported. Its boundary orientation, position,
physical labels, rendering formula, and test arrays are not supplied.

Fit a cubic polynomial plus max(0, v dot u - b) times a quadratic polynomial.
Here u is column-standardized input. v,b are fitted from data by variable
projection, initialized along generic coordinate axes and deterministic random
unit directions. The plane has no preselected orientation. First reduce
multiple outputs to the leading residual pixel pattern of a training cubic
fit, learn the plane on that scalar projection, then fit all output pixels
using the learned plane. This is a continuous piecewise-cubic program.

The hinge is a generic piecewise smoothness assumption, not a known cap,
strip, pixel-clipping, or subrow formula. Test states are generated only after
freezing the plane and coefficient table. Compare an ordinary cubic on the
same data. Require the unchanged empirical thresholds (max pixel 2e-6 and
image L2 1e-5). Record coefficients, training-only optimization, complexity,
and learning cost. A local success is explicitly NOT a full-family decoder,
a global chart, or recovery of the subrow raster mechanism.

The first pilot cube happened to cross a second (side) event as well as the
intended cap event. Its learned crease followed the dominant side event and
left residual error. It is retained as a multiple-event failure. A second
cube shifts only the evaluator's sampling center to isolate one transition;
this oracle-assisted localization is disclosed, and is not a learned global
partition. Fresh test states are used. Compare both information conditions,
float32 observations and float64 diagnostics. The actual isolated region and
its formulas are never given to either fit routine.
