# Incremental transport of generated-1 knob vectors

The experiments use the actual `grey_ones.py` renderer, all 784 coverage pixels, and the canonical root `(14, 14, 19.75, 3.2, 0)`.
A vector is the raw pixel derivative per pixel of position/size or per degree of lean. Error means Euclidean vector error divided by the true vector length. No per-step renormalization is applied.

## Mathematical question

For a smooth decoder `F(p)` with independent coordinate derivatives, define the physical knob flow by `T_i^d(F(p)) = F(p + d e_i)`. Its differential, called the pushforward, satisfies `D T_i^d [J_j(p)] = J_j(p + d e_i)`. This follows by differentiating `T_i^d(F(p+t e_j)) = F(p+d e_i+t e_j)` with respect to `t`.
This is exact on regular smooth chart regions, for finite permitted moves as well as infinitesimal ones. It relies on independent commuting knob coordinates, not just having five vectors. An arbitrary moving frame, normalized knob field, or fixed spatial warp need not satisfy it.
Applying `T` directly to a vector is generally wrong; the vector operation is `D T`. For a linear image operator `A`, these coincide because `D A = A`.

The known-coordinate frame map is `J(new) pinv(J(old))`. It uses both measured/analytic frames, so its success is a calibration control, not a discovery algorithm. Repeated maps compose exactly in exact arithmetic. At kink states we use explicitly defined positive one-sided columns; this algebraic map is then a convention, not a unique manifold differential.

## Repeated pixel shear

The pixel baseline horizontally interpolates every row at its center, with constant-zero values outside the canvas. Both the base image and the five vectors receive the same linear operator. Slope increments are `tan(new lean)-tan(old lean)`, so the prescribed endpoint geometry is held fixed while step counts vary.
At a frozen pivot, an exact geometric shear preserves cx/height/width coordinate derivatives, but maps `J_cy` to `J_cy(new)-delta_slope*J_cx(new)` and maps `J_lean` to `sec(old lean)^2/sec(new lean)^2 * J_lean(new)`. Thus even an exact frozen geometric shear is not the pure five-coordinate lean flow in a neighborhood. The spatial operator agrees on the base path but has different derivatives on nearby states.

| Steps to 10° | Step (degrees) | Width error | Maximum of five errors | State image error |
|---:|---:|---:|---:|---:|
| 1 | 10 | 40.617% | 47.174% | 16.898% |
| 4 | 2.5 | 64.291% | 71.033% | 24.970% |
| 16 | 0.625 | 67.831% | 75.433% | 27.630% |
| 64 | 0.15625 | 68.590% | 76.410% | 28.214% |
| 256 | 0.0390625 | 68.774% | 76.647% | 28.356% |
| 512 | 0.0195312 | 68.804% | 76.686% | 28.379% |

Every step is smaller, but repeated linear interpolation diffuses pixel information. The limiting error can increase rather than vanish. The frozen-shear mixing/scaling errors are separate from interpolation errors.
Using the same `tan(delta_angle)` at every step is also not the exact angle flow: shear parameters add as slopes, not as angles. The corresponding experiment is in `results.json`.

## Exact frame control and local curvature-only integration

Seven root paths move cx, cy, height, width, lean, joint knobs, or lean to 10°. Each is tested with 1–512 steps. Exact target frames are supplied to the coordinate control.
Worst repeated known-frame error: **1.98677e-12**; 64 held-out states: **2.34496e-15**.
For a separate local integrator, change coordinates to `(top, bottom, intercept, width, slope)`, integrate `J_q += H_q(current) delta_q`, then convert back to physical coordinates. It uses current exact local curvature but omits jumps at formula boundaries. It is exact in a single renderer formula cell because the decoder there is quadratic, while it can fail to converge over raster boundary crossings. This tests that specific update rule, not all possible integrators.

| Root path | Curvature-only error, 1 step | 512 steps |
|---|---:|---:|
| cx | 141.4214% | 141.4214% |
| cy | 100.0000% | 100.0000% |
| height | 141.4214% | 141.4214% |
| width | 0.0000% | 0.0000% |
| lean | 608.6709% | 154.9843% |
| joint | 382.5637% | 159.5668% |
| lean10 | 140.8129% | 140.8129% |

## Correct transformation with valid companion images

Carry six valid images: the base and five nearby examples made by tiny positive knob probes. At every step apply the same image operation to all six: `render(invert(image) + knob_increment)`. At the endpoint, subtract the transformed base from each transformed companion and divide by its original knob probe. This approximates the correct differential without treating a signed derivative vector as an image.
This method uses the existing known-family inverse and renderer, which contain the nonlinear family information. It is not a method based only on the original five vectors. Probes can cross events; finite probe bias is intentional and is tested by refinement. Float32 storage introduces subtraction and repeated inverse/render errors.

| Storage | Fraction of knob range used for probes | Steps to 10° | Maximum of five errors |
|---|---:|---:|---:|
| float64 | 1e-05 | 1 | 0.00013849% |
| float64 | 1e-05 | 16 | 0.0001385% |
| float64 | 1e-05 | 256 | 0.000138519% |
| float32 | 0.001 | 1 | 1.15548% |
| float32 | 0.001 | 16 | 1.15519% |
| float32 | 0.001 | 256 | 1.15478% |
| float32 | 1e-06 | 1 | 3.80142% |
| float32 | 1e-06 | 16 | 12.2717% |
| float32 | 1e-06 | 256 | 10.9871% |

The same companion method also tests all seven root paths (each individual knob and two joint/lean controls) at 1, 16, and 256 increments; see `companion_all_knobs.csv`.

| Move applied to all six images | Float64 max vector error after 256 steps | Float32 max vector error after 256 steps |
|---|---:|---:|
| cx | 1.80783e-05% | 0.427333% |
| cy | 6.12935e-07% | 0.352508% |
| height | 9.76653e-06% | 0.0628789% |
| width | 7.44022e-07% | 0.685618% |
| lean | 0.000549953% | 0.249827% |
| joint | 0.000366284% | 0.239454% |
| lean10 | 0.000138519% | 0.504656% |

The float64 companion probe is 1e-5 of each knob range; the float32 probe is 1e-4. These probes differ, so this table is not a controlled precision-only comparison. The full probe sweep above compares equal probes.

A forward 10-degree shear followed by its reverse should recover the source geometry. Repeated pixel warps instead leave irreversible blur; see `shear_round_trips.csv`.

## Small steps do help a correct local approximation

A separate nearby smooth start `(14,14.013,19.73,3.2,0)` moves to lean 1 degree without crossing any renderer formula boundary. A first-order update integrates the current analytic rate of change of all five physical-coordinate vectors. It uses current local derivatives, not the future target vectors. Unlike pixel interpolation, its error decreases with smaller increments.

| Steps to 1 degree | Maximum raw-vector error |
|---:|---:|
| 1 | 0.0304586% |
| 4 | 0.00761582% |
| 16 | 0.00190403% |
| 64 | 0.000476012% |
| 256 | 0.000119003% |
| 512 | 5.95016e-05% |

## Metric parallel transport has a different goal

A smooth control starts at `[14.0, 14.013, 19.73, 3.2, 0.0]` and increases width by 0.6 pixels. All 101 audited points have the same renderer formula signature. Successive orthogonal frame maps approximate induced-metric parallel transport, which preserves lengths and angles, while coordinate knob vectors need not preserve them.
After 512 steps the largest raw-vector discrepancy is **13.9168%**. The height unit-direction discrepancy is **0.0433821**, and the true normalized frame Gram matrix changes by **0.0427948**. Thus this is a failure of same-knob preservation even without renderer kinks; it is not a failure of parallel transport itself.
On this same smooth path, the local quadratic-curvature rule has maximum error **4.95133e-16** over all tested step counts.

## Renderer events and general counterexample

The root caps lie on 1/16-pixel subrow boundaries. At zero lean the positive and negative derivatives agree in the limit. After lean changes, cy and height probes can have different left and right limits. Then there is no unique ordinary pixel-space Jacobian at that state, even though knob paths remain valid. These are renderer kinks, not evidence for extra independent knobs. See `kink_tests` in the JSON.
The smooth flat-plane counterexample uses the independent unit fields `X = d/dx` and `Y = (sin(x), cos(x))` for `|x| < pi/2`. Transport by the x-translation differential leaves Y unchanged, while the local Y field rotates. After moving x by 0.5, the vector error stays 0.494808 for 1–512 increments. Small steps cannot repair a mismatch between noncommuting fields.

## Scope and reproduction

Derivative probes can extend beyond the default sampling box (for example, a positive lean probe at 35 degrees); the renderer permits this. These are known-control diagnostics with deterministic paths and 64 held-out random states, not exhaustive validation of every point or discovery of knobs from unlabeled pixels. The theorem requires a regular differentiable chart. The raster experiments intentionally include one-sided conventions at events.
Analytic positive-derivative validation against forward renderer probes: maximum relative error **8.47781e-08** over 64 held-out states. Maximum inverse parameter round-trip absolute error **4.92939e-14**.
Run `.venv/bin/python analyze_generated_one_incremental_transport.py` from the repository root.
Files: `results.json`, `frame_control.csv`, `local_curvature.csv`, `pixel_shear.csv`, `companion_transport.csv`, and the two figures.

Mathematical background: [MIT notes on commuting coordinate fields and flows](https://math.mit.edu/classes/18.966/2014SP/965/class4.pdf), [metrics and connections](https://www.wim.uni-mannheim.de/media/Lehrstuehle/wim/schmidt/FSS2024/Riemannian_Geometry/Web/RGch3.html).
