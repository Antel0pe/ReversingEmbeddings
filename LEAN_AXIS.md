# An exact lean axis from a generated image

This note concerns **only** the five-knob family in `grey_ones.py`. Given one
28x28 image `x` from that family and an angle change `delta`, the equations below
produce the member with the same centre, height, width, and coverage rule, but
with its lean increased by `delta`. The input supplies all five coordinates;
there is no optimiser, training set, or search for a similar image.

All pixel indices start at zero. Write `[z]_+ = max(z, 0)` and let
`overlap([a,b],[c,d]) = [min(b,d) - max(a,c)]_+`.

## Read the stroke from its pixels

First sum each row:

```text
m_r = sum_c x_(r,c)                   w = max_r m_r
r_top = first row with m_r > 0        r_bottom = last row with m_r > 0
T = r_top + 1 - m_(r_top)/w            B = r_bottom + m_(r_bottom)/w
cy = (T+B)/2                          h = B-T
```

These equalities hold because each subrow's horizontal interval lies entirely
inside the canvas: its mass is its width `w`. The first and last rows contain
only a fraction of the vertical interval `[T,B]`. There are many fully covered
rows, each with `m_r = w`.

For a full row, consider cumulative ink at an integer column boundary:

```text
C_r(c) = sum_(j<c) x_(r,j)
c_r = argmin_(c in {0,...,28}) |C_r(c) - w/2|
z_r = c_r - C_r(c_r)
```

Let `u = tan(theta)` be the original lean slope and `L0 = cx - w/2`. In the
generator, the left edge at vertical position `y` is

```text
L(y) = L0 - u(y-cy).
```

At the selected boundary `c_r`, that boundary lies *inside every subrow's
stroke interval*. Consequently `C_r(c_r)` is exactly `c_r` minus the mean left
edge of the row. The subrow midpoints are symmetric about `r+1/2`, giving

```text
z_r = L0 - u(r+1/2-cy).
```

Two full rows determine the slope and left edge:

```text
u = (z_a-z_b)/(b-a)
L0 = z_a + u(a+1/2-cy)
cx = L0 + w/2                     theta = arctan(u).
```

`lean_axis.py` uses all full rows in a straight-line fit to reduce float32
rounding error. Why is the central boundary safe? Within a full row, the left
edge moves by at most `tan(35 degrees) * 15/16 < 0.657` pixels across its 16
subrows. Since `w >= 1.8`, the common interior of their intervals has width
greater than `1.143`. Its midpoint is more than `0.571` from either end, so
the nearest integer column boundary lies inside it. That is the boundary
closest to half of the row's cumulative ink.

## Change lean and compute all 784 coordinates

For a requested change `delta`, put `theta' = theta + delta` and
`u' = tan(theta')`. For each pixel row `r`, split its height into 16 subrows.
Their midpoint and vertical coverage fraction are

```text
y_(r,k) = r + (k+1/2)/16
omega_(r,k) = 16 * overlap([r+k/16, r+(k+1)/16], [T,B]),   k=0,...,15.
```

The new left and right edges at that midpoint are

```text
L_(r,k)' = L0 - u' * (y_(r,k)-cy)
R_(r,k)' = L_(r,k)' + w.
```

Finally, the output's 784 coordinates are

```text
F_delta(x)_(r,c) = (1/16) * sum_(k=0)^15 omega_(r,k)
    * overlap([c,c+1], [L_(r,k)',R_(r,k)']),   r,c=0,...,27.
```

This is one mathematical function of `x` and `delta`: every symbol on the
right is either supplied by the input or calculated above. The target angle
must stay in the family's declared interval of -10 to 35 degrees. The equation
is piecewise because interval overlap changes form when a stroke edge crosses
a pixel boundary.

## What the lean axis actually looks like

Use `u = tan(theta)` as its coordinate. Within any stretch where the same
stroke edges overlap the same pixels, each term of the output equation is
affine in `u`: the left and right edges are affine, and `omega` is fixed.
Consequently **the entire 784-dimensional lean axis is a polygonal line in
`u`**, with exact straight segments joined at corners. Degrees reparameterise
that same path nonlinearly; they do not round its corners.

All possible corner locations are known mathematically. An edge at subrow
midpoint `y_(r,k)` crosses integer pixel boundary `b` when

```text
u = (L0-b)/(y_(r,k)-cy)       or       u = (L0+w-b)/(y_(r,k)-cy),
```

provided the denominator is nonzero and `u` lies inside the allowed lean
range. Some crossings happen together; some change no active pixel and do not
make a visible turn. Between consecutive actual crossings, every one of the
784 coordinates is exactly linear in `u`. This is a property of the coverage
rule, not an inference from PCA or a finite sample of images.

## Accuracy and scope

`lean_axis.py` evaluates these equations separately from `grey_ones.render`.
Against that renderer, 5,000 random inputs with random target angles across
the entire declared interval had maximum L2 pixel error `3.2e-7` and maximum
single-pixel error `9.6e-8`. All 32 corners of the knob box were recovered.
One hundred complete paths, each sampled at 21 angles, had maximum L2 error
`2.9e-7`. The remaining discrepancy is float32 output rounding in the
generator.

This result establishes the lean axis for every image satisfying this exact
five-knob coverage equation and its declared bounds. It does not assert that
the same equation moves arbitrary MNIST images. The supplied image is checked
against the recovered coverage equation before it is moved.

`make_exact_lean_figure.py` draws one complete path and measures its length and
turns directly in the original 784-dimensional pixel space.
`make_lean_equation_explainer.py` shows the row sums, cumulative edge recovery,
line fit, pixel movement, and output comparison for a 20-to-30-degree example.
`make_lean_process_diagram.py` draws those operations as a numbered flow from
the original pixels to the new pixel coverages.

## What a fixed pixel vector does at a corner

For the example with centre `(14.5, 14.5)`, height `19.75`, and width `3`, the
axis turns by 90 degrees at zero lean. Define `x_a = F(a degrees)` and
`D = x_0 - x_(-2)`. Repeated vector addition gives

```text
z_n = x_(-2) + n D,          n = 0, 1, 2, ...
```

`z_0` is the generated -2-degree image and `z_1` is the generated 0-degree
image. `z_2` is **not** the +2-degree image: their Euclidean distance across
all 784 pixel values is `1.706`. Already at `z_2`, 42 pixel values fall outside
the valid coverage range `[0, 1]` (minimum `-0.314`, maximum `1.314`).
The +2-degree image changes different edge pixels. The previous vector keeps
changing the old set, so the repeated path passes straight through the corner
and leaves the family. Clipping invalid values can make a viewable picture, but
that displayed picture does not follow the coverage equation.

`make_fixed_vector_experiment.py` plots the images, red/blue pixel changes, and
the two paths projected onto their incoming and outgoing directions.
`make_long_fixed_vector.py` extends this comparison through 18 vector additions
(nominally +34 degrees) in a static figure and a 19-frame animation. At the
last step the raw fixed-vector image has 44 pixel values outside `[0, 1]`,
spans `-5.34` to `6.34`, and lies `24.40` in L2 pixel distance from the true
+34-degree image.

For a generated image `x`, `move_lean(x, 10)` returns the image 10 degrees more
leaned. `trace_lean(x)` returns all 181 images on a quarter-degree grid from
-10 to 35 degrees; the mathematical equation also accepts any angle between
those endpoints.
