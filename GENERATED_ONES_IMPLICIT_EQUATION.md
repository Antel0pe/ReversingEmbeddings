# A plain-English validity equation for the generated 1s

This note is about the controlled 28-by-28 grey images made by
[grey_ones.py](grey_ones.py). It does not describe every handwritten MNIST 1.

Each input pixel says how much of its square is covered by the stroke: zero
means no coverage and one means full coverage. The equation takes all 784 pixel
values and returns one nonnegative number.

> **Validity score = knob-range penalty + pixel-coverage mismatch penalty.**

In exact arithmetic, the score is zero precisely for images made by the
five-knob generator within its allowed settings. For the generator's float32
images, use a small numerical tolerance instead of asking for literal zero.

## Why two penalties?

We can reject increasingly subtle invalid images by asking these questions:

| Question | What it catches | What can still slip through |
| --- | --- | --- |
| Is there one connected stroke? | Blank images and detached dots | Bent, tapered, forked, or wrongly shaded strokes |
| Does each row have the right amount of ink? | Wrong height or changing total width | Ink moved into the wrong columns without changing the row total |
| Does the stroke center follow a straight line? | Many bends | Two faint strokes balanced around the same centerline |
| Does every pixel have exactly the coverage this straight stroke requires? | The remaining wrong shapes and grey values | Only a generated 1, if its five settings are also in range |

This is a logical sequence of failures, not a measured ranking of how often
different invalid images occur.

The third question is still insufficient. Imagine averaging two generated
images that have the same center, height, and width, but lean by minus ten and
plus ten degrees. Their average keeps the same ink total in every row and has
a straight average centerline. Yet the grey ink spreads across the edges in
a way the generator does not produce. With center at 14.5 pixels in both
directions, height 19.75 pixels, and width 3 pixels, that average is about
3.69 units away from the generated upright image when all 784 pixel
differences are combined as an ordinary Euclidean distance.

That example is why the equation must eventually inspect exact pixel
coverage. Once it does, separate penalties for connectedness, row totals,
and straightness become redundant: a zero coverage mismatch already
guarantees them.

## First, read the five settings from the image

These measurements produce **candidate settings**. Measuring them does not
yet declare the input valid.

1. **Stroke width.** Add the 28 coverage values in each image row. For a
   generated image, a row fully inside the stroke has a total equal to the
   stroke width. Take the largest row total as the candidate width.

2. **Top edge and bottom edge.** Find the first and last rows containing ink.
   A partially filled end row tells us what fraction of that row the stroke
   occupies: divide its row total by the candidate width. The first row's
   lower boundary minus that fraction gives the continuous top edge. The
   last row's upper boundary plus its fraction gives the continuous bottom
   edge. The distance between those edges is the candidate height; their
   midpoint is the candidate vertical center.

3. **Lean.** Rows 10 and 18, counting the top image row as row 0, are fully
   inside every allowed generated stroke. In each, scan the boundaries
   between pixel columns and choose the boundary with closest to half the
   row's total ink on its left. At that boundary, subtract the ink on its
   left from the boundary's horizontal position. The result is the stroke's
   inferred left-edge position in that row. Compare the two left-edge
   positions, eight rows apart, to find how far the stroke leans sideways
   per row. This is the candidate lean slope. Its angle is the arctangent
   of that slope.

4. **Horizontal center.** Extend that straight left-edge line to the
   vertical center of the stroke, then add half the candidate width.

There is no search through possible five-knob settings. These are direct
measurements from sums of the input pixels. For a blank image, take the
candidate width as zero; the next part will reject it.

## Part one: the knob-range penalty

The allowed settings are:

| Setting | Allowed range |
| --- | --- |
| Horizontal center | 14 to 15 pixels |
| Vertical center | 14 to 15 pixels |
| Stroke height | 19 to 20.5 pixels |
| Stroke width | 1.8 to 4.6 pixels |
| Lean angle | minus 10 to plus 35 degrees |

For each candidate setting:

- Give it zero penalty if it is inside its range.
- Otherwise, measure how far it lies beyond the nearest end of the range
  and square that distance.

Add those five numbers. For example, a candidate width of 5 pixels exceeds
its maximum by 0.4 pixels, so width contributes 0.16 to the score. This part
rejects images whose inferred stroke is too wide, too tall, too tilted, or
too far from the allowed center.

The mathematical equation uses the tangent of the lean angle as its lean
coordinate, because that is the horizontal movement per unit of vertical
movement. The angle limits above are converted to that same coordinate
before comparing them. No new visual property is hidden in that conversion.

## Part two: the pixel-coverage mismatch penalty

The measured settings tell us what coverage each individual pixel *would*
have if the image were truly made by this generator. To determine that
required coverage:

1. Split each image row into 16 thin horizontal strips. This is the
   generator's actual grey-pixel rule.
2. For each strip, find how much lies between the measured top and bottom
   edges.
3. At the strip's midpoint, use the measured horizontal center, width,
   and lean to locate the left and right edges of the straight stroke.
4. For each pixel column, find how much of its one-pixel width overlaps
   that stroke interval.
5. Multiply horizontal overlap by vertical overlap and average across
   the 16 strips. That is the coverage required for that pixel.

Now subtract required coverage from actual coverage for **each of the 784
pixels**, square each difference, and add the squares. A pixel that should
have coverage 0.4 but actually has coverage 0.7 contributes 0.09.

This part catches a dot, a hole, a split stroke, a bend, tapering, and even
a visually subtle error in the grey values along an edge. It uses every
pixel; there is no retained-variance threshold or discarded part of the
image.

## Put the parts together

The final score adds the five range penalties to the 784 squared pixel
mismatches. Nothing can cancel out because each contribution is zero or
positive.

- If the input is generated by an allowed setting, the measurements recover
  that setting and every required pixel value matches. The score is zero.
- If the score is zero, all five measured settings are allowed and every
  input pixel matches the generator's coverage rule at those settings.
  Therefore the input is a member of this generated family.

This is a **membership score**, not a distance along the manifold. A score
of 0.2 does not mean that the image is 0.2 pixel-space units from the
nearest valid image; the knob-range and pixel penalties use different
scales. Only the zero-versus-positive result has the exact interpretation.

I evaluated the described read-off and coverage calculations on 400 random
generator settings and all 32 corners of the allowed five-knob box. The
largest discrepancy for one pixel was about 0.000000127; the largest
784-pixel Euclidean discrepancy was about 0.000000337. Those small
differences are consistent with the generator storing images as float32.

## What this does and does not achieve

This is a short **read-the-geometry, then check-every-pixel** equation. It
does not optimize over five unknown knobs or compare against stored example
images. Its two penalties are enough to define the zero set exactly; adding
separate penalties for disconnected ink or bends would help explain *why*
an image failed but would not change which images pass.

It does reuse the generator's particular 16-strip coverage rule. It is
therefore not an independently discovered pixel-only equation that avoids
the generator altogether. I also have not proved that no shorter equation
exists under a formal measure of complexity.
