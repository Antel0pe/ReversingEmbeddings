---
name: experiment-figures
description: Build figures that explain an idea experiment. Use whenever producing a plot, chart, image grid, animation or any visual whose job is to show what an experiment did and let a human judge whether it worked. Covers layout discipline (nothing overlapping), baseline-before-variation ordering, parameter sweeps wide enough to reveal the effect, and per-panel measurements so the figure can be evaluated rather than admired.
---

# Figures for idea experiments

These figures answer "what happened, and did it work?". A reader must be able to
reach their own verdict from the image alone. That is a higher bar than looking
good, and it fails in specific, repeatable ways.

## Non-negotiable: nothing overlaps

Overlapping text is the single most common defect and it makes a figure worthless.

- **Never rely on `tight_layout` to save a hand-placed figure.** It does not know
  about `inset_axes`, `fig.text`, or manually positioned axes. If any element is
  placed by hand, place *all* of them by hand.
- **Reserve bands.** Decide up front: title occupies y in [0.92, 1.0], content
  [0.10, 0.90], footnotes [0.0, 0.08]. Then keep content strictly inside its band.
  Do not let a 16-thumbnail ring drift into the title.
- **Correct for aspect ratio when placing things on a circle.** Figure coordinates
  are normalised per axis, so `(0.5 + R*cos t, 0.5 + R*sin t)` is an ellipse
  unless the figure is square. Use `RY = RX * fig_w / fig_h`.
- **Put labels below their panel, never beside it**, and give them room with an
  explicit `transform=ax.transAxes` offset (`-0.11` under a 28x28 image works).
  Row labels go to the left of column 0 with `left=` margin in `add_gridspec`
  wide enough to hold them — measure the longest string, do not guess.
- **Long captions: break them yourself.** Two 90-character lines beat one 180.
- **Then look at the rendered PNG.** Read it back and check every text element.
  Layout bugs are invisible in code and obvious in the image.

## Order: baseline first, then variations

A reader cannot judge a variation without knowing what it is varying from.

1. **The target or ground truth**, if one exists.
2. **The trivial or null version** — the thing you would do without the idea.
3. **The variations**, in monotone parameter order, never shuffled.
4. Mark rows 1 and 2 visually (heavier border, colour) so they read as reference.

Sweep *past* where the effect stops. If a parameter is interesting at 1x, show
0x, 1x, 2.5x and 10x. A sweep that stops at the expected answer cannot reveal
that the trend continues, and "it keeps going, monotonically" is often the finding.

## Every panel carries its number

An image grid without measurements is decoration. Each panel gets the quantity
that decides whether it worked, and the threshold is encoded in the figure:

- print the value under each panel
- colour it against the pass/fail threshold (green under, red over)
- draw the threshold explicitly — an `axhline`, a dashed ring, a coloured border
- state the threshold's meaning in the subtitle, in words, once

Include a **calibration row**: what the measure reads for real data. "Ghosting
0.24" is meaningless until you know a real example scores 0.34.

## One figure, one question

If a figure answers two questions it will answer neither. Splitting into three
clean figures is always better than one crowded one. Cost of an extra figure is
approximately zero.

## Make the visual metaphor match the encoding

If the text says "rope of constant length swung around a pole", the radius on the
polar plot **must be** the rope length. Encoding something else as radius while
using that metaphor guarantees the reader misreads it. Pick the metaphor after
the encoding, or change the encoding to fit.

## Sentinels and edge cases

- A sentinel meaning "never happened" must be **outside** the range of real
  values. Using 0.5 for "did not fail" when real failures land near 0.5 makes the
  column unreadable.
- Log axes: check for zeros first.
- Colour scales: fix `vmin`/`vmax` across panels that are meant to be compared,
  or the comparison is a lie.
- Diverging data (a direction, a difference) wants a diverging colormap centred
  at zero with symmetric limits: `vmin=-m, vmax=m` where `m = abs(x).max()`.

## For image data specifically

- A direction in pixel space **is an image**. Render it red/blue rather than
  reducing it — no projection needed and nothing is lost.
- A one-parameter family can be shown losslessly: coordinates down, parameter
  across, as a heatmap. Every dimension present, nothing thrown away.
- Animate the parameter when the motion is the point. A GIF carries one whole
  dimension for free, and motion is read far better than a static extra axis.

## State the claim at the resolution you measured it

A title is a claim, and it is the part most likely to be wrong. "Only two
directions work" was written from an 8-sample sweep; a 3-degree sweep showed two
*windows*, each about ±20 degrees wide, and the 15-degree bar chart in the same
figure visibly contradicted the title. Before writing a claim into a title, run
the sweep finer than the one you are plotting and check the claim survives. If it
does not, the finer result is usually the more interesting one anyway.

## A 2-D slice needs a control from the full space

Any circle, plane or sweep you choose to show is a slice, and a slice through a
high-dimensional space is chosen, not representative. Whenever a figure walks a
low-dimensional path through a high-dimensional space, put a small control panel
beside it: N samples drawn from the *full* space, scored the same way. A window
that looks 40 degrees wide on the chosen circle can still have essentially zero
measure on the real sphere, and only the control shows that. It costs one narrow
histogram and it is the difference between a picture and a result.

## Mark what the render is hiding

`imshow` clips. A value of -0.9 and a value of 0.0 draw the same black pixel, so
a figure can silently discard the most interesting thing in the data. When values
can leave the displayable range, overlay the out-of-range pixels in a flat colour
on top of the normal render and print the discarded mass under each panel. Here
that turned "the strokes look like they have bites out of them" into "the arc
leaves the unit cube," which was the actual finding.

## Always show the finished figure in the message

A figure that only exists as a path on disk has not been delivered. After
generating or regenerating one, send it with `SendUserFile` so it renders in the
conversation, and describe what it shows in the message alongside it. Never write
"wrote figures/foo.png" and leave it at that.

- Send the **final** version, after every fix. If a layout bug was found by
  reading the render back and then corrected, the corrected file is the one that
  goes in the message — do not send an intermediate and then describe a later fix
  in words.
- Send every figure the request produced, not a representative one. Splitting a
  question across three figures was a deliberate choice; showing one of them
  undoes it.
- Use `display: "render"` so it opens inline rather than as a download card.
- Do not re-send an unchanged figure in a later turn. Send it when it is new or
  when it has meaningfully changed.

```python
SendUserFile(files=["figures/bulge_bites.png", "figures/bulge_rotation.png"],
             caption="...", display="render", status="normal")
```

## Checklist before declaring a figure done

- [ ] Read back the rendered file and inspect it
- [ ] Every claim in a title survives a sweep finer than the one plotted
- [ ] Any low-dimensional slice is paired with a full-space control
- [ ] Anything the render clips or hides is marked, not silently dropped
- [ ] No text touches any other text or any panel
- [ ] Baseline rows present, first, and visually distinct
- [ ] Parameter swept past the interesting region
- [ ] Every panel labelled with its measurement and a pass/fail colour
- [ ] Threshold drawn and explained in words
- [ ] Calibration against real data included
- [ ] Title says what the figure shows; subtitle says how to read it
- [ ] The figure answers exactly one question
- [ ] The final rendered file is sent to the user in the message, not just named

## Worked examples in this repo

`make_bulge_figures.py` — baseline-first row ordering, per-frame distances
coloured against THETA, a parameter swept to 20x, and a polar plot whose radius
matches its own metaphor. `make_path_figures.py` — animation where the motion is
the finding. `make_bulge_followups.py` — explicit y-band layout that cannot
overlap, clipped-pixel overlays, and a full-space control beside a chosen
2-D circle.
