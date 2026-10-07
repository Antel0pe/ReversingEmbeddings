# Pixel changes at different starting settings

Start with `.venv/bin/python serve_generated_lean_pixel_overlay.py`, then
open `http://127.0.0.1:8765/generated_lean_pixel_overlay/index.html`.
Regenerate the page and initial figure with
`.venv/bin/python make_generated_lean_pixel_overlay.py`.

Choose the knob to change and a signed amount in the toolbar. Each layer's
five controls are its starting settings. Its colored pixels are
`render(start + selected_knob_step) - render(start)`. The small caption
in each layer shows the exact before and after values. Red gains ink;
blue loses ink. All colors use the same fixed +/-1 coverage-change scale.
The reference is a separate black context image and is not subtracted from
all endpoints. A height change is visible at lean zero. The old version
only subtracted lean and silently erased this case.

**Copy reference to layers** copies all five reference settings into the
three starting states, leaving the chosen change and amount intact.
**Space by** selects the only knob that Auto space is allowed to vary.
**Auto space** generates starting states from the reference and searches
for uniform positive, negative, or symmetric offsets on that knob's
slider resolution. It renders actual before/after images, checks every
pair of changed-pixel masks, and requires at least one unchanged pixel
between groups (8-neighbor dilation). Empty responses and strokes clipped
by the canvas are excluded. It chooses the smallest successful tested
spacing, preserving all other reference settings. On failure it reports
why and leaves the current controls untouched; it never fakes separation
or changes another knob to make the check pass.

The default uses the attached short-reference example: center (14.5, 14.5),
height 6.25, width 3.2, lean zero. The operation is Height +1 px. Auto space
finds starting heights 6.25, 11, and 15.75 (gap 4.75 px); the colored groups
have zero overlap and an unchanged pixel between them. `overlay.png` shows
this default. The older width/lean sample metrics remain in `step_metrics.csv`;
`results.json` distinguishes those fixed-slice checks from the current default.

Every live image comes from `grey_ones.render`. Images are float32 and
subtraction uses float64. No JavaScript renderer, interpolation, averaging,
normalization, pixel subdivisions, or display translations are used.
Differences below 1e-7 are omitted from colored marks and support counts.
The spacing guarantee uses that threshold and applies to the computed
settings and selected finite change, not future manual edits or arbitrary
continuous sweeps. Black reference ink can remain between change groups;
"clear" refers to the absence of colored changes, not necessarily white ink.

Manual changes can cause overlap; the live count appears below the image,
and higher-numbered visible layers overwrite earlier ones at shared pixels.
Show toggles isolate layers. Strokes reaching past the 28x28 canvas are
reported by name. The local server is required for arbitrary settings.
