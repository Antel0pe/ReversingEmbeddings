# Reshaping the 1s before the decoder learns them

`pre_transforms.py` (each reshaping and its undo), `analyze_mnist_one_pretransform.py`
(`python analyze_mnist_one_pretransform.py <pose|deslant|upsample|sdf|blur> <mnist|generated> [k,k,..]`),
`make_mnist_one_pretransform_figure.py` -> `pretransform_summary.png`.

Same plain autoencoder and split as `../README.md`. Each decoded image is undone and scored against
the ORIGINAL held-out image; numbers set aside by a reshaping count toward the total.

| Reshaping | Round trip alone | Result (MNIST, error at matched total numbers) |
|---|---:|---|
| none | 0 | 24.2% @5, 18.9% @10, 17.9% @12, 16.0% @20, 15.0% @30 |
| straighten + centre left-right (2 set aside) | 3.1% | 24.7% @5, 17.7% @10, 16.0% @14, 13.7% @22 |
| straighten + centre + resize (5 set aside) | 3.4% | 34.5% @5, 18.6% @10, 16.5% @13, 15.1% @17, 13.5% @25 |
| check: same 56x56 grid, nothing removed | 3.9% | 23.9% @5, 18.0% @12, 15.6% @20 |
| distance-to-edge map | 20.6% | 28.5% @5, 24.9% @12, 24.5% @20 (stalls above its round-trip loss) |
| blur, sigma 1 px | 33.8% | 37.1% @5, 35.1% @20 |

- Straightening is worth ~1-2 points from ~7 numbers up; the finer-grid check gets only 0-0.4 of that.
  The curve still never stops. Straightened MNIST 1s still spread 29.0% around their average
  (generated 1s: 8.8%), so most of the variety is stroke shape, not pose.
- Resizing and vertical centring cost 3 numbers for ~nothing: height and up-down barely move a long stroke.
- Distance map: MNIST edges are grey (38% of ink); a hard inside/outside edge discards them.
- Blur: a 1 is ~2-4 px wide, so 1 px of blur alone loses a third of the image.
- Generated control: straighten+centre+resize plateaus at 6.5% vs 4.4% without, because resampling a
  sharp 28x28 picture smears edges, and the network already found the five knobs by itself.
