# What the five-coordinate autoencoder learned

This is a small diagnostic on the controlled five-knob generated-1 renderer. The autoencoder sees only images. Its weights are saved at epochs 80, 160, and 240 so the same model can be inspected later.

## Main result

On 512 unseen images, pixel variance reconstructed rises from 95.55% at epoch 80 to 97.14% at 160 and 97.69% at 240. That score does not mean the latent coordinates are the five generator knobs, nor that every knob's local effect is equally accurate.

The strongest interpretable coordinate is lean: at epoch 240, one raw latent coordinate has correlation 0.97 with lean. Width is shared across at least two coordinates (the strongest raw correlation is 0.60). The remaining coordinates mix position and other changes; no coordinate cleanly names `cx`, `cy`, or `height`.

To test whether a knob might be present in a nonlinear combination of coordinates, I fitted small readouts after the autoencoder was trained. The readouts saw training knob labels; those labels never affected autoencoder training. R² below is on the held-out 512 images, with each knob normalized to [0, 1].

| Knob | Linear readout | Quadratic readout | Extra trees readout |
|---|---:|---:|---:|
| `cx` | 0.747 | 0.949 | 0.902 |
| `cy` | 0.127 | 0.328 | 0.534 |
| `height` | 0.035 | 0.680 | 0.309 |
| `width` | 0.930 | 0.994 | 0.988 |
| `lean` | 0.963 | 0.995 | 0.999 |

This says lean and width are readily recoverable from the code, and `cx` is recoverable with a nonlinear readout. `cy` and height are less stable: a quadratic readout helps height substantially, but neither readout family recovers both vertical knobs cleanly. These are diagnostic probes, not a search over all possible inverse functions.

The local image changes tell the same story. At epoch 240, for small isolated knob moves, median cosine alignment between the renderer's pixel change and the autoencoder's pixel change is 0.92 (`cx`), 0.86 (`cy`), 0.53 (`height`), 0.92 (`width`), and 0.96 (`lean`). Height changes are also only about 0.55 of the renderer's magnitude. Thus the autoencoder's high overall score hides a meaningful local weakness for height.

## Function represented by the network

Let `x` be the 266 image pixels that vary in this renderer. The implementation standardizes those pixels with training mean `μ` and one shared scale `s`. The encoder is exactly

```text
h = ReLU(((x - μ) / s) W₀ + b₀)
z = h W₁ + b₁                 # z has five numbers
```

The decoder is

```text
r = ReLU(z W₂ + b₂)
x̂ = s (r W₃ + b₃) + μ
```

The learned function is therefore a piecewise-linear map from pixels to five coordinates and back. The ReLU gates make the map's slopes depend on the image. The autoencoder loss only asks the round trip `x → z → x̂` to preserve pixel values; it does not ask any `z` coordinate to mean a particular knob. The actual matrices and offsets are in `checkpoint_80.npz`, `checkpoint_160.npz`, and `checkpoint_240.npz`.

## Reading the figures

- `latent_meaning_summary.png`: held-out reconstruction, raw coordinate-to-knob correlations, and local change alignment over 96 interior settings.
- `one_seed_knob_pixel_changes.png`: for one held-out generated 1, compare the true renderer's signed pixel change against the autoencoder's change when each knob moves by ±4% of its full range. Red means more ink; blue means less. Each knob column has its own color scale, shared between its two rows.
- `results.json`: all reported metrics and method details.

This is one initialization and one generated-1 sample distribution. The conclusions describe this short experiment, not all five-coordinate autoencoders.
