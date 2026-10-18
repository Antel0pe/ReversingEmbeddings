# SVD knob families: two social figures

Two 1600 × 1200 PNGs, in posting order. The third card was dropped from this set and the download bundle.

1. **Shared patterns to families:** root-relative distance and angle curves for a lean sweep from −10 to 35 degrees at widths 2.4, 3.2, and 4.6 px. Every width uses its own zero-lean root; other settings are the canonical root. Distance is the full 784-pixel L2 displacement. Angle is the acute angle between that displacement and the local zero-lean derivative, ignoring sign; it is undefined at zero lean. These are illustrative known-control curves, not a claim that root-relative curves alone were the assignment objective. Below are actual SVD vectors at the root and original width-walk step 9, assigned by the original path-profile matcher. Names are evaluation labels. The bottom image pairs exaggerate each physical knob so the transformation is visible. Lean: −10→35°; width: 1.8→4.6; cx: 9→19; cy: 10→18; height: 12→25 px. These illustration settings can extend the default box; they are not claimed to be recovered paths.
2. **Generated versus MNIST families:** simplified to two starting images and three representative families per dataset. Generated shows lean, width, and height at root and original width-walk step 9. MNIST shows F1/F2/F3 at the root and farthest root-distance anchor in its graph. The other two families are omitted only for display; the original matcher still assigned all five jointly. Cosine labels were removed to keep attention on the maps and continuation results. The generated width walk was replayed with the original matcher and sampling seed; every state matched saved parameters. Physical labels enter only evaluation. MNIST source indices and all five original frame vectors remain saved in the arrays. The displayed 9/9 and 54.2% describe different tests: signed physical-family continuation for generated data and graph-triangle label closure for MNIST. Vector pattern differences alone do not establish a bad family; route inconsistency supplies the independent failure evidence.

All vector maps show the full 28×28 unit pixel-change arrays, with a common symmetric color scale in each card. Red gains ink and blue loses ink. Source images use white=0 and black=1. The larger example pairs move only the named physical parameter and remain inside the canvas.

The generated matcher compared whole profiles of corresponding-path distance, displacement mismatch, and displacement angle across offset branches. Physical control labels did not enter the assignment score. The renderer and Jacobian supplied valid sampling and probes. Raw MNIST cannot supply those probes and uses an explicitly labeled observed-neighbor adaptation. The omitted finite-cloud control remains documented in `figures/mnist_svd_families/README.md`.

Reproduce from the repository root:
```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/make_twitter_svd_families.py
```

`plotted_data.npz` retains all displayed frames and new curves; `measurements.json` records definitions and larger illustration settings. Figures were inspected with the repository understanding check before delivery. Shareable artifacts contain no machine-specific workspace paths.
