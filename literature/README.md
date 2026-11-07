# Literature

One file per paper. Each file holds the paper's title, a link, and notes —
never the paper's text in bulk, and never a mirrored PDF.

`## What it says (my notes)` belongs to the reader. `## Claude's running
summary` is written by the `read-paper` skill, which also maintains this index.

Claims carry a marking: **[paper]** the paper states it (with a section
number), **[paper, one example]** shown once and not proved, **[mine]** an
inference or a connection to this repo, **[check]** plausible but unverified.

- [Efficient Pattern Recognition Using a New Transformation Distance](efficient-pattern-recognition-using-a-new-transformation-distance.md) — seven hand-written motions beat every neural net of 1992 and tied humans, by not learning.
- [The Multiscale Structure of Non-Differentiable Image Manifolds](multiscale-structure-of-non-differentiable-image-manifolds.md) — sharp-edged image families have no tangent planes at all; the fix is a ladder of blur scales, and it recovers five knobs from one picture.

## Reading queue

Candidates for the standing question: recover the map from a few knobs to
high-dimensional data, in as few coordinates as possible, by construction or
closed form rather than by training a network. Not yet read. Entries marked
**[unverified]** come from memory, not from a search in this session.

### Closest to the problem

1. **Donoho & Grimes, "Image manifolds which are isometric to Euclidean space"**,
   J. Math Imaging Vis 23, 2005 · doi:10.1007/s10851-005-4965-4 ·
   https://www.math.ucdavis.edu/~saito/data/high-dimensions/WhenIsometry.pdf
   The Wakin paper's ref 11, where the 1/2 exponent is actually proved. Asks
   exactly our question — when are the underlying parameters recoverable — and
   answers: isometric to a convex Euclidean set for translations of a symmetric
   object, rotations, horizon articulations and cartoon-face expressions;
   breaks when parts move independently.
2. **Simard, LeCun, Denker & Victorri, "Transformation Invariance in Pattern
   Recognition: Tangent Distance and Tangent Propagation"**, Neural Networks:
   Tricks of the Trade, LNCS 1524, 1998 · doi:10.1007/3-540-49430-8_13 ·
   https://hal-ens.archives-ouvertes.fr/file/index/docid/60948/filename/Tangent_distance.pdf
   Hand-built tangent vectors for handwritten digits: both translations,
   rotation, scaling, two hyperbolic deformations, and **thickness**. That is
   `figures/generated_one_gradient_fields/`'s six affine fields plus
   `warp_charts.py`'s gain field, written down in 1998, on MNIST.
3. **Hastie, Simard & Säckinger**, NIPS 1994 — learns the tangent vectors from
   the digits instead of deriving them by hand. The bridge from "I know the
   five motions" to "find the motions in MNIST".

### A direction as a rule (the motion-field idea, formalised)

4. **Rao & Ruderman, "Learning Lie Groups for Invariant Visual Perception"**,
   NIPS 1998 ·
   https://papers.neurips.cc/paper/1998/hash/277281aada22045c03945dcb2ca6f2ec-Abstract.html
   Matrix-exponential generative model; learns the transformation's
   **generators** from image pairs. The generator is the motion field v.
   First-order Taylor, so small transformations only.
5. **Miao & Rao, "Learning the Lie Groups of Visual Invariance"**, Neural
   Computation, 2007 · https://homes.cs.washington.edu/~rao/liegroups-07.pdf
   EM from image pairs, up to six affine types: two translations, rotation,
   scaling, two shears. The same six.
6. **Sohl-Dickstein, Wang & Olshausen, "An unsupervised algorithm for learning
   Lie group transformations"**, 2010 · https://arxiv.org/abs/1001.1027
   Handles multiple transformations and **large** inter-frame differences,
   which is the limitation that matters when MNIST's nearest neighbour is
   24.8% of an image away.
7. **"Unsupervised Transformation Learning via Convex Relaxations"**,
   https://arxiv.org/abs/1711.02226 — modern treatment and a survey of 4-6.

### Alignment without labels

8. **Learned-Miller, "Data driven image models through continuous joint
   alignment"**, IEEE PAMI, 2006 ·
   https://people.cs.umass.edu/~elm/papers/PAMI_congeal.pdf
   Congealing: jointly warp a pile of unlabelled images to minimise pixel-stack
   entropy, yielding a factorised model that separates shape from nuisance
   warp. Needs moderate initial alignment to converge (noted by BMVC 2011).

### Closed-form coordinates from fixed measurements

9. **Suk & Flusser, affine normalisation by moments** (UTIA; and their book
   *Moments and Moment Invariants in Pattern Recognition*). Decompose the
   affine transform into elementary steps and read each step's parameters off
   low-order moments. The general form of
   `figures/generated_one_moment_chart/`. Search did not surface an explicit
   closed-form parameter formula — check the PDFs.

### Charts, and a global coordinate system from local pieces

10. **Brand, "Charting a manifold"**, NIPS 2002, and US patent 6947042, which
    describes it as a closed-form kernel mixture of linear projections **with
    an inverse mapping** — i.e. a non-neural decoder.
11. **Roweis, Saul & Hinton, "Global coordination of local linear models"**,
    NIPS 2001. **[unverified]**
12. **Donoho & Grimes, "Hessian Eigenmaps"**, PNAS 100, 2003 — the non-convex
    case that defeats Isomap.

### Foundations, if the motion view holds up

13. **Freeman & Adelson, "The design and use of steerable filters"**, PAMI
    1991. A direction as a rule: the response at any orientation is a fixed
    combination of a few basis filters. **[unverified]**
14. Grenander's pattern theory; Miller & Younes, "Group actions,
    homeomorphisms and matching"; Trouvé & Younes on LDDMM — the
    image-as-a-warp-of-a-template tradition. **[unverified]**
15. **Wakin, "High-resolution navigation on non-differentiable image
    manifolds"** (ICASSP) ·
    https://people.mines.edu/mwakin/wp-content/uploads/sites/262/2024/06/wakin-icassp-iams.pdf

### The local-linear route: a Jacobian at every point, then glue (searched 2026-11-05)

The question "treat the map as locally linear, then recover the map" splits into
five distinct tactics in the literature. Listed by tactic, not by date.

**Tactic 1 — fit a flat patch at each point, then stitch the patches.**

16. **Zhang & Zha, "Principal manifolds and nonlinear dimensionality reduction
    via local tangent space alignment" (LTSA)**, SIAM J. Sci. Comput. 26, 2004 ·
    https://math.ucdavis.edu/~saito/data/high-dimensions/LTSA_SISC7.pdf ·
    preprint https://arxiv.org/abs/cs/0212008
    Fit an affine subspace to each neighbourhood, then solve one eigenproblem
    that makes all the local coordinate patches agree. Ye, Zha & Li later
    proved the alignment matrix's null space gives the global coordinates up to
    an affine map, fixable to a rigid motion. **[paper]** Gives coordinates,
    not a forward map: no decoder.
17. **Brand, "Charting a Manifold"**, NIPS 2002 / MERL TR2003-13 ·
    https://papers.nips.cc/paper/2165-charting-a-manifold ·
    https://merl.com/publications/TR2003-13
    Already item 10 above; the local-linear reading is the point of it. A
    Gaussian mixture whose covariances are overlapping locally linear patches,
    then a kernel-weighted blend of the patches' linear projections. Convex
    objectives, closed-form solutions, and **pseudo-invertible** — one of the
    few in this family with a reverse map. **[paper]**

**Tactic 2 — learn the Jacobian as a function of position, then walk it.**
Closest to this repo's standing question.

18. **Dollár, Rabaud & Belongie, "Learning to Traverse Image Manifolds"**,
    NIPS 2006 · https://papers.nips.cc/paper/3035-learning-to-traverse-image-manifolds
    and **"Non-Isometric Manifold Learning: Analysis and an Algorithm"**,
    ICML 2007 · doi:10.1145/1273496.1273527 ·
    https://vision.ucsd.edu/sites/default/files/publications/pdfs/Non-Isometric%20Manifold%20Learning.pdf
    LSML. Learns a map from a point to its tangent basis — a Jacobian field —
    and then generates a point's neighbours by stepping along it. States the
    goal as a representation that handles NEW points rather than an embedding
    of the training set, and scores itself with a held-out test error.
    Explicitly aimed at non-isometric manifolds. **[paper]**
19. **Bengio & Monperrus, "Non-Local Manifold Tangent Learning"**, NIPS 2004 ·
    https://papers.neurips.cc/paper_files/paper/2004/file/0b7e926154c1274e8b602ff0d7c133d7-Paper.pdf ·
    journal version with Larochelle, "Nonlocal estimation of manifold
    structure", Neural Computation 2006.
    A tangent-plane predictor trained to make predicted tangent vectors line up
    with neighbour displacement vectors (criterion: minimise the sine of the
    projection angle). Argues purely local tangent estimates generalise badly.
    Headline: a learned notion of rotation transferred from digits to letters,
    where a local method fails. **[paper, one example]**

**Tactic 3 — hold the Jacobian fixed and call it a group generator.**
Items 4-7 above (Rao & Ruderman, Miao & Rao, Sohl-Dickstein). Same object,
assumed not to vary with position.

**Tactic 4 — use the Jacobian as a ruler rather than a basis.**

20. **Arvanitidis, Hansen & Hauberg, "Latent Space Oddity: on the Curvature of
    Deep Generative Models"**, ICLR 2018 · https://arxiv.org/abs/1710.11379 ·
    with Shao, Kumar & Fletcher (2018) and **"Pulling back information
    geometry"**, AISTATS 2022 · https://proceedings.mlr.press/v151/arvanitidis22b.html
    The decoder's Jacobian J pulls a metric JᵀJ back onto the knob space;
    distances and interpolations are then measured along geodesics of that
    metric rather than straight lines. This is exactly `knob_frames.metric`
    and `figures/generated_one_knob_atlas/`, with a neural decoder in place of
    `render`. **[paper]** Mine for the geometry, not the training.

**Tactic 5 — glue local linear models in the system-identification tradition.**

21. **Nelles & Isermann, LOLIMOT (local linear model tree)**, 1996 ·
    doi:10.1007/3-540-61510-5_23. Partition the input space with a tree, fit a
    linear model per cell, blend with overlapping basis functions. Equivalent
    to a Takagi-Sugeno fuzzy system. Stated drawbacks: axis-orthogonal cuts
    only, and interpolation error between local fits. **[paper]** This is the
    engineering version of `figures/generated_one_line_interpolation/`'s patch
    tiling, and the stated drawback is the ~1.3e8-patch cost found there.
    **[mine]**

### The two things this branch says we cannot have

22. **Hyvärinen & Pajunen, "Nonlinear independent component analysis:
    Existence and uniqueness results"**, Neural Networks 12, 1999 · reviewed in
    Hyvärinen, Khemakhem & Monti, https://arxiv.org/abs/2302.02672
    The no-go. From the data distribution alone the generating nonlinear map is
    **not identifiable**: infinitely many maps produce independent components
    that are still mixtures of the true sources (the Darmois 1952
    construction). Identifiability returns only with added structure — time,
    an auxiliary variable, or sparsity of the mixing map. **[paper]**
    This is the formal reason `figures/mnist_one_knob_search/` needed five
    labelled examples to pin a five-number gauge: unsupervised recovery of
    the generator's own knobs is provably impossible without such an anchor.
    **[mine]**  The proof is CONSTRUCTIVE (Darmois 1952 / Rosenblatt 1952:
    the conditional CDFs), and it has been run on this repo's own family in
    `analyze_darmois_alternative_knobs.py` /
    `figures/darmois_alternative_knobs/README.md`: an explicit second set of
    five knobs, pixel-identical images, independence indistinguishable from
    the true knobs, additivity unable to separate them, decoder simplicity
    able to.
23. **Frobenius / involutive distributions** — the named concept for
    `check_generated_one_direction_commutators.py`. A field of candidate
    tangent planes can be straightened into coordinates only if the Lie
    bracket of any two fields in it stays inside the plane (involutivity); a
    commuting frame then integrates into a coordinate grid.
    https://en.wikipedia.org/wiki/Frobenius_theorem_(differential_topology) ·
    https://maths-people.anu.edu.au/~andrews/DG/DG_chap7.pdf
    **Field note:** a search found no manifold-learning paper that tests
    involutivity on estimated tangent fields, approximately or otherwise. Items
    16-19 all estimate tangent planes and none check whether they integrate.
    That gap is where this repo's measured nonzero brackets live. **[check]**

### Field note

A search for a closed-form, non-neural chart decoder from 2023-2025 returned
almost only **neural** chart autoencoders (e.g. arXiv 2208.10570, 2510.13622).
The non-training route to this problem lives in the 1991-2007 literature, not
in current work.
