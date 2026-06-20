# Normal/Full consistency + accuracy pass

Branch: `improve-normal-full-consistency`. Goal: make the full-photo
(`locate:true`) pipeline more consistent with the already-cropped
(`locate:false`) pipeline, and raise accuracy generally — without
hard-coding to the test images.

## Diagnosis (what the divergence actually is)

Investigated `20260313_213443` (the largest normal↔full gap, 98/14336 diffs):

- The 98 diffs are spread across the **whole** image and dominated by the
  two closest palette pairs (LG↔DG, WH→LG) — a global classification
  sensitivity, not a localized warp/locate failure.
- The warp **geometry** is essentially identical between the two paths
  (warp re-detects the frame corners and re-warps to the canonical
  1280×1152, normalizing away locate's framing). Camera-region means match.
- The real difference is **per-pixel content**: the already-cropped sample
  (`sample-pictures/…jpg`, 1340 px) is a *recompressed, rescaled* JPEG —
  a genuinely different lossy source than the 4032 px original. The two
  warps therefore differ ~10/px even after geometric alignment, and the
  `correct` step's affine gain (which necessarily stretches the ~50-unit
  washed-out range back to full range) amplifies that to ~17/px. This is an
  **irreducible input floor**, not a pipeline bug.
- `locate`'s final extraction is already near-1:1 with `INTER_LINEAR`;
  `INTER_CUBIC` was tested and is **worse** (rings at the high-contrast LCD
  pixel edges → full tier-1 34→53). Locate interpolation is already optimal.

### Dominant error across all tier-1 images

Aggregating output-vs-reference transitions, **LG→DG dominates** (we output
LG where truth is DG): 21/26 normal errors, concentrated in prison + bathhouse.
These are DG pixels whose sampled **R** is lifted into the sparse density
valley between the DG (R≈142) and LG (R≈232) modes, landing closer to the LG
centroid. Root cause: front-light **inter-pixel light bleed**, which AGENTS.md
notes is "especially vertical" — in dithered LG/DG regions a DG pixel
sandwiched between brighter warm rows has its R pulled up.

## Approaches tried

| Approach | Result |
|---|---|
| `locate` `INTER_LINEAR`→`INTER_CUBIC` | worse (rings), reverted |
| Global R-axis valley boundary (snap DG/LG split to density valley) | trades errors badly (a blunt 1-D override discards the 2-D + per-region refinements), reverted |
| Isolated-pixel spatial denoise | **catastrophic** (fix 4 / break ~20k) — the reference images are *dithered*, so isolated single pixels are real content, not noise. Rules out all spatial smoothing. |
| Undirected vertical R unsharp | helps detail images but explodes prison (lifts DG-between-BK into LG), reverted |
| **Directed vertical R de-bleed** (shipped) | net win, see below |

## Shipped change — directed vertical R de-bleed (`sample.ts`)

Removes vertically-bled light from genuine bleed victims before
classification. Three constraints, each verified to matter:

1. **Directed (local-minimum only)** — correct R only where the pixel is
   dimmer than *both* vertical neighbours (a real bleed victim). Local
   maxima (DG-between-BK) are untouched, preserving flat/dark regions.
2. **B-gate** — only correct pixels *bluer* than their vertical neighbours.
   DG carries high B; a bleed-lifted DG stays a vertical B-maximum while a
   genuinely-dim warm feature does not. This is what lets bathhouse/park
   improve while prison's sharp features stay protected.
3. **Magnitude floor** — ignore sub-`MARGIN` neighbour gaps (flat-region
   noise).

Parameters (`DEBLEED_R=0.25`, `DEBLEED_BGATE=10`) are physical-bleed
constants, env-overridable for re-tuning.

### Results (diffs vs reference; lower better)

| corpus | baseline | shipped |
|---|---|---|
| tier-1 normal (`locate:false`) | 26 | **24** |
| tier-1 full (`locate:true`) | 34 | **30** |
| consistency `213443` (full vs normal) | 98 | **91** |
| consistency `165926` | 7 | **2** |

Marquee wins: **park full 8→3**, bathhouse normal 7→4 / full 4→3.
Cost: **thing-1 normal 0→4** (a *second-order* effect — de-bleed lowers
genuine DG pixels' R, nudging the k-means DG centre, which reclassifies a
few borderline LG pixels; not fixable per-pixel since it's a distribution
shift), plus prison full +1 and thing-2 full +1.

Net −6 combined tier-1 errors with **both** paths and consistency improved.

`pnpm test` note: `pipeline.test.ts` asserts `toBe(0)` (100%) and was already
red for 5/6 of those images on `main`; this change keeps the count at 5
(thing-1 now fails where zelda-2 now passes).

## Shipped change 2 — warp refinement divergence guard (`warp.ts`)

Some full-photo warps placed the **left inner border ~1 GB pixel too far
right** (user-measured ~−7 px corner offsets; a doubled/folded left border).
Traced to the two perspective-refinement passes: on a biased edge (a blurry
or doubled border — e.g. the dim leftmost B sub-pixel column muddying the
WH→LCD transition) the correction feeds back *positively* and that edge's
offset GROWS pass-over-pass (left edge 3.8 → 7.9 px) instead of converging.

Crucial enabling fact: each refinement pass rebuilds the transform by
back-projecting onto the **original** photo, so every pass output is an
independent single resample (no accumulated blur). So the fix just keeps
whichever pass is best-aligned: measure each pass's max per-edge border
offset and default to the fully-refined warp2, falling back to an earlier
pass only when it is better by a margin (1.5 px). Convergent images (every
reliable photo) always have warp2 best → **unchanged**.

- RMSE-gating was tried first and rejected: park-1's left edge has the same
  high border-fit RMSE (≈6) as the broken private images while its
  correction is *needed and correct*, so reliability-gating breaks park
  (1→900). Divergence (does the edge get worse across passes?) is the
  correct discriminator; RMSE (scatter) is not.

Results: tier-1 normal/full and self-consistency **completely unchanged**
(guard never fires on convergent images); private left-border residuals
`184719` 7.7→3.6 px, `184739` 7.9→2.3 px (dashes back at the outer edge,
doubling gone); `184650` (which converged) untouched.

## Shipped change 3 — local adaptive WH/LG threshold (`quantize.ts`)

On `184650` a whole interior region of WH/LG **dither** flattened to solid
LG (WH dots lost). Cause: the front-light gradient leaves WH spatially
varying — that region's WH is dimmed to G≈210 while bright WH elsewhere is
G≈250. The histogram of high-R G is then *trimodal* (LG≈130, dim-WH≈210,
bright-WH≈250) and the single global G-valley lands at 231 (just under the
bright-WH spike), demoting every dim-WH dot to LG.

Fix: a windowed (radius 6) per-pixel WH/LG threshold taken from the local
warm-pixel G valley (same robust two-mode + deep-dip test as the per-column
step). Gated to fire ONLY where the local would-be-WH mode sits at/below the
global threshold — the "spatially-dimmed WH" signature — so well-exposed
regions (where the global already classifies the local bright mode as WH)
are skipped. Tier-1 normal/full and self-consistency are **byte-identical**
to before; `184650` recovers 354 dither pixels (checkerboard restored,
confirmed visually). Env-overridable (`LOCALWH_RADIUS`).

## Remaining floor / ideas not pursued

- prison's residual LG→DG blob is sparse, top-edge, surrounded by WH/BK
  (no DG neighbours) — global thresholds and local votes both trade errors;
  a calibrated 2-D PSF deconvolution is the real (high-risk) lever.
- The private blotchy images (`184650/184719/184739`) were inspected: their
  **warp is clean** (crisp aligned dashes — the "imprecise corners"
  hypothesis did not hold); the blotches are inherent DG/LG quantization of
  smooth-gradient content, not a detectable bug.
