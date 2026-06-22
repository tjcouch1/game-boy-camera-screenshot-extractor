# Warp Accuracy Pass — Implementation Plan

**Starting commit:** `ec1df75bd81e9102e2733889813f89d534697c86`
(branch `retry-improve-warp`)

**Starting accuracy** (`pnpm test:pipeline`):

| Test            | Accuracy |
| --------------- | -------- |
| bathhouse-1     | 99.77%   |
| park-1          | 96.36%   |
| thing-1         | 99.87%   |
| thing-2         | 99.70%   |
| thing-3         | 99.92%   |
| zelda-poster-1  | 99.97%   |
| zelda-poster-2  | 99.86%   |
| zelda-poster-3  | 99.90%   |
| **Average**     | **99.42%** |

**Goal:** Push every test to ≥ 99.99 % by fixing the warp step's
residual curvature/bias, primarily on the right side and around the
bottom-left corner of `park-1`.

---

## User feedback (verbatim, condensed)

The user observed in `test-output/park-1/debug/park-1_warp.png` that:

* **Right border** — top-right corner is 1–1.5 px too far *left* (needs to
  move right). Around Y 238 the border curves *outward*; it is roughly
  correct near Y 456; then curves back *inward* and the bottom-right
  corner is 3–4 px too far left.
* **Left border** — top-left is roughly correct (maybe 0.5 px too far
  right). Starting around Y 295 the border slopes right; bottom-left is
  ~3 px too far right.
* **Top border** — top-left height is right; around X 860 it starts to
  curve downward and the top-right is 1–2 px too far down.
* **Bottom border** — bottom-left is 0.5–1 px too high; around X 448 it is
  correct; around X 889 it starts curving up and the bottom-right is
  2–2.5 px too high.

Errors cluster near the bottom-left corner and right side. The pipeline's
own corner-error metric reports the *opposite* sign on the right side
(`pass2.cornerErrors.BR = (+2.4, +1.0)`), so the **detection itself is
biased** on the right/bottom, not just the perspective transform.

### Physical model that the user provided

1. **LCD TN sub-pixel arrangement is B G R left-to-right.** Therefore the
   colour transition that the algorithm searches for is *not* symmetric
   between left/right and top/bottom:
   * **Left border (DG inside, WH frame outside-left):** sub-pixels look
     like `_GRB__` — the WH→DG transition is clean.
   * **Right border (DG inside, WH frame outside-right):** sub-pixels look
     like `B___GR` — there is a thick dark gap between the DG border and
     the WH frame.
2. **Blur is uneven** across each photo, generally worsening from
   top-left to bottom-right (and more along Y than X). Blur can be
   measured by the contrast inside WH-frame columns (dark `B` vs bright
   `GR`); blurry areas wash that contrast out.
3. **WH frame columns are vertical stripes** (`_GR` repeating). Their
   straightness and alignment encode lens curvature and per-row
   horizontal alignment. They are a free ruler we are not yet using.

The user explicitly asks:

* Do **not** fine-tune magic constants against one test.
* Bigger rewrites are fine; combine multiple improvements at once.
* Add diagnostics as needed to measure these things.
* Commit at every milestone; track average accuracy in commit titles.
* Keep going until 99.99 % or all ideas exhausted.

---

## Architecture summary

The current pipeline runs `warp → correct → crop → sample → quantize`.
The warp step itself runs:

1. Detect the outer white-filmstrip quad (`findScreenCorners`).
2. Apply an initial perspective warp.
3. Two refinement passes: detect inner-border corners + 9-point edge
   samples on an R-B-shifted channel, fit a small average curvature
   correction, back-project, re-warp.

The two refinement passes use **only a perspective transform**
(`getPerspectiveTransform`). A perspective transform has 8 DoF and
*cannot* remove residual lens curvature. The current "curvature
correction" only nudges the four corner positions by 0.45 × the mean
edge offset, which cannot fix a curve that changes direction along the
edge (which the user described for park-1's right and bottom edges).

The plan attacks the problem in three layers — diagnostics first so we
can see what is going on, then asymmetric detection so the measurements
are trustworthy, then a non-linear final pass so we can correct the
residual curvature.

---

## Phase 0 — Capture state (Task 1)

* Save this plan file.
* Note the starting accuracy in the plan and in commit messages going
  forward.

## Phase 1 — Diagnostics (Task 2)

We cannot fix what we cannot see. Add to `warp.ts`:

* **`warp_b_borders.png`** — full warped image with:
  * Detected border points (dense — see Phase 2) drawn as small red
    dots.
  * Detected corner positions drawn as green crosses.
  * Expected-rectangle border lines drawn in cyan, 1 px.
  * The current (averaged) detected border line drawn in magenta, 1 px.
* **`warp_c_edges.png`** — four 256-px wide horizontal/vertical strips:
  the actual pixel content along each edge cropped to ±srch around the
  expected border, scaled 4×, with detected border position drawn as a
  red line. Lets us visually verify detection on every edge.
* **Per-edge polynomial residual** in metrics: the maximum and signed
  RMS deviation of detected points from the fitted polynomial — a low
  number means our model captures the real edge well.

## Phase 2 — Densified, polynomial-modelled detection (Task 3)

Change `findBorderPoints`:

* Sample 25 points per edge instead of 9. Use the *interior* range
  `[INNER_LEFT + 2, INNER_RIGHT - 2]` (resp. top/bottom) so the corners
  are excluded — they're noisy.
* For each point: same R-B channel + `firstDarkFromFrame`, but reuse the
  improved profile from Phase 3 (right/bottom flipped & widened search,
  see below).

Add `fitBorderCurve(points, axis)`:

* Fit a 2nd-degree polynomial (`y = a + b·x + c·x²` for top/bottom,
  `x = a + b·y + c·y²` for left/right) via least squares.
* Discard outliers > 3·σ and re-fit.
* Return both the polynomial and the residual statistics.

Use this polynomial fit:

* As input to the **curvature-corrected back-projection** in
  `refineWarpWithMetrics` (replace the averaged `edge_curvatures`
  scalar with values sampled at the corners themselves so each corner
  is corrected independently).
* As input to the **non-linear final pass** (Phase 4).

## Phase 3 — Asymmetric sub-pixel-aware detection (Task 4)

Goal: remove the systematic bias the user describes (right side
detection thinks the border is further right than it really is, so the
algorithm pulls the right edge inward).

Concrete changes inside `findBorderPoints` and `findBorderCorners`:

* Replace the single fixed `smoothSigma=1.5` in `firstDarkFromFrame`
  with `smoothSigma = scale * 0.35` (≈ 2.8 at scale 8). One scale's
  worth of smoothing kills sub-pixel ringing without blurring the real
  transition. This is symmetric for now but tied to scale.
* For **right** edges add an explicit offset: the WH→DG transition the
  algorithm finds is the **leading edge of the dark `B___` gap**, not
  the centre of the DG border. Determine the empirical offset
  by computing both the WH→DG drop *and* the `DG→content` rise inside
  the same profile and reporting the gap width to metrics. Use the
  mid-point of the two transitions as the actual DG-strip centre, then
  subtract `0.5·scale` to land on the outer DG edge.
* Apply the symmetric change to the **bottom** edge (same
  arrangement: WH frame after the DG border, scanned right-to-left in
  reversed Y).
* Add a metric `phaseOffset.{left,right,top,bottom}` so we can see the
  actual offset chosen on each test, and verify the right/left offsets
  differ as expected.

Also expose **`subPixelGapWidth.{left,right}`** — the width of the
trough between WH and DG. Used as the asymmetry signal; it should be
> 0 on the right and ≈ 0 on the left.

## Phase 4 — Non-linear final correction (Task 5)

A perspective transform has 8 DoF. The user described an S-curve on the
right edge (out then back in) — that needs at least a quadratic, which
no perspective transform can produce. After the two existing perspective
refinement passes, add a **per-row / per-column remap pass**:

1. From Phase 2, we have a 2-D polynomial for each of the four borders
   in the current warped image:
   * left:  `x_lhs(y) = a_L + b_L·y + c_L·y²`
   * right: `x_rhs(y) = a_R + b_R·y + c_R·y²`
   * top:   `y_top(x) = a_T + b_T·x + c_T·x²`
   * bot:   `y_bot(x) = a_B + b_B·x + c_B·x²`
2. The **ideal** borders are the constant lines `x = INNER_LEFT*scale`,
   `x = INNER_RIGHT*scale`, `y = INNER_TOP*scale`, `y = INNER_BOT*scale`.
3. Construct a 2-D remap field `M(x, y) → (x', y')` such that the
   detected borders map to the ideal borders. Simplest workable model:
   * For each row `y`, compute a horizontal affine that sends
     `x_lhs(y) → INNER_LEFT*scale` and `x_rhs(y) → INNER_RIGHT*scale`.
   * For each column `x`, compute a vertical affine that sends
     `y_top(x) → INNER_TOP*scale` and `y_bot(x) → INNER_BOT*scale`.
   * Combine row-affine and column-affine into one displacement field.
     This is not a pure bilinear blend; we apply the row warp first to
     get the x coordinate right, then apply the column warp at that
     new x to fix y. (We can validate in a follow-up that the order
     doesn't matter much given the small magnitudes — < 5 px.)
4. Use OpenCV `cv.remap` with `cv.INTER_LANCZOS4` to apply the
   displacement field. Build the maps in float32; sub-pixel accuracy
   matters.
5. Validate by re-running detection on the remapped image and logging
   the new edge residuals — they should be ≪ 0.5 px.

This is purely **local** to the warp step; downstream steps see the
same `(160·scale, 144·scale)` RGB output and don't change.

## Phase 5 — Iterate (Task 6)

* Run `pnpm test:pipeline` after each meaningful change.
* Inspect `park-1` diagnostics every time, plus at least one
  previously-passing test to make sure we have not regressed.
* When average accuracy crosses a milestone (99 / 99.5 / 99.75 / 99.9 /
  99.99), commit and put the average in the commit title.

---

## Risks / open considerations

* **Polynomial fit overfit.** With 25 points and degree 2 we should be
  safe, but real lens distortion is degree ≥ 3. If quadratic isn't
  enough we'll bump to a constrained cubic.
* **Sub-pixel offset assumption.** The 0.5·scale offset for the right
  edge is hypothesised, not measured. Phase 3 logs the measured gap so
  we can adjust empirically.
* **Bottom-edge symmetry.** Bottom and right see the same physical
  arrangement (WH frame after DG, scanned in reverse). They get the
  same treatment.
* **Downstream effects.** A warp that is more accurate to within < 1 px
  should *only* improve correct/crop/sample/quantize. But the crop step
  uses a fixed offset; if we now place the inner border *exactly* at
  `INNER_*`, the crop will work better, not worse.
* **Sample alignment.** `sample.ts` uses fixed subpixel column indices
  `[1,3)/[3,5)/[5,7)` for B/G/R. If the warp's per-row horizontal
  alignment is off by < 1 sub-pixel column we may still see some
  asymmetric error on heavily-blurred areas. Out of scope for this
  pass; revisit if we plateau before 99.99 %.
