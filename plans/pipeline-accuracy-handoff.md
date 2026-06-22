# Pipeline Accuracy — Handoff & Continuation Notes

Living context for continuing accuracy work on the **TypeScript** pipeline
(`packages/gbcam-extract/`). Read this first, then `git log` and the two
companion plans (`normal-full-consistency.md`,
`warp-locate-improvements.md`) for the blow-by-blow.

Pipeline: **locate → warp → correct → crop → sample → quantize**. Goal:
turn a phone photo of a Game Boy Camera image on a GBA-SP screen into a
faithful 128×112 four-colour image. Palette (grayscale / on-screen):
`BK 0 #000000`, `DG 82 #9494FF`, `LG 165 #FF9494`, `WH 255 #FFFFA5`.

## Current state (as of this handoff, all merged to `main`)

Tier-1 reference accuracy (pixel diffs vs hand-corrected references; lower
is better; 14336 px total, so 2 diffs = 99.99%):

| corpus | diffs | notes |
|---|---|---|
| tier-1 normal (`locate:false`, `test-output/`) | **20** | bathhouse 3, park 1, prison 8, thing-1 3, thing-2 1, thing-3 1, zelda-1 1, zelda-2 0, zelda-3 2 |
| tier-1 full (`locate:true`, `test-output-full/`) | **25** | bathhouse 3, park 2, prison 11, thing-2 3, zelda-1 4, zelda-3 1, rest ≤1 |
| self-consistency `213443` (full vs normal) | 58 | down from 98 |
| self-consistency `165926` | 1 | |

Goal: **every tier-1 image ≤ 2 diffs (99.99%+)**, robustly (no hard-coding
to specific test images), while improving normal/full consistency.

## What's shipped (5 changes, all on `main`)

1. **`sample.ts` — directed vertical R de-bleed.** Front-light bleeds
   brighter pixels into dimmer ones ("especially vertical"), lifting a
   dithered DG pixel's R across the DG→LG boundary (the dominant error). A
   directed de-bleed removes bled-in light, gated 3 ways: local-minimum-only,
   B-gate (only pixels bluer than vertical neighbours = genuinely DG),
   magnitude floor. Env: `DEBLEED_R=0.25`, `DEBLEED_BGATE=10`, `DEBLEED_G=0`
   (G de-bleed tested, hurts).
2. **`warp.ts` — keep best-aligned refinement pass.** The 2 perspective
   passes normally converge but on blurry/doubled edges diverge (left edge
   drifts ~1 GB px/pass → "doubled left border"). Each pass is an independent
   resample of the original photo, so keep whichever measures best-aligned
   (margin `DIVERGENCE_MARGIN=1.5`). No-op on convergent images.
3. **`quantize.ts` — local adaptive WH/LG threshold.** The global G-valley
   applies one LG↔WH boundary frame-wide, but front-light gradient leaves WH
   spatially varying; dimmed-WH dither flattens to solid LG. A windowed
   local-valley threshold (gated on genuine local bimodality) fixes it.
   Constants: `LOCALWH_RADIUS=6`, MIN_SPREAD 45, MIN_DEPTH 0.6, edge cols
   excluded (owned by the per-column step).
4. **`quantize.ts` — warp-R DG-dot recovery.** `correct`'s affine gain
   amplifies bleed-lifted DG R across the boundary, but the **pre-correct
   warp** still separates DG (low R) / LG (high R) cleanly. `index.ts` now
   passes `warped` into `quantize`. Flips output-LG→DG only when 3 signals
   agree: (1) warp R below the DG/LG warp-centroid midpoint, (2) B above the
   DG/LG B-midpoint, (3) ≥3 BK neighbours (sparse-dot-on-black structure).
   Thresholds image-derived (centroid midpoints). Env: `DOT_FRAC=0.5`,
   `DOT_BK=3`.

## Established principles / hard-won facts (don't relearn these)

- **References are dithered.** Isolated single pixels are real content, not
  noise. Any spatial denoise / median / isolated-pixel vote is catastrophic
  (measured: fix 4 / break ~20k). Never smooth the label map.
- **Normal↔full divergence is largely an irreducible input floor.** The
  already-cropped sample JPEGs are recompressed/rescaled — a different lossy
  source than the 4032px originals. Warp re-aligns geometry, but ~10/px
  residual remains and `correct`'s gain amplifies it ~2×. Not a locate bug.
- **`correct` amplifies the DG floor.** Its affine gain maps [dark,white]→
  [148,255]; a bleed-lifted DG pixel (warp R≈124, ~24 above the local DG
  floor) gets amplified to ≈173-190, crossing into LG. The clean DG/LG
  separation is in the **warp**, before correct. (This is the lever change
  #4 exploits — and a likely lever for more.)
- **B is itself bled.** True-LG B is lifted to overlap true-DG B; per-pixel
  B cannot separate DG/LG.
- **LG→DG is the dominant error and a per-pixel floor.** In the ambiguous-R
  band, true-LG outnumbers true-DG ~100-1000:1, so any single-signal
  reclassifier (R, B, G, warp-R, neighbour-darkness) is net-negative. Only
  **multiple independent signals ANDed together** beat the imbalance (that's
  how change #4 works). Local adaptive thresholds beat global thresholds
  whenever the discriminating signal varies spatially.
- **Local methods need the right structure.** Local *G*-valley works for
  WH/LG (finely dithered, balanced, clean G separation). A local *R*-valley
  for DG/LG does NOT (DG regions have internal R variation → spurious
  bimodality → splits solid DG; broke prison 10→108).

## Dead ends — measured, do NOT re-attempt without a new idea

- `locate` INTER_LINEAR → INTER_CUBIC final resample: rings, worse (34→53).
- Global R-valley DG/LG boundary: trades errors badly.
- Per-pixel B threshold for DG/LG: class-imbalance catastrophe.
- B + neighbour-darkness (2 signals): still net-negative (fix 14 / break ~4k).
- 3D RGB k-means (B as weighted dim): BW 0.5 no effect; BW 1.0 catastrophic.
- Vertical G de-bleed (`DEBLEED_G>0`): hurts (prison 10→19).
- White-surface interior refinement in `correct` (analogue of the dark
  refinement): a global polynomial can't localize a dim interior dip without
  oscillating; the gate that protects tier-1 also blocks the fix.
- Local R-valley DG/LG: breaks prison (splits solid DG).

## Remaining error landscape (what's left to attack)

- **LG→DG still dominant.** Change #4 fixed the DG-on-black-dot subset. The
  remainder: thing-1's middle cluster (DG dots NOT on black — among warm
  pixels, so the BK-neighbour gate misses them), prison's tight top blob,
  bathhouse mid. These need a *different* combination of signals (the
  DG-on-black structure isn't present).
- **DG→BK** (few, ~7): dim bluish pixels at the BK/DG brightness boundary;
  also bleed-contaminated.
- **LG→WH / WH→LG** (few): boundary cases the local WH/LG valley doesn't
  reach.

## Robustness assessment of shipped changes (firing on 43 held-out private images)

| change | fires on held-out | confidence |
|---|---|---|
| warp keep-best-pass | 6/43 | highest — pure algorithmic safety net, no content thresholds |
| sample R de-bleed | all (by design) | high — physics + relative/structural gates; one conservative constant (0.25) |
| local WH/LG valley | 17/43 | medium — image-derived threshold, shape-based gates; constants tuned on small corpus |
| warp-R DG-dot recovery | 13/43 | medium — image-derived thresholds, self-disables when DG/LG don't separate; but `≥3 BK` gate has least margin (BK=2 regressed) |

Each fires on a meaningful fraction of held-out images → targets a recurring
phenomenon, not specific test pixels. **Caveat:** "zero regressions" is
proven only on the 15 ground-truth images (9 tier-1 + 6 consistency); the
held-out 43 fire *sensibly* but have no reference to measure accuracy.

## Highest-leverage next directions

1. **More hand-corrected reference images** (especially DG-on-black texture
   and dim-WH regions, and a few full-photo captures). This is the single
   biggest confidence multiplier — it converts held-out "fires sensibly"
   into measured accuracy and de-risks the tuned gate constants.
2. **More 3-signal combinations for the remaining LG→DG.** The winning
   pattern (change #4) is "find ≥2-3 independent signatures that co-occur
   only for the error class." For DG dots NOT on black (thing-1), find what
   distinguishes them from surrounding LG besides BK-neighbours — e.g. warp-R
   + B + local-DG-context, or warp-R + a frame-anchor colour prediction.
3. **Attack bleed at the source.** A calibrated PSF/deconvolution in
   `warp`/`correct` space would reduce the bleed that lifts DG's R, shrinking
   the whole LG→DG class rather than reclassifying after the fact. High
   effort, high risk, but addresses root cause.
4. **`frameAwareQuantize` is dead code.** `corrected` is passed to
   `quantize()` but never used, and `frameAwareQuantize`/`classifyByFrame`
   exist but aren't called. The frame contains all 4 palette colours at known
   positions — a per-image, per-location colour calibration. This is the
   remaining "big unused information" lever. Partially explored before
   (commit daf8354); high-risk rewrite — get sign-off before committing to it.

## Workflow that works (use this)

- **Investigate before coding.** Trace specific error pixels through
  warp→correct→sample→quantize (mean of inner block, per channel). Measure
  color-pair separability with percentiles. ALWAYS check the class balance
  and the fix/break ratio of a proposed rule *before* implementing it.
- **Gate experiments behind env vars**, sweep params, bake the best as the
  default constant once proven.
- **Prove zero tier-1 regression** with `pnpm test:pipeline:all` before
  committing. Compare per-image diff counts against the baseline table above.
- **Use the 43 private images** (`sample-pictures-out-private/`) as a
  held-out firing-rate sanity check (no ground truth, but confirms a change
  isn't overfit — it should fire on several, not just the tuning image).
- **Revert cleanly** if a change trades errors; restore tracked
  `test-output*/` + `sample-pictures-out*/` artifacts with `git checkout`.
- Commit + push after each robust win; keep the per-image numbers in the
  commit message.

## Commands

```bash
pnpm test:pipeline        # quick: sample-pictures smoke + test-input-full
pnpm test:pipeline:all    # full: all six corpora (tier-1 + consistency)
pnpm typecheck            # from root
cd packages/gbcam-extract && pnpm test   # vitest (5 pre-existing toBe(0)
                                         # aspirational fails are expected)
```

Per-image numbers: `test-output/test-summary.log`,
`test-output-full/test-summary.log`, `sample-pictures-out-full/test-summary.log`.
Per-pixel debug: `test-output/<name>/debug/<name>_debug.json` (`metrics` +
`log`) and the `*_warp.png` / `*_correct.png` / `*_sample.png` /
`*_quantize_c_rg_scatter.png` images. Diagnostic helpers:
`packages/gbcam-extract/scripts/diff.mjs`, `zoom.mjs`.

## Key files

- `src/warp.ts`, `correct.ts`, `crop.ts`, `sample.ts`, `quantize.ts`,
  `locate.ts`, `index.ts` (orchestrator — passes `corrected`/`warped` to
  quantize).
- `plans/normal-full-consistency.md`, `plans/warp-locate-improvements.md` —
  prior detailed write-ups + rejected approaches.
- Rollback tag `stable-debleed-warpguard` — mid-branch checkpoint before the
  quantize work, if ever needed.
