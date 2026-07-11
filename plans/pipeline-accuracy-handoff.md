# Pipeline Accuracy — Handoff & Continuation Notes

Living context for continuing accuracy work on the **TypeScript** pipeline
(`packages/gbcam-extract/`). Read this first, then `git log` and the two
companion plans (`normal-full-consistency.md`,
`warp-locate-improvements.md`) for the blow-by-blow.

Pipeline: **locate → warp → correct → crop → sample → quantize**. Goal:
turn a phone photo of a Game Boy Camera image on a GBA-SP screen into a
faithful 128×112 four-colour image. Palette (grayscale / on-screen):
`BK 0 #000000`, `DG 82 #9494FF`, `LG 165 #FF9494`, `WH 255 #FFFFA5`.

## Current state (as of `pipeline-accuracy-2`; earlier rows = `main`)

Tier-1 reference accuracy (pixel diffs vs hand-corrected references; lower
is better; 14336 px total, so 2 diffs = 99.99%):

| corpus | main | pipeline-accuracy-2 | notes (current) |
|---|---|---|---|
| tier-1 normal (`test-output/`) | 20 | **10** | bathhouse 3, park 1, prison 2, thing-2 1, thing-3 1, zelda-3 2, rest 0 |
| tier-1 full (`test-output-full/`) | 25 | **13** (12 real) | bathhouse 2, park 2, prison 3 (1 = ref-pair phantom, see below), thing-2 2, thing-3 1, zelda-1 2, zelda-3 1, rest 0 |
| self-consistency total (6 images) | 72 | **62** | `213443` 58→54, `213416` 3→0 |
| private a/b/c (`test-output-private/`) | 14 | **7** | a-1 1, b-1 6 (2 of them suspected ref errors), c-1 0 |

New private corpora (gitignored, never commit): `test-input-private/`
(a/b/c with references — auto-picked-up by both test modes),
`sample-pictures-private/` (44 photos, no references — held-out firing
checks, `--mode=all` only).

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

### Added on `pipeline-accuracy-2` (2026-07-10)

5. **`quantize.ts` — palette-colour presence detection.** An image may
   contain no BK at all (`sample-pictures-private/20260613_134936.jpg`).
   Forced k=4 splits a present colour and the bijective cluster→palette
   match assigns real pixels to the absent colour (catastrophic). Now:
   count pixels whose nearest RG target is each colour; colours with < 24
   supporters are dropped and k-means runs with k = present colours
   (global + strips + all downstream loops). Support is 0 for the no-BK
   image vs ≥ 500 everywhere else — huge margin.
6. **`quantize.ts` — fake-DG-cluster dissolution.** With true DG nearly
   absent, the DG cluster migrates into the dim tail of the LG cloud. Real
   DG is blue: DG-labelled mean B sits ≥ 14 above LG's on every image with
   real DG, ≈ 0 for a migrated cluster. On failed validation (sep < 8),
   dissolve the cluster into LG/WH and recover as DG only individually-blue
   warm pixels (B − R ≥ 12). Recovery is warm-only (BK reads blue under the
   front light).
7. **`quantize.ts` — DG-dot recovery second pass (3h).** Dots with warp R
   just ABOVE the midpoint (f < 0.8) flip too, behind stricter structural
   gates: T1 bk ≥ 6 (fully on black), or T2 bk ≥ 4 AND ≤ 1 DG neighbour
   AND f < 0.6 AND bRel > 0.1. The "≤ 1 DG neighbour" gate is what kills
   the false candidates (LG pixels at DG-region boundaries have many DG
   neighbours; true sparse dots have black). Warp-R separation guard
   lowered 40 → 28 (thing-2/3 sit at 31-35 and the recovery is fix-only
   there). Measured before implementing: fix 17 / break 0 on 21 refs.
8. **`quantize.ts` — warp-G BK recovery (3i).** The same correct-gain
   amplification lifts dim BK pixels' R/B into the DG range (dimmest
   corners), but warp G still separates BK/DG decisively: true DG ≥ 0.74
   of the BK→DG warp-G span (p1), mislabelled BK ≤ 0.50. Flip DG→BK when
   warp-G fraction < 0.52 AND sample-G fraction < 0.55. Measured: fix 12 /
   break 2, and both breaks are suspected reference errors.

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
- (2026-07-10) "T3" tier for bathhouse col-46 (bk≥3, dg=0, bRel>0.2,
  f<0.85): fix 3 / break 6-9 — those errors are class-indistinguishable
  from correct LG-on-black in zelda-poster-2/3.
- (2026-07-10) bk≥4 tier WITHOUT the dg≤1 gate: park-1-full (74,54)
  breaks at f=0.58/bRel=0.23 — overlaps the prison error distribution;
  only the DG-neighbour count separates them.

## Remaining error landscape (what's left to attack)

(2026-07-10 update: the old "thing-1 middle cluster / prison top blob"
LG→DG errors turned out to BE on black — they were missed by the old f<0.5
warp-R cut, not the BK gate. Changes #7/#8 cleared them.)

- **bathhouse column x=46** (3 px, normal): vertical LG/DG dither line,
  bk=3, dg=0, f 0.69-0.77. Measured indistinguishable from correct LG-on-
  black pixels in zelda-poster-2/3 (same bk/dg/f/bRel ranges) — needs a
  genuinely new signal, do not retune the existing gates for it.
- **Scattered one-offs** (1-2 px/image): prison (61,13) LG→DG with bRel
  −0.13 (B says LG — no signal agrees); prison (1,60) DG→LG reverse;
  park (83,107) LG→WH; zelda-3 (51,68) LG→DG, (127,110) WH→LG edge;
  thing-2 (42,47) LG→DG (bRel and dg-gate both fail); thing-3 (122,106)
  DG→BK (fG 0.53, just above the 0.52 cut — resist the urge, margin is
  already thin).
- **b-1 bottom-right dither corner**: at the noise floor; 2 of its
  remaining diffs are suspected reference errors (see below).

## Suspected reference errors (reported to user 2026-07-10, do not edit)

- In `test-input-private/b-output-corrected.png`, pixels (121,96), (127,98)
  and possibly (4,1) are marked DG but measure closer to the local BK
  population on every available signal (sample RGB, warp G) — same
  signature as the 8 confirmed DG→BK fixes around them. If corrected,
  b-1 should drop to ~3-4 diffs.
- `test-input-full/prison-output-corrected.png` disagrees with
  `test-input/prison-output-corrected.png` at exactly (78,63): full says
  WH, normal says LG (pipeline agrees with LG). All other normal/full
  reference pairs are identical. This inflates prison-full's diff count
  by 1 permanently until synced.

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
2. **More 3-signal combinations** — (2026-07-10: largely exhausted for
   LG→DG and DG→BK; changes #7/#8 came from exactly this pattern. What's
   left are one-offs where no measured signal agrees with the reference —
   see the error landscape. New rules here now risk fitting noise; get
   more reference data first.)
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
