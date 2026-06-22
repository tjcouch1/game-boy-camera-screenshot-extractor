# Locate Step — Design Spec

**Date:** 2026-05-01
**Branch:** crop-rotate-step

## Problem

The current TypeScript pipeline (`warp → correct → crop → sample → quantize`) starts from images that are already roughly cropped and oriented around the Game Boy Screen — see `test-input/` (~1300×1180). Real phone photos are full-frame (~4032×1816, see `test-input-full/`) and contain background environment, the GBA SP plastic bezel, the GBA SP LCD outside the rendered Game Boy Screen, and only then the Game Boy Screen itself. Today the pipeline cannot ingest those photos directly.

We need a new first step that takes a full phone photo and produces an approximately upright crop suitable for `warp` — i.e. the existing `test-input/`-style intermediate.

## Approach

Add a new pipeline step **`locate`** before `warp`:

- Find the Game Boy Screen within the full photo.
- Extract a rotated rectangle around it, expanded by a proportional margin so dark pixels still surround Frame 02 (the existing `warp` step needs that contrast).
- Output an axis-aligned image at the screen's native pixel scale in the original photo (no resampling beyond the rotation itself).

Detection uses a **hybrid candidates → frame validation** strategy: cheap brightness-based candidate generation, then validation against Frame 02-specific structural features. The exact thresholds, downsampling factor, validation feature set, and margin proportion are starting points to **tune empirically against the test set during implementation** — the design favours small, clearly-named tunables over hard-coded magic numbers.

`locate` is opt-in via `PipelineOptions.locate`, **defaulting to `true`**. Already-cropped inputs (existing `test-input/`, `sample-pictures/`) pass through cleanly because the step is designed to be a near-no-op when the screen already fills the input — but the test runner explicitly passes `locate: false` for those corpora to skip the work.

---

## Pipeline Integration

### `STEP_ORDER` and types (`common.ts`)

`STEP_ORDER` becomes `["locate", "warp", "correct", "crop", "sample", "quantize"]`. `StepName` picks up `"locate"`.

### `PipelineOptions.locate`

```ts
/**
 * Run the {@link locate} step before {@link warp} to find the Game Boy
 * Screen within a full phone photo and produce an upright crop.
 *
 * Defaults to `true`. Set to `false` for inputs that are already cropped
 * and roughly upright (e.g. the existing `test-input/` and
 * `sample-pictures/` corpora) to skip the work.
 *
 * @default true
 */
locate?: boolean;
```

### `processPicture()` (`index.ts`)

Runs `locate(input)` first when `options.locate !== false`. Progress callback fires `"locate"` step name. New optional `intermediates.locate?: GBImageData` slot when `debug: true`.

### `extract.ts` CLI

- Default `args.start` changes from `"warp"` to `"locate"` to match the API default.
- `STEP_SUFFIX` gains `locate: "_locate"`. `STEP_FUNCTIONS` gains a `locate` entry. `STEP_INPUT_SUFFIX` is unchanged (locate has no input suffix; it consumes the original photo). `collectForStart("locate", …)` collects original photos exactly like the previous `"warp"` case.
- Help text updates:
  - `STEPS (in order):` line picks up `locate`.
  - `--start STEP` / `--end STEP` parenthetical adds `locate`.
  - Two new examples show how to skip `locate` for already-cropped inputs:

    ```
    pnpm extract -- --start warp --dir ../../test-input -o ../../test-output
    pnpm extract -- --start warp photo_already_cropped.jpg -o ./out
    ```

No new CLI flag — `--start warp` already gives skipping for free.

### `AGENTS.md`

A new `### 1. Locate (`locate.ts`)` subsection is added under `## Pipeline Steps`, in the existing per-step style. Existing 1–5 (Warp through Quantize) are renumbered to 2–6. The "How to Run / Tests" section gets the new test corpora and output directories.

### Web frontend

`useProcessing` already passes `PipelineOptions`-shaped configs through. With the default `true` and no explicit UI control, full-photo inputs work out of the box. A user-facing toggle is out of scope for v1.

### Python (`gbcam-extract-py/`)

**Untouched.** Python is historical reference. The `interleave` script's `--start warp` form (or its equivalent) handles already-cropped inputs naturally; no Python `locate` is needed.

---

## Algorithm — `locate.ts`

A new file `packages/gbcam-extract/src/locate.ts` exposes:

```ts
export interface LocateOptions {
  debug?: DebugCollector;
  // Tunables exposed only if useful during empirical tuning;
  // not part of v1 unless tests demand them:
  // workingMaxDim?: number;
  // marginRatio?: number;
  // minValidationScore?: number;
}

/**
 * Locate the Game Boy Screen within a full phone photo and produce an
 * approximately upright crop suitable for the {@link warp} step.
 *
 * Detection: generate candidate bright quadrilaterals at a downsampled
 * working resolution, validate each against Frame 02 features
 * (inner-border ring, surrounding LCD-black ring), pick the highest-
 * scoring candidate, expand by a proportional margin, and extract the
 * rotated rectangle in original-image pixel space (no resampling beyond
 * the rotation itself).
 *
 * Already-cropped inputs pass through cleanly: with no room to expand,
 * the margin step clamps to image bounds and the output is essentially
 * the input.
 *
 * @throws if no candidate passes minimum frame validation.
 */
export function locate(input: GBImageData, options?: LocateOptions): GBImageData;
```

### a. Downsample

Resize the input to a working resolution where the Game Boy Screen is roughly 300–500 px wide (e.g. ~1000 px max dimension). All candidate generation and validation work happens at this resolution; only the final crop is mapped back to original-image pixel coordinates.

### b. Generate candidate quads

Threshold on brightness (loose). Find contours, fit `minAreaRect` to each, and keep candidates that pass:

- **Lower size bound** — large enough to plausibly be the screen (rejects specks and small bright regions).
- **No upper size bound** — already-cropped inputs may have the screen filling almost the whole image.
- Aspect ratio close to 160:144 (1.111) — used as a *score component*, not a hard reject, to allow for some perspective skew.
- "Quad-ness" (low deviation of the contour from its `minAreaRect`).

Produce a ranked top-N list of candidates.

### c. Validate each candidate against Frame 02

For each candidate, perspective-warp the candidate region to a normalized 160×144 (or a small multiple) and score it on Frame-02-specific features. v1 uses two complementary signals:

- **Inner-border ring presence.** At the expected location of Frame 02's 1-px-thick `#9494FF` inner border (inset by 16 px from the outer edge in the normalized frame), measure how dark the ring is relative to the surrounding white frame. Distinguishes the GB Screen from a plain white rectangle (poster, paper).
- **Surrounding LCD-black ring.** Immediately *outside* the candidate, measure how dark the surrounding band is. This is the GBA SP LCD displaying black under the front-light — consistent across photos. Helps distinguish the GB Screen from random bright shapes that happen to have white edges.

**Optional / nice-to-have:** dash-pattern correlation against `supporting-materials/Frame 02.png` along the four frame strips. Skip in v1 unless inner-border + dark-ring scoring is not discriminating enough on the test set.

Combine signals into a total score. Pick the highest-scoring candidate. If no candidate passes a minimum validation score, **throw a descriptive error** with the top candidate's component scores in the message, so the failure is debuggable.

### d. Map back, expand, rotate, crop

- Scale the chosen candidate's four corners back from working-resolution to original-image coordinates.
- Expand the rectangle outward by a proportional margin (starting value: ~5–8 % of the screen's longest side; tuned empirically to match the look of `test-input/`).
- **Clamp the expanded rectangle to the input image bounds.** For an already-cropped input, this clamp is what makes the step a near-no-op: there's no room to expand, the output ends up essentially equal to the input.
- Extract the rotated rectangle as an axis-aligned image. Output dimensions are whatever the rectangle's width and height are in original-image pixel space — *no resampling beyond the rotation itself*.

The output's aspect ratio is approximately 160:144 (preserved from the screen aspect, modulo the proportional margin and any small perspective skew that gets baked into the rotated rectangle's bounding box).

---

## Testing

### Unit test — `tests/locate.test.ts`

Vitest test that, for each image in `test-input-full/<name>.jpg`:

1. Runs `locate()`.
2. Reads the corresponding `corners.json` entry.
3. Asserts the locate step's *output rectangle* — the four corners of the final crop region in original-image coords — is within tolerance (starting tolerance: ~20 px in 4032×1816 space, tunable) of the hand-marked rectangle in `corners.json`.

This validates the whole step end-to-end against ground truth. Tight feedback loop during development.

### Pipeline tests — `scripts/run-tests.ts`

Refactor the test runner from one hardcoded corpus to a list of corpus configs. There are two tiers:

**Tier 1 — Accuracy against ground truth.** `test-input/` has hand-corrected reference images (`thing-output-corrected.png`, `zelda-poster-output-corrected.png`). The pipeline output is diffed pixel-by-pixel against the reference.

| Corpus | `locate` mode | Output dir | Reference | Purpose |
|---|---|---|---|---|
| `test-input/` | `false` | `test-output/` | hand-corrected refs in `test-input/` | **Baseline accuracy.** Existing pre-cropped inputs through the existing pipeline. Locks in current numbers; any regression here is a `warp`/`correct`/etc. issue, not a `locate` issue. |
| `test-input-full/` | `true` | `test-output-full/` | hand-corrected refs in `test-input/` | **Primary `locate` check.** End-to-end accuracy starting from full photos. Bottom-line metric for whether the new step works. |
| `test-input/` | `true` | `test-output-locate/` | hand-corrected refs in `test-input/` | **Robustness check.** Confirms `locate` is a near-no-op on already-cropped inputs. A drop vs the baseline indicates `locate` is mangling something it shouldn't touch. |

**Tier 2 — Self-consistency on `sample-pictures/`.** No hand-corrected truth exists, but we can use the `sample-pictures + locate:false` pipeline output as a *de facto reference* and check that the locate-enabled runs produce essentially the same final image. This widens coverage to 6 real-world photos that aren't in the unit-test set.

| Corpus | `locate` mode | Output dir | Reference | Purpose |
|---|---|---|---|---|
| `sample-pictures/` | `false` | `sample-pictures-out/` | (none — this run *is* the reference) | **Reference baseline** for tier-2 checks. Existing extraction behavior; outputs are emitted normally. |
| `sample-pictures/` | `true` | `sample-pictures-out-locate/` | `sample-pictures-out/` outputs from this same test-pipeline run | **Self-consistency robustness.** Confirms `locate` doesn't mangle already-cropped real-world inputs. |
| `sample-pictures-full/` | `true` | `sample-pictures-out-full/` | `sample-pictures-out/` outputs from this same test-pipeline run | **Self-consistency primary.** Confirms `locate` on real full-photo inputs reproduces the same final image as the manually-cropped version. |

The tier-2 reference is regenerated each test run (the locate:false sample-pictures run produces it, then the two locate:true runs are diffed against it). This makes it self-correcting: if the `warp/correct/...` pipeline changes, all three sample-pictures runs change together and the diffs stay meaningful.

Tier-2 diffs are reported in the summary log as pixel-difference percentages, alongside the tier-1 accuracy numbers. They're soft signals — large divergence is a flag to investigate, not an automatic test failure.

Run order matters for tier 2: the locate:false `sample-pictures` run must complete before the two diff-against-it runs start. The test runner orders the config list accordingly; locate:true runs that need a reference declare it via a `referenceFrom: <outputDir>` field on their config.

Each corpus produces its own `test-summary.log`, per-image `<name>.log`, and `debug/` directory under its respective output dir. The existing `test-output/` and `sample-pictures-out/` layouts are preserved unchanged for back-compat.

The robustness/self-consistency checks (rows 3, 5, 6) are soft signals — informative but not hard gates. Worth running from day one because they're cheap and regressions are interesting feedback.

---

## Debug Output

Following AGENTS.md conventions (`<stem>_<step>_<letter>_<name>.png`, plus structured metrics in `<stem>_debug.json`):

### Regular intermediate

- `<stem>_locate.png` — final cropped/rotated image (variable size, axis-aligned, GB Screen + proportional margin).

### Debug images (under `debug/`, `debug: true` only)

- `<stem>_locate_a_thresholded.png` — working-resolution downsampled photo with brightness threshold applied (binary). Confirms the screen "popped out" against the background.
- `<stem>_locate_b_candidates.png` — original (or working-resolution scaled-back-up) photo with all candidate quads drawn — green for the chosen one, red for rejects, validation scores labeled.
- `<stem>_locate_c_validation.png` — chosen candidate warped to a normalized 160×144 with overlays showing the regions used to score (inner-border ring band, surrounding dark-ring band, etc.). Lets you eyeball *why* a candidate scored the way it did.
- `<stem>_locate_d_output_region.png` — original photo with the final output region drawn (chosen quad expanded by margin, in green) alongside the chosen-screen quad (in cyan). Confirms the crop is taking pixels from the right place.

### Structured metrics (`metrics.locate` in `<stem>_debug.json`)

| Field | Description |
|---|---|
| `workingDim` | `[width, height]` of the downsampled working image |
| `threshold` | Brightness threshold used |
| `candidateCount` | How many candidate quads survived initial filtering |
| `chosenCandidate.{score, aspect, area, corners}` | Winning candidate's stats (corners in original-image coords) |
| `chosenCandidate.validation.{innerBorderScore, darkRingScore, totalScore}` | Component scores. Precise field set tunable during implementation. |
| `rejectedScores` | Array of rejected candidates' total scores, for diagnosing close calls |
| `marginRatio` | Margin proportion applied |
| `outputCorners` | Four corners of the final crop region in original-image coords |
| `outputSize` | `[width, height]` of the final output image |
| `passThrough` | `true` if the input looked already-cropped (margin clamped to bounds with no expansion) |

The `log` array gets human-readable lines like `[locate] threshold=180, candidates=3, chose score=0.91, output=1320×1188`.

---

## File Structure Summary

### New

- `packages/gbcam-extract/src/locate.ts` — step implementation
- `packages/gbcam-extract/tests/locate.test.ts` — vitest unit test against `corners.json`
- `docs/superpowers/specs/2026-05-01-locate-step-design.md` — this file

### Modified

- `packages/gbcam-extract/src/common.ts` — `STEP_ORDER`, `StepName`, `PipelineOptions.locate`, `PipelineResult.intermediates.locate`
- `packages/gbcam-extract/src/index.ts` — export `locate`, run it conditionally in `processPicture`
- `packages/gbcam-extract/scripts/extract.ts` — register `locate` in step maps, default `--start` to `"locate"`, update help text and examples
- `packages/gbcam-extract/scripts/run-tests.ts` — iterate over the six corpus configs above (tier 1 + tier 2); add tier-2 self-consistency diffing against `sample-pictures-out/` produced earlier in the same run
- `AGENTS.md` — new `### 1. Locate` under `## Pipeline Steps` (existing 1–5 renumbered to 2–6); test-results section picks up `test-output-full/`, `test-output-locate/`, `sample-pictures-out-locate/`, `sample-pictures-out-full/`; debug-output section documents the new images and metrics; describe the new `sample-pictures-full/` corpus alongside the existing `sample-pictures/` description

### Untouched

- `packages/gbcam-extract-py/` — historical reference only
- Web frontend — picks up the new default automatically

---

## Open / Empirical

These start as reasonable defaults and are intentionally tuned during implementation:

- Working-resolution max dimension (~1000 px is a starting point).
- Brightness threshold for candidate generation.
- Lower size bound for candidate filtering.
- Component-score weights and the minimum total score for "passes validation."
- Margin proportion (~5–8 % to start).
- Unit-test corner-tolerance (~20 px in 4032×1816 space to start).

Whether to add the optional dash-pattern correlation to validation is decided based on whether inner-border + dark-ring scoring suffices on the full test set.

---

## Success Criteria

**Hard gates:**

- `pnpm test:pipeline` produces six output directories (`test-output/`, `test-output-full/`, `test-output-locate/`, `sample-pictures-out/`, `sample-pictures-out-locate/`, `sample-pictures-out-full/`), each with its own summary log.
- `test-output/` accuracy is unchanged from current numbers (no regressions to the existing pipeline).
- `test-output-full/` accuracy is comparable to `test-output/` — starting from a full phone photo doesn't meaningfully degrade end-to-end accuracy.
- `tests/locate.test.ts` passes with all `test-input-full/` images within tolerance of `corners.json`.

**Soft signals (informative, not gates):**

- `test-output-locate/` accuracy ≈ `test-output/` accuracy (locate is a near-no-op on already-cropped inputs).
- `sample-pictures-out-locate/` outputs ≈ `sample-pictures-out/` outputs (self-consistency on already-cropped real-world photos).
- `sample-pictures-out-full/` outputs ≈ `sample-pictures-out/` outputs (self-consistency on full real-world photos — closest analogue to the bottom-line user experience).
