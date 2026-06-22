# Locate Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `locate` pipeline step that finds the Game Boy Screen within a full phone photo and produces an approximately upright crop suitable for the existing `warp` step.

**Architecture:** A hybrid candidate generation + Frame 02 validation algorithm in a new `src/locate.ts` module. Pipeline runs `locate → warp → correct → crop → sample → quantize`, with `locate` opt-in via `PipelineOptions.locate` (default `true`). The pipeline test runner is refactored to a corpus-driven config with six runs: three tier-1 (against hand-corrected refs) and three tier-2 (sample-pictures self-consistency).

**Tech Stack:** TypeScript, OpenCV.js (`@techstark/opencv-js`) for image ops, vitest for unit tests, sharp for I/O. All existing patterns and helpers — `withMats`, `imageDataToMat`, `DebugCollector` — are reused.

**Reference spec:** `docs/superpowers/specs/2026-05-01-locate-step-design.md`

**Files touched:**

- Create: `packages/gbcam-extract/src/locate.ts`, `packages/gbcam-extract/tests/locate.test.ts`
- Modify: `packages/gbcam-extract/src/common.ts`, `packages/gbcam-extract/src/index.ts`, `packages/gbcam-extract/scripts/extract.ts`, `packages/gbcam-extract/scripts/run-tests.ts`, `AGENTS.md`

**Empirical-tuning note:** Tasks 6–8 implement the detection algorithm. Concrete starting constants are given below (threshold values, margin ratios, score weights). They are *starting points* — Task 9 runs the algorithm against the real `corners.json` ground truth, and you may need to revisit constants in 6–8 to make Task 9 pass within tolerance. That's expected.

---

## Task 1: Add `locate` to the pipeline step registry

**Files:**
- Modify: `packages/gbcam-extract/src/common.ts:19-20`, `:30-47`, `:49-53`

- [ ] **Step 1: Edit `STEP_ORDER` and `StepName` types**

In `packages/gbcam-extract/src/common.ts`, change the existing `STEP_ORDER` definition (line 19) to include `"locate"` as the first element:

```ts
// ─── Pipeline step registry ───
export const STEP_ORDER = ["locate", "warp", "correct", "crop", "sample", "quantize"] as const;
export type StepName = (typeof STEP_ORDER)[number];
```

- [ ] **Step 2: Add `locate?: GBImageData` to `PipelineResult.intermediates`**

In the same file, change the `PipelineResult` interface (around line 30) so `intermediates` includes `locate`:

```ts
export interface PipelineResult {
  grayscale: GBImageData;
  intermediates?: {
    locate: GBImageData;
    warp: GBImageData;
    correct: GBImageData;
    crop: GBImageData;
    sample: GBImageData;
  };
  /**
   * Populated when `options.debug` is true. Contains diagnostic images,
   * structured per-step metrics, and a chronological log.
   */
  debug?: {
    images: Record<string, GBImageData>;
    log: string[];
    metrics: Record<string, Record<string, unknown>>;
  };
}
```

- [ ] **Step 3: Add `locate` field to `PipelineOptions`**

In the same file, update `PipelineOptions` (around line 49):

```ts
export interface PipelineOptions {
  scale?: number;
  debug?: boolean;
  onProgress?: (step: string, pct: number) => void;
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
}
```

- [ ] **Step 4: Run typecheck — expected to fail**

Run: `cd packages/gbcam-extract && pnpm typecheck`

Expected: errors complaining that `intermediates.locate` is required but not provided in `index.ts` (where `processPicture` builds the result), and possibly an unresolved `{@link locate}` (a TSDoc warning, harmless). These break in Task 3.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/src/common.ts
git commit -m "Add locate to pipeline step registry and PipelineOptions"
```

---

## Task 2: Stub `locate.ts` with a passthrough implementation

The first version of `locate` is a no-op that returns the input unchanged. This unblocks Tasks 3–4 (pipeline + CLI integration) before the real algorithm lands.

**Files:**
- Create: `packages/gbcam-extract/src/locate.ts`
- Create: `packages/gbcam-extract/tests/locate.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/gbcam-extract/tests/locate.test.ts`:

```ts
import { describe, it, expect, beforeAll } from "vitest";
import { locate } from "../src/locate.js";
import { initOpenCV } from "../src/init-opencv.js";
import { createGBImageData } from "../src/common.js";

beforeAll(async () => {
  await initOpenCV();
}, 30_000);

describe("locate (stub)", () => {
  it("returns an image when given an image (passthrough stub)", () => {
    const input = createGBImageData(100, 80);
    // Fill with mid-gray so it's not all zeroes
    for (let i = 0; i < input.data.length; i += 4) {
      input.data[i] = 128;
      input.data[i + 1] = 128;
      input.data[i + 2] = 128;
      input.data[i + 3] = 255;
    }
    const out = locate(input);
    expect(out.width).toBe(100);
    expect(out.height).toBe(80);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: FAIL with `Cannot find module '../src/locate.js'` or similar.

- [ ] **Step 3: Create the stub `locate.ts`**

Create `packages/gbcam-extract/src/locate.ts`:

```ts
/**
 * locate.ts — Find the Game Boy Screen within a full phone photo.
 *
 * The first step in the pipeline. Takes a full original photo (e.g. ~4032×1816
 * with the GBA SP somewhere inside) and produces an approximately upright crop
 * around the Game Boy Screen, suitable for the {@link warp} step.
 *
 * Algorithm: hybrid candidate generation + Frame 02 validation. Bright
 * quadrilateral candidates are generated at a downsampled working resolution,
 * then validated against Frame 02-specific structural features (inner-border
 * ring, surrounding LCD-black ring). The highest-scoring candidate is mapped
 * back to original-image coordinates, expanded by a proportional margin, and
 * extracted as an axis-aligned image.
 */

import type { GBImageData } from "./common.js";
import type { DebugCollector } from "./debug.js";

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
export function locate(input: GBImageData, _options?: LocateOptions): GBImageData {
  // STUB: passthrough. Real implementation in Tasks 6–8.
  return input;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/src/locate.ts packages/gbcam-extract/tests/locate.test.ts
git commit -m "Add locate.ts stub (passthrough) with TSDoc and unit test"
```

---

## Task 3: Wire `locate` into `processPicture()`

**Files:**
- Modify: `packages/gbcam-extract/src/index.ts`

- [ ] **Step 1: Add the import + export and run-conditional**

Edit `packages/gbcam-extract/src/index.ts`. Replace the file contents with:

```ts
export type {
  GBImageData,
  PipelineResult,
  PipelineOptions,
  GBColorValue,
  StepName,
} from "./common.js";
export {
  GB_COLORS,
  STEP_ORDER,
  CAM_W,
  CAM_H,
  SCREEN_W,
  SCREEN_H,
  createGBImageData,
} from "./common.js";
export { initOpenCV } from "./init-opencv.js";
export { applyPalette } from "./palette.js";
export { locate } from "./locate.js";
export { warp } from "./warp.js";
export { correct } from "./correct.js";
export { crop } from "./crop.js";
export { sample } from "./sample.js";
export { quantize } from "./quantize.js";
export type { PaletteEntry } from "./data/palettes-generated.js";
export {
  MAIN_PALETTES,
  ADDITIONAL_PALETTES,
  FUN_PALETTES,
} from "./data/palettes-generated.js";

import type { GBImageData, PipelineResult, PipelineOptions } from "./common.js";
import { locate } from "./locate.js";
import { warp } from "./warp.js";
import { correct } from "./correct.js";
import { crop } from "./crop.js";
import { sample } from "./sample.js";
import { quantize } from "./quantize.js";
import { createDebugCollector } from "./debug.js";

export async function processPicture(
  input: GBImageData,
  options?: PipelineOptions,
): Promise<PipelineResult> {
  const scale = options?.scale ?? 8;
  const debug = options?.debug ?? false;
  const runLocate = options?.locate ?? true;
  const onProgress = options?.onProgress;

  const collector = debug ? createDebugCollector() : undefined;

  onProgress?.("locate", 0);
  const located = runLocate ? locate(input, { debug: collector }) : input;
  onProgress?.("locate", 100);

  onProgress?.("warp", 0);
  const warped = warp(located, { scale, debug: collector });
  onProgress?.("warp", 100);

  onProgress?.("correct", 0);
  const corrected = correct(warped, { scale, debug: collector });
  onProgress?.("correct", 100);

  onProgress?.("crop", 0);
  const cropped = crop(corrected, { scale, debug: collector });
  onProgress?.("crop", 100);

  onProgress?.("sample", 0);
  const sampled = sample(cropped, { scale, debug: collector });
  onProgress?.("sample", 100);

  onProgress?.("quantize", 0);
  const quantized = quantize(sampled, { debug: collector });
  onProgress?.("quantize", 100);

  const result: PipelineResult = { grayscale: quantized };
  if (debug) {
    result.intermediates = {
      locate: located,
      warp: warped,
      correct: corrected,
      crop: cropped,
      sample: sampled,
    };
    if (collector) {
      result.debug = collector.data;
    }
  }
  return result;
}
```

- [ ] **Step 2: Run typecheck — should now pass**

Run: `cd packages/gbcam-extract && pnpm typecheck`

Expected: no errors.

- [ ] **Step 3: Run unit tests — should still pass**

Run: `cd packages/gbcam-extract && pnpm test`

Expected: all existing tests + the new locate stub test pass.

- [ ] **Step 4: Commit**

```bash
git add packages/gbcam-extract/src/index.ts
git commit -m "Run locate as first step in processPicture (default on)"
```

---

## Task 4: Update `extract.ts` CLI to know about `locate`

**Files:**
- Modify: `packages/gbcam-extract/scripts/extract.ts`

- [ ] **Step 1: Add `locate` to the CLI's step registries and change default `--start`**

Edit `packages/gbcam-extract/scripts/extract.ts`. Make four edits:

a) Add `locate` to the imports at the top (line 13–17 area):

```ts
import { locate } from "../src/locate.js";
import { warp } from "../src/warp.js";
import { correct } from "../src/correct.js";
import { crop } from "../src/crop.js";
import { sample } from "../src/sample.js";
import { quantize } from "../src/quantize.js";
```

b) Update `STEP_SUFFIX` (around line 64) to include `locate`:

```ts
const STEP_SUFFIX: Record<string, string> = {
  locate: "_locate",
  warp: "_warp",
  correct: "_correct",
  crop: "_crop",
  sample: "_sample",
  quantize: "_gbcam",
};
```

c) Update `STEP_INPUT_SUFFIX` (around line 72). `locate` has no input suffix (consumes the original photo), and `warp` consumes the locate output:

```ts
const STEP_INPUT_SUFFIX: Record<string, string> = {
  warp: "_locate",
  correct: "_warp",
  crop: "_correct",
  sample: "_crop",
  quantize: "_sample",
};
```

d) Update `STEP_FUNCTIONS` (around line 174):

```ts
const STEP_FUNCTIONS: Record<string, (input: GBImageData, scale: number) => GBImageData> = {
  locate: (input, _scale) => locate(input),
  warp: (input, scale) => warp(input, { scale }),
  correct: (input, scale) => correct(input, { scale }),
  crop: (input, scale) => crop(input, { scale }),
  sample: (input, scale) => sample(input, { scale }),
  quantize: (input, _scale) => quantize(input),
};
```

e) Update the default `args.start` in `parseArgs` (around line 200) from `"warp"` to `"locate"`:

```ts
const args: CLIArgs = {
  inputs: [],
  scale: 8,
  start: "locate",
  end: "quantize",
  cleanSteps: false,
  debug: false,
  help: false,
};
```

f) Update `collectForStart` (around line 110). Replace the early-return guard:

```ts
function collectForStart(positionalArgs: string[], dir: string | undefined, startStep: string): string[] {
  if (startStep === "locate" || startStep === "warp") {
    // locate consumes original photos; warp consumes the prior step's output
    // (handled by STEP_INPUT_SUFFIX). When --start is locate, we want plain
    // photo inputs. When --start is warp, we want either plain photos
    // (back-compat) or `_locate.png` files. To keep behavior simple, both
    // start steps load whatever the user passes via positional args / --dir.
    if (startStep === "locate") {
      return collectInputFiles(positionalArgs, dir);
    }
  }

  const suffix = STEP_INPUT_SUFFIX[startStep];
  // ... existing logic continues unchanged
```

Wait — re-reading the existing function: when `startStep === "warp"`, it currently calls `collectInputFiles` (no suffix filter). To preserve back-compat for `--start warp` against an arbitrary directory of photos, *and* to support starting from `_locate.png` files, the simplest change is to early-return `collectInputFiles` for both `locate` and `warp`:

Replace the existing function with:

```ts
function collectForStart(positionalArgs: string[], dir: string | undefined, startStep: string): string[] {
  if (startStep === "locate" || startStep === "warp") {
    return collectInputFiles(positionalArgs, dir);
  }

  const suffix = STEP_INPUT_SUFFIX[startStep];
  const files: string[] = [];

  for (const f of positionalArgs) {
    const abs = resolve(f);
    if (existsSync(abs)) {
      files.push(abs);
    }
  }

  if (dir) {
    const absDir = resolve(dir);
    if (existsSync(absDir)) {
      const entries = readdirSync(absDir);
      for (const entry of entries) {
        const stem = basename(entry, extname(entry));
        if (stem.endsWith(suffix) && extname(entry).toLowerCase() === ".png") {
          files.push(join(absDir, entry));
        }
      }
    }
  }

  return [...new Set(files)];
}
```

- [ ] **Step 2: Update the `printHelp` text**

Replace the `printHelp` function body with the updated help text (only the `STEPS` line, `--start`/`--end` text, and `EXAMPLES` change):

```ts
function printHelp() {
  console.log(`
Game Boy Camera image extractor — TypeScript pipeline CLI

USAGE
  pnpm extract -- [options] [input files...]

POSITIONAL ARGUMENTS
  input files       One or more image files to process (.jpg, .jpeg, .png)

OPTIONS
  -d, --dir DIR     Directory of input images to glob for .jpg/.jpeg/.png files
  -o, --output-dir DIR
                    Output directory (created if needed). Default: same as input.
  --scale N         Working scale (default: 8)
  --start STEP      Start pipeline at this step
                    (locate/warp/correct/crop/sample/quantize)
  --end STEP        End pipeline at this step
  --clean-steps     Delete intermediate files after pipeline completes
  --debug           Save intermediate step images
  --help            Show this help message

STEPS (in order): locate -> warp -> correct -> crop -> sample -> quantize

  locate    Find the Game Boy Screen within a full phone photo and produce
            an upright crop with margin (input to warp). Already-cropped
            photos pass through cleanly. Skip with --start warp.

EXAMPLES
  pnpm extract -- --dir ../../sample-pictures-full -o ../../sample-pictures-out
  pnpm extract -- photo.jpg -o ./out
  pnpm extract -- --start warp --dir ../../test-input -o ../../test-output
  pnpm extract -- --start warp photo_already_cropped.jpg -o ./out
  pnpm extract -- --start quantize --dir ./out -o ./out
  pnpm extract -- --dir ./photos -o ./out --debug --clean-steps
`);
}
```

- [ ] **Step 3: Run typecheck**

Run: `cd packages/gbcam-extract && pnpm typecheck`

Expected: no errors.

- [ ] **Step 4: Smoke-test the CLI runs**

Run: `cd packages/gbcam-extract && pnpm extract -- --help`

Expected: help text shows the new `locate` step in the STEPS line and the new examples.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/scripts/extract.ts
git commit -m "Register locate step in extract CLI; default --start to locate"
```

---

## Task 5: Add a working-resolution downsample helper to `locate.ts`

This task starts replacing the stub. It implements the *plumbing* of the algorithm — converting the input to a working-resolution OpenCV Mat — without yet doing any real detection.

**Files:**
- Modify: `packages/gbcam-extract/src/locate.ts`

- [ ] **Step 1: Add a synthetic-input test for downsampling**

Add to `packages/gbcam-extract/tests/locate.test.ts`:

```ts
describe("locate (downsample)", () => {
  it("does not throw on a small synthetic image with a clear bright rectangle", () => {
    const w = 1200, h = 900;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 20; data[i + 1] = 20; data[i + 2] = 20; data[i + 3] = 255;
    }
    const rectW = 600, rectH = 540;
    const rx = Math.floor((w - rectW) / 2);
    const ry = Math.floor((h - rectH) / 2);
    for (let y = ry; y < ry + rectH; y++) {
      for (let x = rx; x < rx + rectW; x++) {
        const idx = (y * w + x) * 4;
        data[idx] = 255; data[idx + 1] = 255; data[idx + 2] = 165; data[idx + 3] = 255;
      }
    }

    expect(() => locate({ data, width: w, height: h })).not.toThrow();
  });
});
```

This test will pass for the stub *and* should keep passing as the algorithm comes online with a synthetic GB-screen-shaped bright rectangle (later tasks will tighten its assertions).

- [ ] **Step 2: Run the test**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: PASS (stub doesn't throw).

- [ ] **Step 3: Replace the stub with the algorithm scaffolding**

Replace the body of `locate()` in `packages/gbcam-extract/src/locate.ts`. The full file becomes:

```ts
/**
 * locate.ts — Find the Game Boy Screen within a full phone photo.
 * (Module-level TSDoc unchanged from Task 2.)
 */

import type { GBImageData } from "./common.js";
import { type DebugCollector, cloneImage } from "./debug.js";
import { getCV, withMats, imageDataToMat, matToImageData } from "./opencv.js";

// ─── Tunables ───
//
// Starting values, tunable empirically against the corners.json unit test
// (Task 9). If detection fails on any of the test-input-full images, revisit
// these constants.

/** Target max dimension (px) of the working-resolution image. */
const WORKING_MAX_DIM = 1000;

/** Brightness threshold for candidate generation (0–255). */
const BRIGHTNESS_THRESHOLD = 180;

/** Minimum candidate area as a fraction of the working-resolution image area. */
const MIN_CANDIDATE_AREA_FRAC = 0.02;

/** Number of top-ranked candidates to validate. */
const TOP_N_CANDIDATES = 5;

/** Margin to expand the chosen rectangle by, as a fraction of its longest side. */
const MARGIN_RATIO = 0.06;

/** Minimum total validation score to accept a candidate (0–1). */
const MIN_VALIDATION_SCORE = 0.35;

/** Target screen aspect (160 / 144). */
const TARGET_ASPECT = 160 / 144;

// ─── Public interface ───

export interface LocateOptions {
  debug?: DebugCollector;
}

/**
 * Locate the Game Boy Screen within a full phone photo and produce an
 * approximately upright crop suitable for the {@link warp} step.
 * (Function-level TSDoc unchanged from Task 2.)
 */
export function locate(input: GBImageData, options?: LocateOptions): GBImageData {
  const dbg = options?.debug;

  const cv = getCV();
  const src = imageDataToMat(input);

  return withMats((track) => {
    track(src);

    // ── 2a. Downsample to working resolution ──
    const work = downsampleToWorking(src, WORKING_MAX_DIM);
    track(work.mat);

    // Threshold to binary at working resolution
    const gray = track(new cv.Mat());
    cv.cvtColor(work.mat, gray, cv.COLOR_RGBA2GRAY);
    const binary = track(new cv.Mat());
    cv.threshold(gray, binary, BRIGHTNESS_THRESHOLD, 255, cv.THRESH_BINARY);

    if (dbg) {
      const binaryRgba = track(new cv.Mat());
      cv.cvtColor(binary, binaryRgba, cv.COLOR_GRAY2RGBA);
      dbg.addImage("locate_a_thresholded", matToImageData(binaryRgba));
      dbg.log(
        `[locate] working-res ${work.mat.cols}×${work.mat.rows} ` +
          `(scale=${work.scale.toFixed(3)} from ${input.width}×${input.height}); ` +
          `threshold=${BRIGHTNESS_THRESHOLD}`,
      );
      dbg.setMetric("locate", "workingDim", [work.mat.cols, work.mat.rows]);
      dbg.setMetric("locate", "threshold", BRIGHTNESS_THRESHOLD);
    }

    // ── 2b. Generate candidate quads ──
    // (added in Task 6)

    // ── 2c. Validate candidates against Frame 02 ──
    // (added in Task 7)

    // ── 2d. Map back, expand, rotate, crop ──
    // (added in Task 8; for now, return passthrough)

    return cloneImage(input);
  });
}

// ─── Helpers ───

/**
 * Downsample `src` so the longest side is at most `maxDim`. Returns the
 * downsampled Mat (caller is responsible for tracking/deletion via withMats)
 * and the scale factor (working-resolution px per original-image px) so
 * detected coordinates can be mapped back later.
 */
function downsampleToWorking(src: any, maxDim: number): { mat: any; scale: number } {
  const cv = getCV();
  const w = src.cols;
  const h = src.rows;
  const longest = Math.max(w, h);
  if (longest <= maxDim) {
    // No downsampling needed; clone so the caller's track/delete contract is uniform
    const out = new cv.Mat();
    src.copyTo(out);
    return { mat: out, scale: 1 };
  }
  const scale = maxDim / longest;
  const newW = Math.round(w * scale);
  const newH = Math.round(h * scale);
  const out = new cv.Mat();
  cv.resize(src, out, new cv.Size(newW, newH), 0, 0, cv.INTER_AREA);
  return { mat: out, scale };
}
```

- [ ] **Step 4: Run tests — should still pass**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: both locate tests PASS (function still returns a clone of the input, but now also performs downsample + threshold).

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/src/locate.ts packages/gbcam-extract/tests/locate.test.ts
git commit -m "locate: downsample + threshold scaffolding with debug image"
```

---

## Task 6: Implement candidate generation (step 2b)

Find contours at working resolution, fit `minAreaRect` to each, keep the top-N by score (aspect ratio + quad-ness).

**Files:**
- Modify: `packages/gbcam-extract/src/locate.ts`

- [ ] **Step 1: Add types and the candidate-generation function**

In `packages/gbcam-extract/src/locate.ts`, add after the `Helpers` heading (before `downsampleToWorking`):

```ts
type Point = [number, number];
type Corners = [Point, Point, Point, Point]; // TL, TR, BR, BL

interface Candidate {
  /** Corners ordered TL, TR, BR, BL in working-resolution pixel coords. */
  corners: Corners;
  /** Width/height from the candidate's minAreaRect, sorted so width >= height. */
  width: number;
  height: number;
  area: number;
  /** Composite score (lower = better fit to expected screen shape). */
  score: number;
}

/**
 * Order four points TL, TR, BR, BL using the same sum/diff heuristic that
 * warp.ts uses, so detected corners are consistent across the codebase.
 */
function orderCornersTLTRBRBL(pts: Point[]): Corners {
  const sums = pts.map(([x, y]) => x + y);
  const yMinusX = pts.map(([x, y]) => y - x);
  const tlIdx = sums.indexOf(Math.min(...sums));
  const brIdx = sums.indexOf(Math.max(...sums));
  const trIdx = yMinusX.indexOf(Math.min(...yMinusX));
  const blIdx = yMinusX.indexOf(Math.max(...yMinusX));
  return [pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx]];
}

/**
 * Find candidate quads in a binary (already-thresholded) working-resolution
 * image. Returns up to `topN` candidates ranked by score (lower is better).
 */
function findCandidates(binary: any, topN: number): Candidate[] {
  const cv = getCV();
  const imgArea = binary.cols * binary.rows;
  const minArea = imgArea * MIN_CANDIDATE_AREA_FRAC;

  return withMats((track) => {
    const contours = track(new cv.MatVector());
    const hierarchy = track(new cv.Mat());
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const candidates: Candidate[] = [];
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      if (area < minArea) continue;

      const rect = cv.minAreaRect(contour);
      // boxPoints: returns 4 vertices of the rotated rect
      const boxMat = track(new cv.Mat());
      cv.boxPoints(rect, boxMat);
      const pts: Point[] = [];
      for (let k = 0; k < 4; k++) {
        pts.push([boxMat.data32F[k * 2], boxMat.data32F[k * 2 + 1]]);
      }
      const corners = orderCornersTLTRBRBL(pts);

      const w = rect.size.width;
      const h = rect.size.height;
      const longSide = Math.max(w, h);
      const shortSide = Math.max(Math.min(w, h), 1);
      const aspect = longSide / shortSide;
      const aspectErr = Math.abs(aspect / TARGET_ASPECT - 1);

      // Quad-ness: how close the contour's area is to the minAreaRect's area.
      // Genuine rectangles fill their minAreaRect tightly.
      const rectArea = w * h;
      const fillRatio = rectArea > 0 ? area / rectArea : 0;
      const quadnessErr = Math.max(0, 1 - fillRatio);

      const score = aspectErr * 1.5 + quadnessErr * 1.0;

      candidates.push({
        corners,
        width: longSide,
        height: shortSide,
        area,
        score,
      });
    }

    candidates.sort((a, b) => a.score - b.score);
    return candidates.slice(0, topN);
  });
}
```

- [ ] **Step 2: Add a debug image helper for candidate visualization**

Add `drawCandidates` near the bottom of `locate.ts`, below `downsampleToWorking`:

```ts
/**
 * Draw all candidates on a copy of the working-resolution photo. The
 * `chosen` candidate (index into `candidates`, or -1 for none) is drawn in
 * green; all others in red. Each is labeled with its score.
 */
function drawCandidates(
  workingRgba: GBImageData,
  candidates: Candidate[],
  chosen: number,
): GBImageData {
  // We avoid pulling in font rendering — just a polyline per candidate is
  // enough for visual debugging. Scores appear in the structured JSON metrics.
  const out = cloneImage(workingRgba);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const color: [number, number, number] = i === chosen ? [0, 255, 0] : [255, 0, 0];
    drawPolylineRGBA(out, c.corners, color, 2, true);
  }
  return out;
}

function drawPolylineRGBA(
  img: GBImageData,
  pts: Point[],
  color: [number, number, number],
  thickness: number,
  closed: boolean,
): void {
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    if (i === n - 1 && !closed) break;
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % n];
    drawLineRGBA(img, x0, y0, x1, y1, color, thickness);
  }
}

function drawLineRGBA(
  img: GBImageData,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: [number, number, number],
  thickness: number,
): void {
  // Bresenham with a thickness pad
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const steps = Math.max(dx, dy);
  const r = Math.max(1, Math.floor(thickness / 2));
  for (let i = 0; i <= steps; i++) {
    const t = steps === 0 ? 0 : i / steps;
    const x = Math.round(x0 + (x1 - x0) * t);
    const y = Math.round(y0 + (y1 - y0) * t);
    for (let dyp = -r; dyp <= r; dyp++) {
      for (let dxp = -r; dxp <= r; dxp++) {
        const px = x + dxp;
        const py = y + dyp;
        if (px < 0 || py < 0 || px >= img.width || py >= img.height) continue;
        const idx = (py * img.width + px) * 4;
        img.data[idx] = color[0];
        img.data[idx + 1] = color[1];
        img.data[idx + 2] = color[2];
        img.data[idx + 3] = 255;
      }
    }
  }
}
```

- [ ] **Step 3: Wire candidate generation into `locate()`**

In `locate()`, replace the `// (added in Task 6)` placeholder with:

```ts
    // ── 2b. Generate candidate quads ──
    const candidates = findCandidates(binary, TOP_N_CANDIDATES);

    if (candidates.length === 0) {
      throw new Error(
        `[locate] No candidate quadrilaterals found at threshold=${BRIGHTNESS_THRESHOLD}. ` +
          `The Game Boy Screen may be too dark or the photo too distant.`,
      );
    }

    if (dbg) {
      const workingRgba = matToImageData(work.mat);
      // Chosen index is unknown until validation runs (Task 7). Draw all
      // candidates as red here; Task 7 overwrites this debug image with
      // the chosen candidate highlighted in green.
      dbg.addImage("locate_b_candidates", drawCandidates(workingRgba, candidates, -1));
      dbg.log(
        `[locate] found ${candidates.length} candidate(s); ` +
          `top score=${candidates[0].score.toFixed(3)}`,
      );
      dbg.setMetric("locate", "candidateCount", candidates.length);
    }
```

- [ ] **Step 4: Run tests**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: both locate tests PASS. The synthetic-rectangle test passes because there's a clear bright rectangle to find; the stub test passes because the input is uniformly mid-gray (no bright contour, but threshold above 128 produces an empty binary, so we'd hit the "no candidate" throw).

Wait — that's a regression. The stub test fills with `data[i] = 128`, but a uniform image at 128 thresholded at 180 gives all-zeros and zero candidates → throw. Update the stub test to use a brighter rectangle so it has something to find:

In `tests/locate.test.ts`, replace the stub test with:

```ts
describe("locate (basic)", () => {
  it("returns an image when given a synthetic photo with a bright rectangle", () => {
    const w = 600, h = 500;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 30; data[i + 1] = 30; data[i + 2] = 30; data[i + 3] = 255;
    }
    const rectW = 320, rectH = 288;
    const rx = Math.floor((w - rectW) / 2);
    const ry = Math.floor((h - rectH) / 2);
    for (let y = ry; y < ry + rectH; y++) {
      for (let x = rx; x < rx + rectW; x++) {
        const idx = (y * w + x) * 4;
        data[idx] = 255; data[idx + 1] = 255; data[idx + 2] = 165; data[idx + 3] = 255;
      }
    }
    const out = locate({ data, width: w, height: h });
    expect(out.width).toBeGreaterThan(0);
    expect(out.height).toBeGreaterThan(0);
  });
});
```

(The other test from Task 5 with the larger 1200×900 image already has a bright rectangle and continues to work.)

- [ ] **Step 5: Run tests again**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/gbcam-extract/src/locate.ts packages/gbcam-extract/tests/locate.test.ts
git commit -m "locate: candidate generation with score = aspect + quad-ness"
```

---

## Task 7: Implement Frame 02 validation (step 2c)

For each candidate, perspective-warp it to a normalized 160×144 image and score it on (a) inner-border ring darkness and (b) surrounding LCD-black ring darkness.

**Files:**
- Modify: `packages/gbcam-extract/src/locate.ts`

- [ ] **Step 1: Add validation types and functions**

In `locate.ts`, add below `findCandidates`:

```ts
interface ValidationScore {
  /** Score 0–1 measuring how dark the expected inner-border ring is. */
  innerBorderScore: number;
  /** Score 0–1 measuring how dark the band immediately outside the candidate is. */
  darkRingScore: number;
  /** Composite total, 0–1 (higher = better). */
  totalScore: number;
}

/**
 * Validate a candidate by perspective-warping it to a normalized 160×144
 * image and scoring two Frame 02 features:
 *   - Inner-border ring: at the expected location of Frame 02's #9494FF
 *     inner border (inset 16 px from the outer edge), the ring should be
 *     darker than the surrounding white frame.
 *   - Surrounding dark ring: a band immediately outside the candidate (in
 *     working-resolution coords) should be darker than the candidate's
 *     interior — this is the GBA SP LCD-black under the front-light.
 *
 * Both signals are normalized to 0–1 and averaged into `totalScore`.
 */
function validateCandidate(
  workingRgba: any /* cv.Mat */,
  candidate: Candidate,
): ValidationScore {
  const cv = getCV();

  return withMats((track) => {
    // ── Inner-border ring: warp candidate to normalized 160×144 ──
    const N = 160; // normalized width
    const M = 144; // normalized height
    const srcPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      candidate.corners[0][0], candidate.corners[0][1],
      candidate.corners[1][0], candidate.corners[1][1],
      candidate.corners[2][0], candidate.corners[2][1],
      candidate.corners[3][0], candidate.corners[3][1],
    ]));
    const dstPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      N - 1, 0,
      N - 1, M - 1,
      0, M - 1,
    ]));
    const Mhom = track(cv.getPerspectiveTransform(srcPts, dstPts));
    const warped = track(new cv.Mat());
    cv.warpPerspective(workingRgba, warped, Mhom, new cv.Size(N, M), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());

    const warpedGray = track(new cv.Mat());
    cv.cvtColor(warped, warpedGray, cv.COLOR_RGBA2GRAY);

    // Inner-border ring sits at row/col 15 of the normalized frame
    // (16-px-thick frame, inner border at outer edge of the camera area).
    // We measure two means:
    //   meanFrame   — interior of the 16-px frame band (excluding the ring itself)
    //   meanRing    — the inner-border ring at row/col 15
    // Score = clamp((meanFrame - meanRing) / 80, 0, 1)
    //   80 is a reasonable expected contrast (white-frame ≈ 230, ring ≈ 100).
    const ringRow = 15;
    let meanFrame = 0, frameCnt = 0;
    let meanRing = 0, ringCnt = 0;
    const data = warpedGray.data; // Uint8Array, length N*M
    for (let y = 0; y < M; y++) {
      for (let x = 0; x < N; x++) {
        const v = data[y * N + x];
        const inFrame =
          (y < 16 || y >= M - 16 || x < 16 || x >= N - 16) &&
          !(y === ringRow || y === M - 1 - ringRow || x === ringRow || x === N - 1 - ringRow);
        const onRing =
          (y === ringRow && x >= ringRow && x <= N - 1 - ringRow) ||
          (y === M - 1 - ringRow && x >= ringRow && x <= N - 1 - ringRow) ||
          (x === ringRow && y > ringRow && y < M - 1 - ringRow) ||
          (x === N - 1 - ringRow && y > ringRow && y < M - 1 - ringRow);
        if (inFrame) { meanFrame += v; frameCnt++; }
        if (onRing) { meanRing += v; ringCnt++; }
      }
    }
    meanFrame = frameCnt > 0 ? meanFrame / frameCnt : 0;
    meanRing = ringCnt > 0 ? meanRing / ringCnt : 0;
    const innerBorderScore = clamp((meanFrame - meanRing) / 80, 0, 1);

    // ── Surrounding dark ring: in working-resolution coords ──
    // Sample a band of width = ringWidth pixels just outside each edge of
    // the candidate's bounding box and compute its mean. Compare to the
    // candidate's overall interior mean.
    const ringWidth = Math.max(4, Math.round(Math.min(candidate.width, candidate.height) * 0.05));
    const bbox = boundingBoxOfCorners(candidate.corners, workingRgba.cols, workingRgba.rows);
    const interiorMean = meanGrayInBox(workingRgba, bbox, 0);
    const outsideMean = meanGrayInRingAround(workingRgba, bbox, ringWidth);
    // Expected: outsideMean << interiorMean. Score normalized to 0–1.
    const darkRingScore = clamp((interiorMean - outsideMean) / 100, 0, 1);

    const totalScore = (innerBorderScore + darkRingScore) / 2;
    return { innerBorderScore, darkRingScore, totalScore };
  });
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

interface Bbox { x0: number; y0: number; x1: number; y1: number; }

function boundingBoxOfCorners(corners: Corners, imgW: number, imgH: number): Bbox {
  const xs = corners.map((c) => c[0]);
  const ys = corners.map((c) => c[1]);
  return {
    x0: Math.max(0, Math.floor(Math.min(...xs))),
    y0: Math.max(0, Math.floor(Math.min(...ys))),
    x1: Math.min(imgW, Math.ceil(Math.max(...xs))),
    y1: Math.min(imgH, Math.ceil(Math.max(...ys))),
  };
}

function meanGrayInBox(rgba: any /* cv.Mat */, b: Bbox, channel: number): number {
  const cv = getCV();
  return withMats((track) => {
    const gray = track(new cv.Mat());
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    let sum = 0, cnt = 0;
    for (let y = b.y0; y < b.y1; y++) {
      for (let x = b.x0; x < b.x1; x++) {
        sum += gray.data[y * gray.cols + x];
        cnt++;
      }
    }
    return cnt > 0 ? sum / cnt : 0;
    // `channel` parameter retained for future per-channel scoring; unused for now.
    void channel;
  });
}

/**
 * Render an 8x-upscaled normalized-160×144 view of the chosen candidate
 * with overlays showing the inner-border ring (red) and an annotation of
 * the score values (drawn as colored squares — top-left red square's
 * brightness encodes innerBorderScore, top-right encodes darkRingScore).
 *
 * The visualization is intentionally minimal — clusters of pixels with
 * known meaning rather than text — so we don't pull in font rendering.
 */
function renderValidationOverlay(
  workingMat: any /* cv.Mat */,
  candidate: Candidate,
  score: ValidationScore,
): GBImageData {
  const cv = getCV();
  const N = 160, M = 144, UPSCALE = 8;
  return withMats((track) => {
    const srcPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      candidate.corners[0][0], candidate.corners[0][1],
      candidate.corners[1][0], candidate.corners[1][1],
      candidate.corners[2][0], candidate.corners[2][1],
      candidate.corners[3][0], candidate.corners[3][1],
    ]));
    const dstPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0, N - 1, 0, N - 1, M - 1, 0, M - 1,
    ]));
    const Mhom = track(cv.getPerspectiveTransform(srcPts, dstPts));
    const warped = track(new cv.Mat());
    cv.warpPerspective(workingMat, warped, Mhom, new cv.Size(N, M), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());

    const upscaled = track(new cv.Mat());
    cv.resize(warped, upscaled, new cv.Size(N * UPSCALE, M * UPSCALE), 0, 0, cv.INTER_NEAREST);

    const out = matToImageData(upscaled);

    // Overlay the inner-border ring at row/col 15 in normalized space — i.e.
    // row/col 15*UPSCALE in upscaled space — as a red rectangle outline.
    const ringPx = 15 * UPSCALE;
    const ringPts: Point[] = [
      [ringPx, ringPx],
      [(N - 1 - 15) * UPSCALE, ringPx],
      [(N - 1 - 15) * UPSCALE, (M - 1 - 15) * UPSCALE],
      [ringPx, (M - 1 - 15) * UPSCALE],
    ];
    drawPolylineRGBA(out, ringPts, [255, 0, 0], 2, true);

    // Score annotations: two filled squares in the top-left corner whose
    // brightness encodes the two component scores.
    const sq = 12 * UPSCALE;
    fillRectRGBA(out, 4, 4, sq, sq, [
      Math.round(255 * score.innerBorderScore),
      0,
      0,
    ]);
    fillRectRGBA(out, 4 + sq + 4, 4, sq, sq, [
      0,
      Math.round(255 * score.darkRingScore),
      0,
    ]);

    return out;
  });
}

function fillRectRGBA(
  img: GBImageData,
  x: number,
  y: number,
  w: number,
  h: number,
  color: [number, number, number],
): void {
  for (let yy = y; yy < y + h; yy++) {
    for (let xx = x; xx < x + w; xx++) {
      if (xx < 0 || yy < 0 || xx >= img.width || yy >= img.height) continue;
      const idx = (yy * img.width + xx) * 4;
      img.data[idx] = color[0];
      img.data[idx + 1] = color[1];
      img.data[idx + 2] = color[2];
      img.data[idx + 3] = 255;
    }
  }
}

function meanGrayInRingAround(rgba: any /* cv.Mat */, b: Bbox, ringWidth: number): number {
  const cv = getCV();
  return withMats((track) => {
    const gray = track(new cv.Mat());
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    let sum = 0, cnt = 0;
    const W = gray.cols;
    const H = gray.rows;
    const xa = Math.max(0, b.x0 - ringWidth);
    const xb = Math.min(W, b.x1 + ringWidth);
    const ya = Math.max(0, b.y0 - ringWidth);
    const yb = Math.min(H, b.y1 + ringWidth);
    for (let y = ya; y < yb; y++) {
      for (let x = xa; x < xb; x++) {
        const inInner = (x >= b.x0 && x < b.x1 && y >= b.y0 && y < b.y1);
        if (inInner) continue;
        sum += gray.data[y * W + x];
        cnt++;
      }
    }
    return cnt > 0 ? sum / cnt : 0;
  });
}
```

- [ ] **Step 2: Wire validation into `locate()`**

In `locate()`, replace `// (added in Task 7)` with:

```ts
    // ── 2c. Validate candidates against Frame 02 ──
    let bestIdx = -1;
    let bestScore: ValidationScore | null = null;
    const allScores: ValidationScore[] = [];
    for (let i = 0; i < candidates.length; i++) {
      const score = validateCandidate(work.mat, candidates[i]);
      allScores.push(score);
      if (!bestScore || score.totalScore > bestScore.totalScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    if (!bestScore || bestScore.totalScore < MIN_VALIDATION_SCORE) {
      const top = bestScore
        ? `top totalScore=${bestScore.totalScore.toFixed(3)} ` +
          `(innerBorder=${bestScore.innerBorderScore.toFixed(3)}, ` +
          `darkRing=${bestScore.darkRingScore.toFixed(3)})`
        : "no candidates";
      throw new Error(
        `[locate] No candidate passed Frame 02 validation. ${top}, ` +
          `min required = ${MIN_VALIDATION_SCORE}.`,
      );
    }

    if (dbg) {
      // Re-emit the candidates debug image with the chosen one in green.
      const workingRgba = matToImageData(work.mat);
      dbg.addImage("locate_b_candidates", drawCandidates(workingRgba, candidates, bestIdx));
      // Emit the chosen-candidate validation visualization.
      dbg.addImage(
        "locate_c_validation",
        renderValidationOverlay(work.mat, candidates[bestIdx], bestScore),
      );
      dbg.log(
        `[locate] chose candidate ${bestIdx}: ` +
          `totalScore=${bestScore.totalScore.toFixed(3)} ` +
          `(innerBorder=${bestScore.innerBorderScore.toFixed(3)}, ` +
          `darkRing=${bestScore.darkRingScore.toFixed(3)})`,
      );
      dbg.setMetrics("locate", {
        chosenCandidate: {
          score: candidates[bestIdx].score,
          area: candidates[bestIdx].area,
          // corners in original-image coords are written in Task 8
          validation: bestScore,
        },
        rejectedScores: allScores.map((s, i) => ({
          index: i,
          totalScore: s.totalScore,
          innerBorderScore: s.innerBorderScore,
          darkRingScore: s.darkRingScore,
        })).filter((_, i) => i !== bestIdx),
      });
    }

    const chosen = candidates[bestIdx];
    void chosen; // used in Task 8
```

- [ ] **Step 3: Run tests**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: synthetic-rectangle tests will likely **fail** because a flat-color rectangle has no inner-border ring (inner-border score will be ~0). That's fine for now — this is why we need real photo testing. Update the synthetic tests to skip validation by lowering `MIN_VALIDATION_SCORE` only for synthetic inputs — *no, don't*. Instead, accept that synthetic tests don't model Frame 02, so soften them:

Replace both synthetic tests in `tests/locate.test.ts` with a single test that just verifies the function throws a clear error (since the synthetic input lacks Frame 02):

```ts
describe("locate (synthetic)", () => {
  it("throws a clear validation error when given a flat bright rectangle (no Frame 02)", () => {
    const w = 1200, h = 900;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 30; data[i + 1] = 30; data[i + 2] = 30; data[i + 3] = 255;
    }
    const rectW = 600, rectH = 540;
    const rx = Math.floor((w - rectW) / 2);
    const ry = Math.floor((h - rectH) / 2);
    for (let y = ry; y < ry + rectH; y++) {
      for (let x = rx; x < rx + rectW; x++) {
        const idx = (y * w + x) * 4;
        data[idx] = 255; data[idx + 1] = 255; data[idx + 2] = 165; data[idx + 3] = 255;
      }
    }
    expect(() => locate({ data, width: w, height: h })).toThrow(/Frame 02 validation/);
  });
});
```

(Real-data tests against `corners.json` come in Task 9 — those exercise the full happy path.)

- [ ] **Step 4: Run tests**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/src/locate.ts packages/gbcam-extract/tests/locate.test.ts
git commit -m "locate: Frame 02 validation (inner-border + dark-ring)"
```

---

## Task 8: Implement output extraction (step 2d)

Map the chosen candidate's corners back to original-image coords, expand by `MARGIN_RATIO`, clamp to bounds, and extract the rotated rectangle as an axis-aligned image.

**Files:**
- Modify: `packages/gbcam-extract/src/locate.ts`

- [ ] **Step 1: Add expansion + extraction helpers**

In `locate.ts`, add near the bottom (below the validation helpers):

```ts
/**
 * Scale a corner array from working-resolution coords to original-image
 * coords. `workToOrig` = 1 / `scale` from `downsampleToWorking`.
 */
function scaleCorners(corners: Corners, workToOrig: number): Corners {
  return corners.map(([x, y]) => [x * workToOrig, y * workToOrig] as Point) as Corners;
}

/**
 * Expand a (possibly rotated) rectangle outward by a fraction of its
 * longest side. The expansion is along the rectangle's own axes — the
 * rectangle stays the same shape, just bigger. Corners are returned in
 * the same TL/TR/BR/BL order.
 */
function expandRotatedRect(corners: Corners, ratio: number): Corners {
  const [TL, TR, BR, BL] = corners;
  const cx = (TL[0] + TR[0] + BR[0] + BL[0]) / 4;
  const cy = (TL[1] + TR[1] + BR[1] + BL[1]) / 4;
  const expand = (p: Point): Point => {
    const dx = p[0] - cx;
    const dy = p[1] - cy;
    return [cx + dx * (1 + ratio), cy + dy * (1 + ratio)];
  };
  return [expand(TL), expand(TR), expand(BR), expand(BL)];
}

/**
 * Clamp each corner to [0, imgW]×[0, imgH]. This keeps the warp from
 * sampling out-of-bounds; for already-cropped inputs the clamp is what
 * makes the step a near-no-op.
 */
function clampCorners(corners: Corners, imgW: number, imgH: number): Corners {
  return corners.map(([x, y]) => [
    clamp(x, 0, imgW - 1),
    clamp(y, 0, imgH - 1),
  ] as Point) as Corners;
}

/**
 * Extract the rotated rectangle defined by `corners` from `srcRgba`,
 * producing an axis-aligned RGBA image. Output dimensions equal the
 * average side lengths of the rectangle (rounded to integers).
 */
function extractRotatedRect(srcRgba: any /* cv.Mat */, corners: Corners): GBImageData {
  const cv = getCV();
  const [TL, TR, BR, BL] = corners;
  const topLen = Math.hypot(TR[0] - TL[0], TR[1] - TL[1]);
  const botLen = Math.hypot(BR[0] - BL[0], BR[1] - BL[1]);
  const leftLen = Math.hypot(BL[0] - TL[0], BL[1] - TL[1]);
  const rightLen = Math.hypot(BR[0] - TR[0], BR[1] - TR[1]);
  const outW = Math.max(1, Math.round((topLen + botLen) / 2));
  const outH = Math.max(1, Math.round((leftLen + rightLen) / 2));

  return withMats((track) => {
    const srcPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      TL[0], TL[1], TR[0], TR[1], BR[0], BR[1], BL[0], BL[1],
    ]));
    const dstPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0, outW - 1, 0, outW - 1, outH - 1, 0, outH - 1,
    ]));
    const Mhom = track(cv.getPerspectiveTransform(srcPts, dstPts));
    const out = track(new cv.Mat());
    cv.warpPerspective(srcRgba, out, Mhom, new cv.Size(outW, outH), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
    return matToImageData(out);
  });
}

/**
 * Draw a polyline on a copy of `img` showing the final output region.
 * `screenCorners` are the chosen-screen corners (cyan); `outputCorners`
 * are the post-margin, post-clamp corners (green).
 */
function drawOutputRegion(
  img: GBImageData,
  screenCorners: Corners,
  outputCorners: Corners,
): GBImageData {
  const out = cloneImage(img);
  const thick = Math.max(2, Math.round(Math.min(img.width, img.height) / 400));
  drawPolylineRGBA(out, screenCorners, [0, 255, 255], thick, true); // cyan
  drawPolylineRGBA(out, outputCorners, [0, 255, 0], thick, true);   // green
  return out;
}
```

- [ ] **Step 2: Wire output extraction into `locate()`**

Replace the `// ── 2d. Map back, expand, rotate, crop ──` block in `locate()`:

```ts
    // ── 2d. Map back, expand, rotate, crop ──
    const workToOrig = 1 / work.scale;
    const screenCornersOrig = scaleCorners(chosen.corners, workToOrig);
    const expanded = expandRotatedRect(screenCornersOrig, MARGIN_RATIO);
    const clamped = clampCorners(expanded, input.width, input.height);

    // Detect pass-through: if every clamped corner equals its expanded
    // counterpart, no clamping happened and the margin was applied freely.
    // If they differ, the margin was clipped — likely an already-cropped input.
    const passThrough = expanded.some((p, i) => p[0] !== clamped[i][0] || p[1] !== clamped[i][1]);

    const output = extractRotatedRect(src, clamped);

    if (dbg) {
      dbg.addImage(
        "locate_d_output_region",
        drawOutputRegion(input, screenCornersOrig, clamped),
      );
      dbg.log(
        `[locate] output region: ${output.width}×${output.height} ` +
          `(margin=${(MARGIN_RATIO * 100).toFixed(1)}%, ` +
          `passThrough=${passThrough})`,
      );
      dbg.setMetrics("locate", {
        marginRatio: MARGIN_RATIO,
        outputCorners: clamped.map(([x, y]) => [Math.round(x), Math.round(y)]),
        outputSize: [output.width, output.height],
        passThrough,
        chosenCandidate: {
          score: chosen.score,
          area: chosen.area,
          corners: screenCornersOrig.map(([x, y]) => [Math.round(x), Math.round(y)]),
          validation: bestScore,
        },
      });
    }

    return output;
```

(Replace the prior `return cloneImage(input);` placeholder.)

- [ ] **Step 3: Run all unit tests**

Run: `cd packages/gbcam-extract && pnpm test`

Expected: all existing tests pass; the synthetic locate test still throws (no Frame 02) and that's the expected behavior.

- [ ] **Step 4: Smoke-test against a real photo**

Run: `cd packages/gbcam-extract && pnpm extract -- ../../test-input-full/thing-1.jpg -o ../../scratch-out --debug`

Expected: produces `../../scratch-out/thing-1_locate.png` (the cropped/rotated image), `../../scratch-out/thing-1_warp.png`, …, `../../scratch-out/thing-1_gbcam.png`. The `_locate.png` should look like a tighter crop around the GBA SP screen with the GB Screen approximately upright. Inspect visually.

If detection fails or the crop is way off:
- Open `../../scratch-out/debug/thing-1_locate_a_thresholded.png` — does the screen pop out vs the background?
- Open `../../scratch-out/debug/thing-1_locate_b_candidates.png` — are the right candidates being found?
- Adjust `BRIGHTNESS_THRESHOLD`, `MIN_CANDIDATE_AREA_FRAC`, or `MIN_VALIDATION_SCORE` in `locate.ts` and re-run. Concrete tuning happens in Task 9 against all six images.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/src/locate.ts
git commit -m "locate: extract rotated rect with margin + output region debug"
```

---

## Task 9: Real-data unit test against `corners.json`

This is the empirical-tuning task. Add a vitest test that runs `locate()` on every `test-input-full/*.jpg` and asserts the output corners are within tolerance of the hand-marked rectangle in `corners.json`. Iterate on Tasks 6–8 constants until all six pass.

**Files:**
- Modify: `packages/gbcam-extract/tests/locate.test.ts`

- [ ] **Step 1: Add the corners.json fixture loader and the per-image test**

Append to `packages/gbcam-extract/tests/locate.test.ts`:

```ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadImage as loadHelperImage, repoRoot } from "./helpers/load-image.js";
import { initOpenCV as ensureCV } from "../src/init-opencv.js";
import { createDebugCollector } from "../src/debug.js";

interface CornersFixture {
  images: Record<string, {
    imageSize: [number, number];
    corners: {
      topLeft: [number, number];
      topRight: [number, number];
      bottomRight: [number, number];
      bottomLeft: [number, number];
    };
  }>;
}

/** Pixel tolerance (in original-image space). 4032×1816 photos, hand-drawn rects. */
const CORNERS_TOLERANCE_PX = 50;

function loadCornersFixture(): CornersFixture {
  const path = repoRoot("supporting-materials", "hand-edited-rectangles", "corners.json");
  return JSON.parse(readFileSync(path, "utf-8"));
}

describe("locate (real photos vs corners.json)", () => {
  const fixture = loadCornersFixture();
  const stems = Object.keys(fixture.images);

  for (const stem of stems) {
    it(`${stem}: output corners within ${CORNERS_TOLERANCE_PX}px of corners.json`, async () => {
      await ensureCV();

      const inputPath = repoRoot("test-input-full", `${stem}.jpg`);
      const input = await loadHelperImage(inputPath);
      const dbg = createDebugCollector();

      // Run locate; capture its outputCorners metric for comparison.
      locate(input, { debug: dbg });
      const m = dbg.data.metrics.locate;
      expect(m, `metrics.locate should be populated`).toBeDefined();

      const outputCorners = m.outputCorners as [number, number][] | undefined;
      expect(outputCorners, `metrics.locate.outputCorners should be set`).toBeDefined();
      expect(outputCorners!.length).toBe(4);

      const expected = fixture.images[stem].corners;
      const expectedOrdered: [number, number][] = [
        expected.topLeft,
        expected.topRight,
        expected.bottomRight,
        expected.bottomLeft,
      ];

      for (let i = 0; i < 4; i++) {
        const [ox, oy] = outputCorners![i];
        const [ex, ey] = expectedOrdered[i];
        const dist = Math.hypot(ox - ex, oy - ey);
        expect(
          dist,
          `corner ${i} (${["TL", "TR", "BR", "BL"][i]}) ` +
            `output=(${ox},${oy}) expected=(${ex},${ey}) dist=${dist.toFixed(1)}`,
        ).toBeLessThan(CORNERS_TOLERANCE_PX);
      }
    }, 60_000);
  }
});
```

- [ ] **Step 2: Run the test**

Run: `cd packages/gbcam-extract && pnpm test -- locate`

Expected: all six per-image tests should ideally PASS. If any fail:

- Read the failing test's error message — it tells you which corner and how far off.
- Run `pnpm extract -- ../../test-input-full/<stem>.jpg -o ../../scratch-out --debug` to inspect debug images for that image.
- Adjust constants in `locate.ts` (most often `BRIGHTNESS_THRESHOLD`, `MARGIN_RATIO`, or `MIN_VALIDATION_SCORE`).
- Re-run.

If a single image cannot pass tolerance no matter what:

- Verify the candidate is actually being detected (look at `locate_b_candidates.png`).
- Consider widening `CORNERS_TOLERANCE_PX` (the rectangles in `corners.json` are hand-drawn — some discrepancy is fine; up to ~80 px is reasonable).
- If the dash-pattern correlation is needed (the optional v1 step from the spec), implement it now — see Task 7 spec note. Add a third score component:

  ```ts
  // Inside validateCandidate, after innerBorderScore and darkRingScore:
  const dashPatternScore = scoreDashPattern(warpedGray, N, M);
  const totalScore = (innerBorderScore + darkRingScore + dashPatternScore) / 3;
  ```

  Implement `scoreDashPattern` by correlating the four frame-strip rows/columns (the 17 horizontal dashes at row 4 and the 14 vertical dashes at column 0) against the equivalent rows/columns in `supporting-materials/Frame 02.png`. (If you reach this point, ask the user before writing the additional scoring code — it's only needed if validation isn't otherwise discriminating.)

- [ ] **Step 3: Commit (after all six pass)**

```bash
git add packages/gbcam-extract/src/locate.ts packages/gbcam-extract/tests/locate.test.ts
git commit -m "locate: real-data unit test against corners.json (all 6 within tolerance)"
```

---

## Task 10: Refactor `run-tests.ts` into a corpus-driven loop

The existing test runner has two hardcoded sections (sample-pictures, then test-input). Refactor it to iterate over a list of corpus configs. This task **only refactors** — same six existing inputs (sample-pictures + test-input), same outputs. New corpora are added in Task 11.

**Files:**
- Modify: `packages/gbcam-extract/scripts/run-tests.ts`

- [ ] **Step 1: Define the `CorpusConfig` interface and helper functions**

Edit `packages/gbcam-extract/scripts/run-tests.ts`. After the existing imports and constants, add:

```ts
// ─── Corpus config ───

interface CorpusConfig {
  /** Human-readable name shown in summary logs. */
  name: string;
  /** Absolute path to the input directory. */
  inputDir: string;
  /** Absolute path to the output directory. */
  outputDir: string;
  /** Whether to run the locate step. */
  locate: boolean;
  /**
   * Comparison mode:
   *  - "reference":   compare against hand-corrected refs in test-input/
   *                   (uses `<baseName>-output-corrected.png`)
   *  - "self":        compare against `referenceFromOutputDir`'s outputs
   *                   (used for sample-pictures self-consistency in Task 12)
   *  - "none":        no comparison (extraction only; e.g. the reference run itself)
   */
  comparison: "reference" | "self" | "none";
  /** When comparison === "self", which output dir to read references from. */
  referenceFromOutputDir?: string;
}

/**
 * Collect input files for a corpus. Includes .jpg/.jpeg/.png; skips reference
 * images (those ending in `-output-corrected.png`). For corpora that have
 * multiple numbered photos per reference (test-input/), all numbered files
 * are run; for corpora with no references (sample-pictures/), all photos run.
 */
function collectCorpusInputs(inputDir: string): string[] {
  if (!existsSync(inputDir)) return [];
  const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png"]);
  return readdirSync(inputDir)
    .filter((f) => {
      if (f.endsWith(REFERENCE_SUFFIX)) return false;
      return IMAGE_EXTS.has(extname(f).toLowerCase());
    })
    .sort()
    .map((f) => join(inputDir, f));
}

/** Find the reference path for an input photo, or null if none. */
function findReferenceFor(
  inputStem: string,
  inputDir: string,
): string | null {
  // The reference uses the *base name* (e.g. "thing" or "zelda-poster"),
  // derived by stripping the trailing "-<number>" off the input stem.
  const m = inputStem.match(/^(.*)-\d+$/);
  if (!m) return null;
  const baseName = m[1];
  const refPath = join(inputDir, `${baseName}${REFERENCE_SUFFIX}`);
  if (!existsSync(refPath)) return null;
  return refPath;
}
```

- [ ] **Step 2: Extract a `runCorpus()` function**

Add this function after `findReferenceFor`. It encapsulates one corpus's run:

```ts
/**
 * Run every input in a corpus through the pipeline. Returns the per-image
 * test results (used for the final summary).
 */
async function runCorpus(config: CorpusConfig): Promise<TestResult[]> {
  const inputs = collectCorpusInputs(config.inputDir);
  if (inputs.length === 0) {
    console.log(`[${config.name}] no inputs found in ${config.inputDir}`);
    return [];
  }

  if (!existsSync(config.outputDir)) mkdirSync(config.outputDir, { recursive: true });

  console.log(`\n${"=".repeat(70)}`);
  console.log(`CORPUS: ${config.name}  (${inputs.length} file(s), locate=${config.locate})`);
  console.log("=".repeat(70));

  const results: TestResult[] = [];
  for (const inputPath of inputs) {
    const inputFilename = basename(inputPath);
    const stem = basename(inputPath, extname(inputPath));

    let perImageOutDir: string;
    if (config.comparison === "reference" || config.comparison === "self") {
      perImageOutDir = join(config.outputDir, stem);
      if (!existsSync(perImageOutDir)) mkdirSync(perImageOutDir, { recursive: true });
    } else {
      perImageOutDir = config.outputDir;
    }

    console.log(`\n  [${config.name}] ${inputFilename}`);

    const logPath = join(perImageOutDir, `${stem}.log`);
    const logLines: string[] = [];
    const log = (msg: string) => { console.log(msg); logLines.push(msg); };

    try {
      log(`PIPELINE RUN`);
      log(`  Input:      ${relative(REPO_ROOT, inputPath)}`);
      log(`  Output dir: ${relative(REPO_ROOT, perImageOutDir)}`);
      log(`  locate:     ${config.locate}`);

      const input = await loadImage(inputPath);
      const result = await processPicture(input, {
        scale: 8,
        debug: true,
        locate: config.locate,
        onProgress: (step, pct) => {
          if (pct === 0) process.stdout.write(`    ${step}...`);
          if (pct === 100) process.stdout.write(" done\n");
        },
      });

      // Save final outputs into the per-image (or shared) output dir
      await saveImage(result.grayscale, join(perImageOutDir, `${stem}_gbcam.png`));
      const rgb = applyPalette(result.grayscale, DOWN_PALETTE);
      await saveImage(rgb, join(perImageOutDir, `${stem}_gbcam_rgb.png`));
      await writeDebugArtifacts(result, perImageOutDir, stem);

      if (result.debug?.log.length) {
        log(`\nPIPELINE DIAGNOSTICS`);
        for (const line of result.debug.log) log(`  ${line}`);
      }

      // ── Comparison ──
      if (config.comparison === "none") {
        results.push({
          name: stem, matchN: null, matchPct: null,
          diffN: null, diffPct: null, verdict: "OK",
        });
        writeFileSync(logPath, logLines.join("\n") + "\n", "utf-8");
        continue;
      }

      // Resolve reference path
      let refPath: string | null;
      if (config.comparison === "reference") {
        refPath = findReferenceFor(stem, TEST_INPUT_DIR);
      } else {
        // "self": reference is `<referenceFromOutputDir>/<stem>/<stem>_gbcam.png`
        // for per-image-dir corpora, or `<referenceFromOutputDir>/<stem>_gbcam.png`
        // for flat corpora (sample-pictures-out is flat).
        const flat = join(config.referenceFromOutputDir!, `${stem}_gbcam.png`);
        const nested = join(config.referenceFromOutputDir!, stem, `${stem}_gbcam.png`);
        refPath = existsSync(flat) ? flat : existsSync(nested) ? nested : null;
      }

      if (!refPath) {
        log(`\n  No reference image found — skipping comparison.`);
        results.push({
          name: stem, matchN: null, matchPct: null,
          diffN: null, diffPct: null, verdict: "NO REF",
        });
        writeFileSync(logPath, logLines.join("\n") + "\n", "utf-8");
        continue;
      }

      const resultGray = extractGrayscale(result.grayscale);
      const referenceGray = await loadReference(refPath);

      const cmp = compare(resultGray, referenceGray, perImageOutDir, stem, log);

      await saveErrorMap(resultGray, referenceGray, perImageOutDir, stem);
      await savePaletteImage(resultGray, perImageOutDir, `${stem}_diag_result.png`);
      await savePaletteImage(referenceGray, perImageOutDir, `${stem}_diag_reference.png`);

      writeFileSync(logPath, logLines.join("\n") + "\n", "utf-8");

      results.push({
        name: stem,
        matchN: cmp.matches,
        matchPct: cmp.matchPct,
        diffN: cmp.wrongs,
        diffPct: cmp.wrongPct,
        verdict: cmp.passed ? "PASS" : "FAIL",
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  ERROR: ${msg}`);
      if (err instanceof Error) console.error(err.stack);
      logLines.push(`PIPELINE ERROR: ${msg}`);
      writeFileSync(logPath, logLines.join("\n") + "\n", "utf-8");
      results.push({
        name: stem, matchN: null, matchPct: null,
        diffN: null, diffPct: null, verdict: "ERROR",
      });
    }
  }

  return results;
}
```

- [ ] **Step 3: Replace `main()` with a corpus-driven loop**

Replace the existing `main()` function (and the existing `writeSummary`) with:

```ts
function writeCorpusSummary(corpus: CorpusConfig, results: TestResult[]): void {
  const lines: string[] = [];
  lines.push("=".repeat(70));
  lines.push(`CORPUS SUMMARY — ${corpus.name}`);
  lines.push(`  inputDir:   ${relative(REPO_ROOT, corpus.inputDir)}`);
  lines.push(`  outputDir:  ${relative(REPO_ROOT, corpus.outputDir)}`);
  lines.push(`  locate:     ${corpus.locate}`);
  lines.push(`  comparison: ${corpus.comparison}` + (corpus.referenceFromOutputDir
    ? `  (referenceFromOutputDir: ${relative(REPO_ROOT, corpus.referenceFromOutputDir)})`
    : ""));
  lines.push("=".repeat(70));
  lines.push("");

  if (results.length === 0) {
    lines.push("  (no inputs found)");
  } else {
    const colW = Math.max(...results.map((r) => r.name.length));
    const header = `  ${"Test".padEnd(colW)}   ${"Matching".padEnd(18)}  ${"Different".padEnd(18)}  Verdict`;
    lines.push(header);
    lines.push("  " + "-".repeat(header.length - 2));
    for (const r of results) {
      const fmt = (n: number | null, pct: number | null): string => {
        if (n === null) return "       N/A       ";
        return `${String(n).padStart(5)} (${pct!.toFixed(2).padStart(6)}%)`;
      };
      lines.push(
        `  ${r.name.padEnd(colW)}   ${fmt(r.matchN, r.matchPct)}   ${fmt(r.diffN, r.diffPct)}   ${r.verdict}`,
      );
    }
    lines.push("");
    const passed = results.filter((r) => r.verdict === "PASS").length;
    const total = results.filter((r) => r.verdict !== "OK").length;
    if (total > 0) lines.push(`  ${passed}/${total} passed`);
  }

  lines.push("");
  const text = lines.join("\n") + "\n";
  console.log("\n" + text);

  if (!existsSync(corpus.outputDir)) mkdirSync(corpus.outputDir, { recursive: true });
  writeFileSync(join(corpus.outputDir, "test-summary.log"), text, "utf-8");
}

async function main() {
  console.log("Initializing OpenCV...");
  await initOpenCV();
  console.log("OpenCV ready.\n");

  // Note: corpus order matters when later corpora set comparison: "self".
  // sample-pictures + locate:false runs first because it produces the
  // self-consistency reference for sample-pictures-out-locate / -full.
  const corpora: CorpusConfig[] = [
    {
      name: "sample-pictures (locate:false)",
      inputDir: SAMPLE_PICTURES_DIR,
      outputDir: SAMPLE_PICTURES_OUT,
      locate: false,
      comparison: "none",
    },
    {
      name: "test-input (locate:false)",
      inputDir: TEST_INPUT_DIR,
      outputDir: TEST_OUTPUT_DIR,
      locate: false,
      comparison: "reference",
    },
  ];

  const allResults: { corpus: CorpusConfig; results: TestResult[] }[] = [];
  let anyError = false;

  for (const corpus of corpora) {
    const results = await runCorpus(corpus);
    writeCorpusSummary(corpus, results);
    allResults.push({ corpus, results });
    if (results.some((r) => r.verdict === "ERROR")) anyError = true;
  }

  // Top-level fail/pass: existing tier-1 ("reference") corpora must all PASS.
  // Tier-2 self-consistency corpora are soft signals (added in Task 12).
  let allPassed = true;
  for (const { corpus, results } of allResults) {
    if (corpus.comparison !== "reference") continue;
    if (results.some((r) => r.verdict !== "PASS")) allPassed = false;
  }

  if (!allPassed || anyError) process.exit(1);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
```

(Delete the previous `writeSummary` function — it's replaced by `writeCorpusSummary`. Also remove the old `runPipeline` function and the `SUMMARY_LOG` constant; their logic is now inside `runCorpus`.)

- [ ] **Step 4: Run the test pipeline**

Run: `cd packages/gbcam-extract && pnpm test:pipeline`

Expected: produces the same `test-output/` and `sample-pictures-out/` as before, with new `test-summary.log` files in each. Tier-1 accuracy numbers in `test-output/test-summary.log` should be unchanged from the previous run (no regressions).

If the numbers differ from before, investigate — the refactor should be a behavioral no-op for the existing two corpora.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/scripts/run-tests.ts
git commit -m "Refactor run-tests into corpus-driven config loop (no behavior change)"
```

---

## Task 11: Add `test-input-full` and `test-input-locate` corpora

Add the two new tier-1 corpora that exercise the new locate step against the hand-corrected references.

**Files:**
- Modify: `packages/gbcam-extract/scripts/run-tests.ts`

- [ ] **Step 1: Add new path constants**

Near the existing path constants (around line 41–46), add:

```ts
const TEST_INPUT_FULL_DIR = join(REPO_ROOT, "test-input-full");
const TEST_OUTPUT_FULL_DIR = join(REPO_ROOT, "test-output-full");
const TEST_OUTPUT_LOCATE_DIR = join(REPO_ROOT, "test-output-locate");
```

- [ ] **Step 2: Append the two new corpus configs**

In `main()`, extend the `corpora` array:

```ts
  const corpora: CorpusConfig[] = [
    {
      name: "sample-pictures (locate:false)",
      inputDir: SAMPLE_PICTURES_DIR,
      outputDir: SAMPLE_PICTURES_OUT,
      locate: false,
      comparison: "none",
    },
    {
      name: "test-input (locate:false)",
      inputDir: TEST_INPUT_DIR,
      outputDir: TEST_OUTPUT_DIR,
      locate: false,
      comparison: "reference",
    },
    {
      name: "test-input-full (locate:true)",
      inputDir: TEST_INPUT_FULL_DIR,
      outputDir: TEST_OUTPUT_FULL_DIR,
      locate: true,
      comparison: "reference",
    },
    {
      name: "test-input (locate:true)",
      inputDir: TEST_INPUT_DIR,
      outputDir: TEST_OUTPUT_LOCATE_DIR,
      locate: true,
      comparison: "reference",
    },
  ];
```

- [ ] **Step 3: Run the pipeline tests**

Run: `cd packages/gbcam-extract && pnpm test:pipeline`

Expected: four corpora run.
- `test-output/` — unchanged from baseline.
- `test-output-full/` — primary `locate` accuracy. Should be comparable to `test-output/`.
- `test-output-locate/` — robustness check. Should be comparable to `test-output/` (locate near-no-op).

If `test-output-full/` accuracy is much lower than `test-output/`, investigate which step is drifting via the `interleave` script (mentioned in AGENTS.md), or by inspecting the locate debug images for affected images.

- [ ] **Step 4: Commit**

```bash
git add packages/gbcam-extract/scripts/run-tests.ts
git commit -m "Add test-input-full and test-input-locate corpora"
```

---

## Task 12: Add tier-2 sample-pictures self-consistency corpora

**Files:**
- Modify: `packages/gbcam-extract/scripts/run-tests.ts`

- [ ] **Step 1: Add new path constants**

Add to the path constants:

```ts
const SAMPLE_PICTURES_FULL_DIR = join(REPO_ROOT, "sample-pictures-full");
const SAMPLE_PICTURES_OUT_LOCATE = join(REPO_ROOT, "sample-pictures-out-locate");
const SAMPLE_PICTURES_OUT_FULL = join(REPO_ROOT, "sample-pictures-out-full");
```

- [ ] **Step 2: Append the two tier-2 corpus configs**

Extend the `corpora` array in `main()`:

```ts
    {
      name: "sample-pictures (locate:true) [self-consistency]",
      inputDir: SAMPLE_PICTURES_DIR,
      outputDir: SAMPLE_PICTURES_OUT_LOCATE,
      locate: true,
      comparison: "self",
      referenceFromOutputDir: SAMPLE_PICTURES_OUT,
    },
    {
      name: "sample-pictures-full (locate:true) [self-consistency]",
      inputDir: SAMPLE_PICTURES_FULL_DIR,
      outputDir: SAMPLE_PICTURES_OUT_FULL,
      locate: true,
      comparison: "self",
      referenceFromOutputDir: SAMPLE_PICTURES_OUT,
    },
```

- [ ] **Step 3: Run the pipeline tests**

Run: `cd packages/gbcam-extract && pnpm test:pipeline`

Expected: six corpora run. Tier-2 corpora produce comparison percentages against the `sample-pictures-out/` outputs from the first corpus. Numbers will be soft signals (high match % is good; significant divergence flags an issue).

The exit code is determined only by tier-1 (reference comparison) corpora. Tier-2 corpora can have any verdict without failing the run.

- [ ] **Step 4: Commit**

```bash
git add packages/gbcam-extract/scripts/run-tests.ts
git commit -m "Add tier-2 sample-pictures self-consistency corpora"
```

---

## Task 13: Update `AGENTS.md`

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the `### 1. Locate` section under `## Pipeline Steps`**

In `AGENTS.md`, find the line `## Pipeline Steps` followed by `The pipeline runs five steps in order: **warp -> correct -> crop -> sample -> quantize**.`. Replace that intro line with:

```
The pipeline runs six steps in order: **locate -> warp -> correct -> crop -> sample -> quantize**.
```

Then immediately after the intro line and before `### 1. Warp`, insert:

```
### 1. Locate (`locate.ts`)

Finds the Game Boy Screen within a full phone photo. Generates candidate
bright quadrilaterals at a downsampled working resolution, validates each
against Frame 02 features (inner-border ring, surrounding LCD-black
ring), picks the best, and extracts the rotated rectangle (expanded by a
proportional margin) as an axis-aligned image. Designed to be a near-no-
op on already-cropped inputs. Opt-in via `PipelineOptions.locate`
(default `true`).

- Input: phone photo (.jpg / .png, any size)
- Output: `<stem>_locate.png` — variable size, GB Screen approximately
  upright with margin around the frame
```

Renumber `### 1. Warp` through `### 5. Quantize` to `### 2. Warp` through `### 6. Quantize`.

- [ ] **Step 2: Update the "Pipeline debug output" section to document locate's debug images**

Find the `**warp**` subsection under `#### Debug images per step`. Just before it, insert:

```
**locate**
- `<stem>_locate.png` — final cropped/rotated image (variable size, axis-aligned, GB Screen + proportional margin). Regular pipeline intermediate.
- `<stem>_locate_a_thresholded.png` — working-resolution downsampled photo with brightness threshold applied (binary). Confirms the screen "popped out" against the background.
- `<stem>_locate_b_candidates.png` — working-resolution photo with all candidate quads drawn — green for the chosen one, red for rejects.
- `<stem>_locate_c_validation.png` — chosen candidate warped to a normalized 160×144 (8× upscaled) with the inner-border-ring location drawn in red, and two score-encoded swatches in the top-left (darker = lower score). Lets you eyeball *why* a candidate was scored the way it was.
- `<stem>_locate_d_output_region.png` — original photo with the final output region drawn (chosen quad expanded by margin, in green) alongside the chosen-screen quad (in cyan). Confirms the crop is taking pixels from the right place.
```

- [ ] **Step 3: Add `locate` to the metrics schema table**

Find the table under `#### Structured metrics` headed with `| Step | Key fields |`. Add a new row at the top of the data section (above the `warp` row):

```
| `locate` | `workingDim`, `threshold`, `candidateCount`, `chosenCandidate.{score, area, corners, validation.{innerBorderScore, darkRingScore, totalScore}}`, `rejectedScores`, `marginRatio`, `outputCorners`, `outputSize`, `passThrough` |
```

- [ ] **Step 4: Update the "How to Run" / test-results sections**

Find the section starting `### Inspecting test results`. Update the bullet points so they reference all six output dirs:

```
- `test-output/<test-name>/` (TypeScript, locate:false), `test-output-full/<test-name>/` (locate:true on full photos), and `test-output-locate/<test-name>/` (locate:true on already-cropped inputs) each contain that corpus's pipeline outputs and reference-comparison diagnostics.
- `sample-pictures-out/`, `sample-pictures-out-locate/`, and `sample-pictures-out-full/` contain extraction results for the sample-pictures corpora. The `-locate` and `-full` directories also have self-consistency comparisons against `sample-pictures-out/`.
- `<output-dir>/test-summary.log` exists per-corpus.
- The other paths (`<test-name>.log`, `*_diag_*.png`, `debug/`) work the same per corpus.
```

- [ ] **Step 5: Add `sample-pictures-full/` to the repo structure**

Find the repo-structure code block under `## Repository Structure`. Add a line for `sample-pictures-full/`:

```
sample-pictures-full/  Full original phone photos (un-cropped) — locate test inputs
```

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md
git commit -m "Document locate step in AGENTS.md"
```

---

## Final Verification

- [ ] **Step 1: Run all unit tests**

Run: `cd packages/gbcam-extract && pnpm test`

Expected: all tests pass, including the corners.json real-data tests.

- [ ] **Step 2: Run the full pipeline test suite**

Run: `cd packages/gbcam-extract && pnpm test:pipeline`

Expected: six corpora run; test-output/ matches baseline; test-output-full/ accuracy ≈ test-output/.

- [ ] **Step 3: Run typecheck**

Run: `pnpm typecheck` (from repo root)

Expected: no errors in any package.

- [ ] **Step 4: Spot-check a real photo end-to-end**

Run: `cd packages/gbcam-extract && pnpm extract -- ../../test-input-full/zelda-poster-2.jpg -o ../../scratch-out --debug`

Expected: produces a clean 128×112 `_gbcam.png` matching what the pipeline produced from the corresponding pre-cropped `test-input/zelda-poster-2.jpg`.
