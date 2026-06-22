# Frame feature — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users pick a Game Boy Camera frame to wrap around their extracted image. Frame picker + per-image override + global default. The chosen frame is baked into the preview, downloaded PNG, shared image, and clipboard image.

**Architecture:** Pure splitter / composer logic lives in `gbcam-extract` (testable in Node via `sharp`). UI plumbing in `gbcam-extract-web`. A build script copies sheet PNGs into the web public folder and emits a generated TypeScript module that lists them. A runtime hook fetches sheets, splits them, dedupes alphabetically, and exposes the catalog. Persisted state stores only frame IDs and is forward/backward compatible.

**Tech Stack:** TypeScript, vitest, sharp (Node decode), Canvas (browser decode), shadcn `Popover` + `Button`, lucide-react icons, `useLocalStorage`.

**Companion spec:** `docs/superpowers/specs/2026-05-05-frame-feature-design.md`

---

## File structure

### Created in `gbcam-extract`

| File | Responsibility |
|---|---|
| `src/frames/types.ts` | `Frame` interface; nothing else |
| `src/frames/split-sheet.ts` | Splits one sheet PNG into `Frame[]` |
| `src/frames/dedupe.ts` | Cross-sheet dedupe with alphabetical-stem tiebreak |
| `src/frames/compose.ts` | Composes image + frame + palette → RGBA |
| `src/frames/index.ts` | Barrel re-export |
| `tests/frames/split-sheet.test.ts` | Unit tests (synthetic + real sheets) |
| `tests/frames/dedupe.test.ts` | Dedupe semantics |
| `tests/frames/compose.test.ts` | Composition semantics |

### Created in `gbcam-extract-web`

| File | Responsibility |
|---|---|
| `scripts/copy-frames.ts` | Copy `*.png|*.jpg|*.jpeg|*.gif|*.webp` from `supporting-materials/frames/` to `public/frames/`; emit `src/generated/FrameSheets.ts` |
| `src/hooks/useFrameCatalog.ts` | Fetch sheets + run splitter + dedupe + cache in module singleton |
| `src/components/FramePicker.tsx` | Popover-based picker (default / result modes) |

### Modified

| File | Change |
|---|---|
| `gbcam-extract/src/index.ts` | Re-export `Frame`, `splitSheet`, `dedupeFrames`, `composeFrame` |
| `gbcam-extract-web/package.json` | Add `build:frames` script; chain it into `dev`/`dev:host`/`build`/`preview`/`preview:host`/`postinstall` |
| `gbcam-extract-web/src/hooks/useAppSettings.ts` | Add `defaultFrame?: FrameSelection` |
| `gbcam-extract-web/src/hooks/useProcessing.ts` | Add `frameOverride?: FrameSelection` to `ProcessingResult` |
| `gbcam-extract-web/src/hooks/useImageHistory.ts` | Same field on history entries |
| `gbcam-extract-web/src/components/ResultCard.tsx` | Layout swap + frame integration + per-result picker |
| `gbcam-extract-web/src/App.tsx` | Catalog hook + default-frame picker + effective-frame resolution |
| `.gitignore` (root) | Ignore `public/frames` and generated `FrameSheets.ts` |

---

## Conventions for every task

- **Working directory:** repo root unless a task says otherwise.
- **Test command (gbcam-extract):** `pnpm --filter gbcam-extract test`
- **Typecheck (root, all packages):** `pnpm typecheck`
- **Build (root, all packages):** `pnpm build`
- **Web dev server:** `pnpm dev` from the repo root.
- **Frame ID format:** `<sheetStem>:<type>:<index>`, e.g. `Frames_USA:normal:1`. `index` is 1-based per `(stem, type)`.
- **Type imports** in `gbcam-extract` use `.js` extensions for ESM resolution (e.g. `from "../common.js"`); existing source already follows this.
- **Commits:** at the end of each task. Use the suggested message; adapt only if hooks reject.
- **No new dependencies** are needed in either package.

---

## Task 1 — `gbcam-extract`: Frame type module

**Files:**
- Create: `packages/gbcam-extract/src/frames/types.ts`

- [ ] **Step 1: Write the type module**

```ts
// packages/gbcam-extract/src/frames/types.ts
/**
 * A single Game Boy Camera frame extracted from a sheet PNG.
 *
 * Pixel values are pre-snapped to the four GB grayscale values
 * {0, 82, 165, 255}. Hole pixels (the 128 × 112 region where the camera
 * image goes) are stored as 255 so frames render with the lightest
 * palette colour when shown alone in the picker.
 */
export interface Frame {
  /** Stable identifier of the form "<sheetStem>:<type>:<index>". */
  id: string;
  /** Stem of the source sheet (e.g. "Frames_USA"). */
  sheetStem: string;
  /** "normal" if dimensions are exactly 160 × 144, else "wild". */
  type: "normal" | "wild";
  /** 1-based index, scoped to (sheetStem, type), in (y, x) reading order. */
  index: number;
  width: number;
  height: number;
  /** length = width × height. Each value is in {0, 82, 165, 255}. */
  pixels: Uint8ClampedArray;
  /** Top-left of the 128 × 112 hole, in frame-local coords. */
  holeX: number;
  holeY: number;
}
```

- [ ] **Step 2: Typecheck**

Run from repo root: `pnpm typecheck`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add packages/gbcam-extract/src/frames/types.ts
git commit -m "Add Frame type for frame-sheet feature"
```

---

## Task 2 — `gbcam-extract`: `splitSheet`

**Files:**
- Create: `packages/gbcam-extract/src/frames/split-sheet.ts`
- Create: `packages/gbcam-extract/tests/frames/split-sheet.test.ts`

- [ ] **Step 1: Write the failing test (synthetic sheet)**

Create `packages/gbcam-extract/tests/frames/split-sheet.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { splitSheet } from "../../src/frames/split-sheet.js";
import type { GBImageData } from "../../src/common.js";
import { loadImage, repoRoot } from "../helpers/load-image.js";

/**
 * Build a synthetic sheet:
 *   • 200 × 200 canvas
 *   • Top-left pixel is the background colour (RGBA 200,180,255,255).
 *   • Background fills everything except:
 *     - One 160 × 144 grayscale frame at (10, 10) with a 128 × 112 hole at
 *       interior (16, 16) (i.e. sheet pixel (26, 26)). Frame value 0 (black).
 *     - One 50 × 50 non-hole rectangle at (200 - 60, 200 - 60) = (140, 140),
 *       value 165 (light gray) — should be filtered out (no hole).
 *
 * Wait — 140 + 50 = 190 which is fine. But the 160×144 frame at (10,10) needs
 * to fit: 10 + 160 = 170, 10 + 144 = 154 — fits in 200 × 200.
 */
function buildSyntheticSheet(): GBImageData {
  const W = 200;
  const H = 200;
  const data = new Uint8ClampedArray(W * H * 4);
  const BG = [200, 180, 255, 255];

  // Fill background.
  for (let i = 0; i < W * H; i++) {
    data[i * 4 + 0] = BG[0];
    data[i * 4 + 1] = BG[1];
    data[i * 4 + 2] = BG[2];
    data[i * 4 + 3] = BG[3];
  }

  const setPixel = (x: number, y: number, v: number) => {
    const i = (y * W + x) * 4;
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
    data[i + 3] = 255;
  };

  // Frame body (160 × 144 of value 0) at (10, 10).
  for (let y = 10; y < 10 + 144; y++) {
    for (let x = 10; x < 10 + 160; x++) {
      setPixel(x, y, 0);
    }
  }

  // Hole (128 × 112 of background) at interior (16, 16) → sheet (26, 26).
  for (let y = 26; y < 26 + 112; y++) {
    for (let x = 26; x < 26 + 128; x++) {
      const i = (y * W + x) * 4;
      data[i] = BG[0];
      data[i + 1] = BG[1];
      data[i + 2] = BG[2];
      data[i + 3] = BG[3];
    }
  }

  // Spurious 50 × 50 rectangle at (140, 140), value 165 (no hole).
  // Fits: 140 + 50 = 190.
  for (let y = 140; y < 140 + 50; y++) {
    for (let x = 140; x < 140 + 50; x++) {
      setPixel(x, y, 165);
    }
  }

  return { data, width: W, height: H };
}

describe("splitSheet — synthetic", () => {
  it("finds the framed rectangle, classifies as normal, ignores the hole-less rectangle", () => {
    const sheet = buildSyntheticSheet();
    const frames = splitSheet(sheet, "Synthetic");
    expect(frames).toHaveLength(1);
    const f = frames[0];
    expect(f.id).toBe("Synthetic:normal:1");
    expect(f.sheetStem).toBe("Synthetic");
    expect(f.type).toBe("normal");
    expect(f.index).toBe(1);
    expect(f.width).toBe(160);
    expect(f.height).toBe(144);
    expect(f.holeX).toBe(16);
    expect(f.holeY).toBe(16);
    expect(f.pixels.length).toBe(160 * 144);

    // Frame body pixel at frame-local (0, 0) was 0 in the source.
    expect(f.pixels[0]).toBe(0);
    // Hole pixel at frame-local (16, 16) is filled with 255.
    expect(f.pixels[16 * 160 + 16]).toBe(255);
    // Every pixel is one of the four GB grayscale values.
    for (let i = 0; i < f.pixels.length; i++) {
      expect([0, 82, 165, 255]).toContain(f.pixels[i]);
    }
  });
});

describe("splitSheet — real sheets", () => {
  it("splits Frames_USA.png into a stable set of frames", async () => {
    const sheet = await loadImage(
      repoRoot("supporting-materials/frames/the-spriters-resource/Frames_USA.png"),
    );
    const frames = splitSheet(sheet, "Frames_USA");

    // Structural invariants.
    expect(frames.length).toBeGreaterThan(0);
    for (const f of frames) {
      expect(f.sheetStem).toBe("Frames_USA");
      expect(f.id).toMatch(/^Frames_USA:(normal|wild):\d+$/);
      if (f.type === "normal") {
        expect(f.width).toBe(160);
        expect(f.height).toBe(144);
      } else {
        const isExact160x144 = f.width === 160 && f.height === 144;
        expect(isExact160x144).toBe(false);
      }
      expect(f.holeX).toBeGreaterThanOrEqual(0);
      expect(f.holeY).toBeGreaterThanOrEqual(0);
      expect(f.holeX + 128).toBeLessThanOrEqual(f.width);
      expect(f.holeY + 112).toBeLessThanOrEqual(f.height);
      expect(f.pixels.length).toBe(f.width * f.height);
      for (let i = 0; i < f.pixels.length; i++) {
        const v = f.pixels[i];
        expect(v === 0 || v === 82 || v === 165 || v === 255).toBe(true);
      }
    }

    // Indices are 1-based and contiguous within (stem, type).
    const normals = frames.filter((f) => f.type === "normal").map((f) => f.index);
    const wilds = frames.filter((f) => f.type === "wild").map((f) => f.index);
    if (normals.length > 0) {
      expect(normals).toEqual(Array.from({ length: normals.length }, (_, i) => i + 1));
    }
    if (wilds.length > 0) {
      expect(wilds).toEqual(Array.from({ length: wilds.length }, (_, i) => i + 1));
    }

    // Lock in the count + per-frame metadata so future regressions surface.
    const summary = {
      total: frames.length,
      normal: normals.length,
      wild: wilds.length,
      shapes: frames.map(
        (f) => `${f.id} ${f.width}x${f.height}@${f.holeX},${f.holeY}`,
      ),
    };
    expect(summary).toMatchInlineSnapshot();
  });

  it("splits Frames_JPN.png into a stable set of frames", async () => {
    const sheet = await loadImage(
      repoRoot("supporting-materials/frames/the-spriters-resource/Frames_JPN.png"),
    );
    const frames = splitSheet(sheet, "Frames_JPN");
    expect(frames.length).toBeGreaterThan(0);
    for (const f of frames) {
      expect(f.sheetStem).toBe("Frames_JPN");
      expect(f.id).toMatch(/^Frames_JPN:(normal|wild):\d+$/);
    }
    const summary = {
      total: frames.length,
      normal: frames.filter((f) => f.type === "normal").length,
      wild: frames.filter((f) => f.type === "wild").length,
      shapes: frames.map(
        (f) => `${f.id} ${f.width}x${f.height}@${f.holeX},${f.holeY}`,
      ),
    };
    expect(summary).toMatchInlineSnapshot();
  });
});
```

- [ ] **Step 2: Run test to verify it fails (no module yet)**

Run: `pnpm --filter gbcam-extract test -- --run tests/frames/split-sheet.test.ts`
Expected: failure — module `../../src/frames/split-sheet.js` cannot be resolved.

- [ ] **Step 3: Implement `splitSheet`**

Create `packages/gbcam-extract/src/frames/split-sheet.ts`:

```ts
import type { GBImageData } from "../common.js";
import type { Frame } from "./types.js";

const HOLE_W = 128;
const HOLE_H = 112;
/** Per-channel tolerance when matching the background colour. */
const BG_TOLERANCE = 2;
/** Drop bounding boxes smaller than this — credits / labels. */
const MIN_FRAME_DIM = 32;

interface BBox {
  x0: number;
  y0: number;
  x1: number; // exclusive
  y1: number; // exclusive
}

/**
 * Split a frame-sheet PNG into individual frames.
 *
 * Algorithm:
 *   1. Read the top-left pixel as the background colour.
 *   2. Build a mask of background pixels (with ±BG_TOLERANCE per channel).
 *   3. Build a 2D prefix sum of the background mask so any rectangle's
 *      "is entirely background" check is O(1).
 *   4. Find connected components of NON-background pixels (4-connectivity).
 *   5. For each component, find the first 128 × 112 sub-rectangle that's
 *      entirely background; if found it's a frame, otherwise drop it.
 *   6. Sort the kept frames in (y, x) reading order and number per type.
 */
export function splitSheet(sheet: GBImageData, sheetStem: string): Frame[] {
  const W = sheet.width;
  const H = sheet.height;
  const data = sheet.data;

  const bgR = data[0];
  const bgG = data[1];
  const bgB = data[2];

  // bgMask[i] = 1 if pixel i is the background colour, else 0.
  const bgMask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const r = data[i * 4 + 0];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    if (
      Math.abs(r - bgR) <= BG_TOLERANCE &&
      Math.abs(g - bgG) <= BG_TOLERANCE &&
      Math.abs(b - bgB) <= BG_TOLERANCE
    ) {
      bgMask[i] = 1;
    }
  }

  // 2D prefix sum of bgMask: psum[(y+1)*(W+1) + (x+1)] = sum over (0..y, 0..x).
  const PW = W + 1;
  const psum = new Int32Array(PW * (H + 1));
  for (let y = 0; y < H; y++) {
    let rowSum = 0;
    for (let x = 0; x < W; x++) {
      rowSum += bgMask[y * W + x];
      psum[(y + 1) * PW + (x + 1)] =
        psum[y * PW + (x + 1)] + rowSum;
    }
  }

  /** Sum of bgMask over [x0, x1) × [y0, y1) (exclusive). */
  const rectSum = (x0: number, y0: number, x1: number, y1: number): number =>
    psum[y1 * PW + x1] -
    psum[y0 * PW + x1] -
    psum[y1 * PW + x0] +
    psum[y0 * PW + x0];

  /** True iff every pixel in [x0,x1)×[y0,y1) is background. */
  const isAllBg = (x0: number, y0: number, x1: number, y1: number): boolean =>
    rectSum(x0, y0, x1, y1) === (x1 - x0) * (y1 - y0);

  // Connected components of non-background pixels (4-connectivity).
  // Iterative flood fill via stack to avoid recursion.
  const seen = new Uint8Array(W * H);
  const bboxes: BBox[] = [];
  const stack: number[] = [];

  for (let i = 0; i < W * H; i++) {
    if (bgMask[i] || seen[i]) continue;
    let minX = i % W;
    let minY = (i / W) | 0;
    let maxX = minX;
    let maxY = minY;
    stack.length = 0;
    stack.push(i);
    seen[i] = 1;
    while (stack.length > 0) {
      const j = stack.pop()!;
      const x = j % W;
      const y = (j / W) | 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x > 0 && !bgMask[j - 1] && !seen[j - 1]) {
        seen[j - 1] = 1;
        stack.push(j - 1);
      }
      if (x < W - 1 && !bgMask[j + 1] && !seen[j + 1]) {
        seen[j + 1] = 1;
        stack.push(j + 1);
      }
      if (y > 0 && !bgMask[j - W] && !seen[j - W]) {
        seen[j - W] = 1;
        stack.push(j - W);
      }
      if (y < H - 1 && !bgMask[j + W] && !seen[j + W]) {
        seen[j + W] = 1;
        stack.push(j + W);
      }
    }
    bboxes.push({ x0: minX, y0: minY, x1: maxX + 1, y1: maxY + 1 });
  }

  // Filter to candidates that are big enough and have a 128 × 112 hole.
  type Candidate = { bbox: BBox; holeX: number; holeY: number };
  const candidates: Candidate[] = [];
  for (const bbox of bboxes) {
    const w = bbox.x1 - bbox.x0;
    const h = bbox.y1 - bbox.y0;
    if (w < MIN_FRAME_DIM || h < MIN_FRAME_DIM) continue;
    if (w < HOLE_W || h < HOLE_H) continue;

    let foundX = -1;
    let foundY = -1;
    outer: for (let yy = bbox.y0; yy + HOLE_H <= bbox.y1; yy++) {
      for (let xx = bbox.x0; xx + HOLE_W <= bbox.x1; xx++) {
        if (isAllBg(xx, yy, xx + HOLE_W, yy + HOLE_H)) {
          foundX = xx;
          foundY = yy;
          break outer;
        }
      }
    }
    if (foundX < 0) continue;
    candidates.push({ bbox, holeX: foundX - bbox.x0, holeY: foundY - bbox.y0 });
  }

  // Sort top-to-bottom, then left-to-right.
  candidates.sort((a, b) => {
    if (a.bbox.y0 !== b.bbox.y0) return a.bbox.y0 - b.bbox.y0;
    return a.bbox.x0 - b.bbox.x0;
  });

  // Number per type and emit Frame objects.
  let normalIdx = 0;
  let wildIdx = 0;
  const frames: Frame[] = [];
  for (const c of candidates) {
    const w = c.bbox.x1 - c.bbox.x0;
    const h = c.bbox.y1 - c.bbox.y0;
    const type: "normal" | "wild" = w === 160 && h === 144 ? "normal" : "wild";
    const index = type === "normal" ? ++normalIdx : ++wildIdx;
    const id = `${sheetStem}:${type}:${index}`;
    const pixels = new Uint8ClampedArray(w * h);
    for (let yy = 0; yy < h; yy++) {
      for (let xx = 0; xx < w; xx++) {
        const sx = c.bbox.x0 + xx;
        const sy = c.bbox.y0 + yy;
        const di = (sy * W + sx) * 4;
        const isBg = bgMask[sy * W + sx] === 1;
        if (isBg) {
          pixels[yy * w + xx] = 255;
        } else {
          pixels[yy * w + xx] = snapToGB(data[di]);
        }
      }
    }
    frames.push({
      id,
      sheetStem,
      type,
      index,
      width: w,
      height: h,
      pixels,
      holeX: c.holeX,
      holeY: c.holeY,
    });
  }
  return frames;
}

/** Snap a 0–255 value to the nearest of {0, 82, 165, 255}. */
function snapToGB(v: number): number {
  // Midpoints between adjacent levels: (0,82)=41, (82,165)=123.5, (165,255)=210.
  if (v < 41) return 0;
  if (v < 124) return 82;
  if (v < 210) return 165;
  return 255;
}
```

- [ ] **Step 4: Run synthetic test**

Run: `pnpm --filter gbcam-extract test -- --run tests/frames/split-sheet.test.ts -t "synthetic"`
Expected: PASS.

- [ ] **Step 5: Run real-sheet tests (snapshots auto-fill on first run)**

Run: `pnpm --filter gbcam-extract test -- --run tests/frames/split-sheet.test.ts`
Expected: First run writes inline snapshots into the test file and passes. If it fails for non-snapshot reasons (e.g. structural assertions), debug the splitter.

- [ ] **Step 6: Re-run to confirm snapshots are stable**

Run: `pnpm --filter gbcam-extract test -- --run tests/frames/split-sheet.test.ts`
Expected: PASS with no further snapshot writes.

- [ ] **Step 7: Commit**

```bash
git add packages/gbcam-extract/src/frames/split-sheet.ts packages/gbcam-extract/tests/frames/split-sheet.test.ts
git commit -m "Add splitSheet for splitting frame sheets into individual frames"
```

---

## Task 3 — `gbcam-extract`: `dedupeFrames`

**Files:**
- Create: `packages/gbcam-extract/src/frames/dedupe.ts`
- Create: `packages/gbcam-extract/tests/frames/dedupe.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `packages/gbcam-extract/tests/frames/dedupe.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { splitSheet } from "../../src/frames/split-sheet.js";
import { dedupeFrames } from "../../src/frames/dedupe.js";
import type { Frame } from "../../src/frames/types.js";
import { loadImage, repoRoot } from "../helpers/load-image.js";

function makeSyntheticFrame(stem: string, index: number, fillByte: number): Frame {
  const w = 160;
  const h = 144;
  const pixels = new Uint8ClampedArray(w * h).fill(fillByte);
  return {
    id: `${stem}:normal:${index}`,
    sheetStem: stem,
    type: "normal",
    index,
    width: w,
    height: h,
    pixels,
    holeX: 16,
    holeY: 16,
  };
}

describe("dedupeFrames", () => {
  it("returns [] for empty input", () => {
    expect(dedupeFrames([])).toEqual([]);
  });

  it("returns the input unchanged when there are no duplicates", () => {
    const a = makeSyntheticFrame("A", 1, 0);
    const b = makeSyntheticFrame("B", 1, 82);
    const out = dedupeFrames([a, b]);
    expect(out.map((f) => f.id)).toEqual(["A:normal:1", "B:normal:1"]);
  });

  it("keeps the alphabetically earlier sheet's frame when duplicates exist", () => {
    // Same pixels, different stems. JPN < USA alphabetically, so JPN wins.
    const usa = makeSyntheticFrame("Frames_USA", 1, 0);
    const jpn = makeSyntheticFrame("Frames_JPN", 1, 0);
    const out = dedupeFrames([usa, jpn]);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("Frames_JPN:normal:1");
  });

  it("treats different dimensions as distinct even when pixel arrays would match in prefix", () => {
    const a = makeSyntheticFrame("A", 1, 0);
    const b: Frame = { ...makeSyntheticFrame("B", 1, 0), width: 160, height: 100 };
    b.pixels = new Uint8ClampedArray(160 * 100).fill(0);
    const out = dedupeFrames([a, b]);
    expect(out).toHaveLength(2);
  });

  it("deduplicates real sheets and yields fewer frames than the sum", async () => {
    const usaSheet = await loadImage(
      repoRoot("supporting-materials/frames/the-spriters-resource/Frames_USA.png"),
    );
    const jpnSheet = await loadImage(
      repoRoot("supporting-materials/frames/the-spriters-resource/Frames_JPN.png"),
    );
    const usa = splitSheet(usaSheet, "Frames_USA");
    const jpn = splitSheet(jpnSheet, "Frames_JPN");
    const all = [...usa, ...jpn];
    const out = dedupeFrames(all);
    expect(out.length).toBeLessThan(all.length);

    // Lock count snapshot — surfaces regressions if the splitter or dedup
    // changes downstream.
    const summary = {
      usa: usa.length,
      jpn: jpn.length,
      combined: all.length,
      deduped: out.length,
      jpnWinners: out.filter((f) => f.sheetStem === "Frames_JPN").length,
      usaWinners: out.filter((f) => f.sheetStem === "Frames_USA").length,
    };
    expect(summary).toMatchInlineSnapshot();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter gbcam-extract test -- --run tests/frames/dedupe.test.ts`
Expected: failure — `dedupeFrames` not defined.

- [ ] **Step 3: Implement `dedupeFrames`**

Create `packages/gbcam-extract/src/frames/dedupe.ts`:

```ts
import type { Frame } from "./types.js";

/**
 * Remove pixel-identical duplicates across frames.
 *
 * Tiebreaker: the alphabetically-earliest `sheetStem` wins. We sort the
 * input by stem first, then walk and keep first-seen unique fingerprints.
 *
 * Fingerprint = `<width>x<height>:<type>:<FNV-1a hash of pixels>`.
 */
export function dedupeFrames(frames: Frame[]): Frame[] {
  const sorted = [...frames].sort((a, b) =>
    a.sheetStem.localeCompare(b.sheetStem),
  );
  const seen = new Set<string>();
  const out: Frame[] = [];
  for (const f of sorted) {
    const fp = `${f.width}x${f.height}:${f.type}:${fnv1a(f.pixels)}`;
    if (seen.has(fp)) continue;
    seen.add(fp);
    out.push(f);
  }
  return out;
}

/** FNV-1a 32-bit on a byte stream — fast and good enough for exact dedup. */
function fnv1a(bytes: Uint8ClampedArray): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193);
  }
  // Convert to unsigned hex.
  return (h >>> 0).toString(16);
}
```

- [ ] **Step 4: Run tests, fill snapshot, re-run**

Run: `pnpm --filter gbcam-extract test -- --run tests/frames/dedupe.test.ts`
Expected: first run fills snapshot and passes.

Run again: `pnpm --filter gbcam-extract test -- --run tests/frames/dedupe.test.ts`
Expected: PASS with no snapshot diffs.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/src/frames/dedupe.ts packages/gbcam-extract/tests/frames/dedupe.test.ts
git commit -m "Add dedupeFrames with alphabetical-stem tiebreaker"
```

---

## Task 4 — `gbcam-extract`: `composeFrame`

**Files:**
- Create: `packages/gbcam-extract/src/frames/compose.ts`
- Create: `packages/gbcam-extract/tests/frames/compose.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `packages/gbcam-extract/tests/frames/compose.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { composeFrame } from "../../src/frames/compose.js";
import { splitSheet } from "../../src/frames/split-sheet.js";
import type { Frame } from "../../src/frames/types.js";
import { applyPalette } from "../../src/palette.js";
import type { GBImageData } from "../../src/common.js";
import { loadImage, repoRoot } from "../helpers/load-image.js";

const PALETTE: [string, string, string, string] = [
  "#FFFFA5", // 255 -> WH
  "#FF9494", // 165 -> LG
  "#9494FF", // 82  -> DG
  "#000000", // 0   -> BK
];

function makeImage(width: number, height: number, value: number): GBImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4 + 0] = value;
    data[i * 4 + 1] = value;
    data[i * 4 + 2] = value;
    data[i * 4 + 3] = 255;
  }
  return { data, width, height };
}

function makeFrame(width: number, height: number, holeX: number, holeY: number, frameValue: number): Frame {
  const pixels = new Uint8ClampedArray(width * height).fill(frameValue);
  // Hole pixels stored as 255 per the splitter convention.
  for (let y = holeY; y < holeY + 112; y++) {
    for (let x = holeX; x < holeX + 128; x++) {
      pixels[y * width + x] = 255;
    }
  }
  return {
    id: "Test:normal:1",
    sheetStem: "Test",
    type: "normal",
    index: 1,
    width,
    height,
    pixels,
    holeX,
    holeY,
  };
}

describe("composeFrame", () => {
  it("places the image inside the hole and renders frame pixels through the palette", () => {
    const frame = makeFrame(160, 144, 16, 16, 0); // frame value 0 -> BK
    const image = makeImage(128, 112, 82);        // image value 82 -> DG
    const out = composeFrame(image, frame, PALETTE);

    expect(out.width).toBe(160);
    expect(out.height).toBe(144);

    // Frame pixel at (0, 0): RGB should be #000000.
    expect(out.data[0]).toBe(0);
    expect(out.data[1]).toBe(0);
    expect(out.data[2]).toBe(0);

    // Hole pixel at frame-local (16, 16): RGB should be #9494FF.
    const hi = (16 * 160 + 16) * 4;
    expect(out.data[hi + 0]).toBe(0x94);
    expect(out.data[hi + 1]).toBe(0x94);
    expect(out.data[hi + 2]).toBe(0xff);

    // Bottom-right hole pixel at frame-local (16+127, 16+111) = (143, 127).
    const bri = (127 * 160 + 143) * 4;
    expect(out.data[bri + 0]).toBe(0x94);
    expect(out.data[bri + 1]).toBe(0x94);
    expect(out.data[bri + 2]).toBe(0xff);

    // Pixel just outside the hole (15, 15) is frame -> #000000.
    const oi = (15 * 160 + 15) * 4;
    expect(out.data[oi + 0]).toBe(0);
    expect(out.data[oi + 1]).toBe(0);
    expect(out.data[oi + 2]).toBe(0);

    // Alpha is fully opaque everywhere.
    for (let i = 3; i < out.data.length; i += 4) {
      expect(out.data[i]).toBe(255);
    }
  });

  it("uses the lightest palette color for hole pixels when no image is supplied (sanity for picker thumbs)", () => {
    // Confirms that frame.pixels[hole] === 255, so applying palette to the
    // frame alone (calling composeFrame with a 128x112 image of value 255)
    // produces a uniform color in the hole region.
    const frame = makeFrame(160, 144, 16, 16, 0);
    const image = makeImage(128, 112, 255);
    const out = composeFrame(image, frame, PALETTE);
    const hi = (16 * 160 + 16) * 4;
    expect(out.data[hi + 0]).toBe(0xff);
    expect(out.data[hi + 1]).toBe(0xff);
    expect(out.data[hi + 2]).toBe(0xa5);
  });

  it("works on a real frame", async () => {
    const sheet = await loadImage(
      repoRoot("supporting-materials/frames/the-spriters-resource/Frames_USA.png"),
    );
    const frames = splitSheet(sheet, "Frames_USA");
    const frame = frames[0];
    const image = makeImage(128, 112, 0); // black image
    const out = composeFrame(image, frame, PALETTE);
    expect(out.width).toBe(frame.width);
    expect(out.height).toBe(frame.height);
    // Hole region should be the BK color #000000.
    const hi = (frame.holeY * frame.width + frame.holeX) * 4;
    expect(out.data[hi + 0]).toBe(0);
    expect(out.data[hi + 1]).toBe(0);
    expect(out.data[hi + 2]).toBe(0);
  });

  it("throws when image dimensions don't match the hole's 128x112", () => {
    const frame = makeFrame(160, 144, 16, 16, 0);
    const wrongImage = makeImage(64, 56, 82);
    expect(() => composeFrame(wrongImage, frame, PALETTE)).toThrow();
  });
});
```

- [ ] **Step 2: Verify tests fail**

Run: `pnpm --filter gbcam-extract test -- --run tests/frames/compose.test.ts`
Expected: failure — module not defined.

- [ ] **Step 3: Implement `composeFrame`**

Create `packages/gbcam-extract/src/frames/compose.ts`:

```ts
import type { GBImageData } from "../common.js";
import type { Frame } from "./types.js";
import { applyPalette } from "../palette.js";

const HOLE_W = 128;
const HOLE_H = 112;

/**
 * Compose a 128 × 112 grayscale image into a frame and render the result
 * with the given palette.
 *
 * Throws if the image is not 128 × 112 or if any input is malformed. The
 * caller is responsible for catching and falling back to a bare
 * `applyPalette(image, palette)` when the frame pipeline shouldn't crash
 * the UI.
 */
export function composeFrame(
  image: GBImageData,
  frame: Frame,
  palette: [string, string, string, string],
): GBImageData {
  if (image.width !== HOLE_W || image.height !== HOLE_H) {
    throw new Error(
      `composeFrame: expected ${HOLE_W}x${HOLE_H} image, got ${image.width}x${image.height}`,
    );
  }
  if (frame.pixels.length !== frame.width * frame.height) {
    throw new Error("composeFrame: frame.pixels length doesn't match dimensions");
  }

  // Build a temporary RGBA grayscale image at frame dimensions, then apply the
  // palette in one pass via the existing applyPalette logic.
  const W = frame.width;
  const H = frame.height;
  const gray = new Uint8ClampedArray(W * H * 4);

  // Fill with frame pixels.
  for (let i = 0; i < W * H; i++) {
    const v = frame.pixels[i];
    gray[i * 4 + 0] = v;
    gray[i * 4 + 1] = v;
    gray[i * 4 + 2] = v;
    gray[i * 4 + 3] = 255;
  }

  // Overwrite hole region with image grayscale values.
  for (let yy = 0; yy < HOLE_H; yy++) {
    for (let xx = 0; xx < HOLE_W; xx++) {
      const fi = ((frame.holeY + yy) * W + (frame.holeX + xx)) * 4;
      const ii = (yy * HOLE_W + xx) * 4;
      gray[fi + 0] = image.data[ii + 0];
      gray[fi + 1] = image.data[ii + 1];
      gray[fi + 2] = image.data[ii + 2];
      gray[fi + 3] = 255;
    }
  }

  return applyPalette({ data: gray, width: W, height: H }, palette);
}
```

- [ ] **Step 4: Run tests**

Run: `pnpm --filter gbcam-extract test -- --run tests/frames/compose.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/src/frames/compose.ts packages/gbcam-extract/tests/frames/compose.test.ts
git commit -m "Add composeFrame for image+frame+palette composition"
```

---

## Task 5 — `gbcam-extract`: barrel and public re-exports

**Files:**
- Create: `packages/gbcam-extract/src/frames/index.ts`
- Modify: `packages/gbcam-extract/src/index.ts`

- [ ] **Step 1: Create the barrel**

Create `packages/gbcam-extract/src/frames/index.ts`:

```ts
export type { Frame } from "./types.js";
export { splitSheet } from "./split-sheet.js";
export { dedupeFrames } from "./dedupe.js";
export { composeFrame } from "./compose.js";
```

- [ ] **Step 2: Re-export from `gbcam-extract/src/index.ts`**

Append to the existing exports section (after the existing palette re-exports):

```ts
export type { Frame } from "./frames/types.js";
export { splitSheet, dedupeFrames, composeFrame } from "./frames/index.js";
```

- [ ] **Step 3: Run all unit tests**

Run: `pnpm --filter gbcam-extract test`
Expected: ALL pass (existing + new frames tests).

- [ ] **Step 4: Typecheck**

Run from repo root: `pnpm typecheck`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add packages/gbcam-extract/src/frames/index.ts packages/gbcam-extract/src/index.ts
git commit -m "Re-export frame APIs from gbcam-extract"
```

---

## Task 6 — Web build script: `copy-frames.ts`

**Files:**
- Create: `packages/gbcam-extract-web/scripts/copy-frames.ts`
- Modify: `packages/gbcam-extract-web/package.json`
- Modify: `.gitignore` (root)

- [ ] **Step 1: Write the script**

Create `packages/gbcam-extract-web/scripts/copy-frames.ts`:

```ts
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const projectRoot = path.join(__dirname, "../../../");
const sourceRoot = path.join(projectRoot, "supporting-materials/frames");
const destPublic = path.join(__dirname, "../public/frames");
const destManifest = path.join(__dirname, "../src/generated/FrameSheets.ts");

const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp"]);

interface SheetEntry {
  url: string;
  stem: string;
}

function walk(dir: string, out: string[] = []): string[] {
  if (!fs.existsSync(dir)) return out;
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

function main() {
  if (!fs.existsSync(sourceRoot)) {
    console.warn(`[copy-frames] No frames at ${sourceRoot}; skipping.`);
    fs.mkdirSync(path.dirname(destManifest), { recursive: true });
    fs.writeFileSync(
      destManifest,
      `// auto-generated — do not edit\nexport interface FrameSheetEntry { url: string; stem: string; }\nexport const FRAME_SHEETS: ReadonlyArray<FrameSheetEntry> = [];\n`,
      "utf-8",
    );
    return;
  }

  fs.mkdirSync(destPublic, { recursive: true });
  fs.mkdirSync(path.dirname(destManifest), { recursive: true });

  const all = walk(sourceRoot);
  const entries: SheetEntry[] = [];
  for (const src of all) {
    const ext = path.extname(src).toLowerCase();
    if (!IMAGE_EXTS.has(ext)) continue;
    const rel = path.relative(sourceRoot, src);
    const dest = path.join(destPublic, rel);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(src, dest);
    const url = "./frames/" + rel.split(path.sep).join("/");
    const stem = path.basename(rel, ext);
    entries.push({ url, stem });
    console.log(`[copy-frames] ${src} -> ${dest}`);
  }

  // Stable order: alphabetical by stem so dedup tiebreaking is implicit.
  entries.sort((a, b) => a.stem.localeCompare(b.stem));

  const manifest = `// auto-generated by scripts/copy-frames.ts — do not edit
export interface FrameSheetEntry {
  /** URL the browser fetches (relative to the deployed root). */
  url: string;
  /** Stem of the source PNG (used as the frame ID prefix). */
  stem: string;
}

export const FRAME_SHEETS: ReadonlyArray<FrameSheetEntry> = ${JSON.stringify(entries, null, 2)};
`;
  fs.writeFileSync(destManifest, manifest, "utf-8");
  console.log(`[copy-frames] wrote ${entries.length} entries to ${destManifest}`);
}

main();
```

- [ ] **Step 2: Wire `package.json`**

Modify `packages/gbcam-extract-web/package.json`. Replace the `scripts` block with the version below (only the changed lines are highlighted; do not modify other fields):

```json
"scripts": {
  "build:instructions": "node scripts/generate-instructions.ts",
  "build:frames": "node scripts/copy-frames.ts",
  "dev": "pnpm build:instructions && pnpm build:frames && vite",
  "dev:host": "pnpm build:instructions && pnpm build:frames && vite --host",
  "build": "pnpm build:instructions && pnpm build:frames && tsc -b && vite build",
  "preview": "pnpm build:instructions && pnpm build:frames && vite preview",
  "preview:host": "pnpm build:instructions && pnpm build:frames && vite preview --host",
  "serve": "node scripts/serve-dist.ts",
  "typecheck": "tsc --noEmit",
  "postinstall": "node scripts/generate-instructions.ts && node scripts/generate-licenses.ts && node scripts/copy-frames.ts"
}
```

- [ ] **Step 3: Update root `.gitignore`**

Append at the end of `.gitignore`:

```
# Frame sheets copied into web/public and the generated manifest
packages/gbcam-extract-web/public/frames
packages/gbcam-extract-web/src/generated/FrameSheets.ts
```

- [ ] **Step 4: Run the script and verify outputs**

Run from repo root: `node packages/gbcam-extract-web/scripts/copy-frames.ts`
Expected: prints two `copy-frames` lines (one per source PNG) and the manifest path.

Verify outputs (use the Read tool to inspect):
- `packages/gbcam-extract-web/public/frames/the-spriters-resource/Frames_JPN.png` exists
- `packages/gbcam-extract-web/public/frames/the-spriters-resource/Frames_USA.png` exists
- `packages/gbcam-extract-web/src/generated/FrameSheets.ts` exists and contains both entries sorted alphabetically (`Frames_JPN` first)

- [ ] **Step 5: Confirm the gitignored files do not appear in git status**

Run: `git status --short`
Expected: no `public/frames` or `generated/FrameSheets.ts` lines.

- [ ] **Step 6: Commit**

```bash
git add packages/gbcam-extract-web/scripts/copy-frames.ts packages/gbcam-extract-web/package.json .gitignore
git commit -m "Add build script that copies frame sheets and emits FrameSheets manifest"
```

---

## Task 7 — Web data layer: `useFrameCatalog`

**Files:**
- Create: `packages/gbcam-extract-web/src/hooks/useFrameCatalog.ts`

- [ ] **Step 1: Create the hook**

```ts
// packages/gbcam-extract-web/src/hooks/useFrameCatalog.ts
import { useEffect, useState } from "react";
import type { Frame, GBImageData } from "gbcam-extract";
import { splitSheet, dedupeFrames } from "gbcam-extract";
import { FRAME_SHEETS } from "../generated/FrameSheets.js";

export type FrameCatalogStatus = "loading" | "ready" | "error";

export interface FrameCatalog {
  status: FrameCatalogStatus;
  frames: Frame[];
  /** Map id -> Frame for O(1) lookup. */
  getFrameById(id: string): Frame | undefined;
  error?: string;
}

let cached: { frames: Frame[]; byId: Map<string, Frame> } | null = null;
let pending: Promise<{ frames: Frame[]; byId: Map<string, Frame> }> | null = null;

async function fetchSheet(url: string): Promise<GBImageData> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error(`Failed to decode ${url}`));
      el.src = objectUrl;
    });
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2d context unavailable");
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    return { data: imageData.data, width: img.width, height: img.height };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function buildCatalog(): Promise<{ frames: Frame[]; byId: Map<string, Frame> }> {
  const all: Frame[] = [];
  for (const entry of FRAME_SHEETS) {
    const sheet = await fetchSheet(entry.url);
    all.push(...splitSheet(sheet, entry.stem));
  }
  const frames = dedupeFrames(all);
  const byId = new Map(frames.map((f) => [f.id, f] as const));
  return { frames, byId };
}

export function useFrameCatalog(): FrameCatalog {
  const [state, setState] = useState<{
    status: FrameCatalogStatus;
    frames: Frame[];
    byId: Map<string, Frame>;
    error?: string;
  }>(() =>
    cached
      ? { status: "ready", frames: cached.frames, byId: cached.byId }
      : { status: "loading", frames: [], byId: new Map() },
  );

  useEffect(() => {
    if (cached) return;
    let mounted = true;
    if (!pending) pending = buildCatalog();
    pending
      .then((result) => {
        cached = result;
        if (mounted) {
          setState({
            status: "ready",
            frames: result.frames,
            byId: result.byId,
          });
        }
      })
      .catch((err) => {
        pending = null;
        if (mounted) {
          setState({
            status: "error",
            frames: [],
            byId: new Map(),
            error: err instanceof Error ? err.message : String(err),
          });
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  return {
    status: state.status,
    frames: state.frames,
    error: state.error,
    getFrameById: (id) => state.byId.get(id),
  };
}
```

- [ ] **Step 2: Typecheck**

Run from repo root: `pnpm typecheck`
Expected: success. If `gbcam-extract` is not yet rebuilt, the IDE may show stale type errors — they will resolve after `pnpm build` in Task 12. Typecheck against current source via `pnpm typecheck` (which runs `tsc --noEmit`) should still succeed because the package's source is referenced.

If typecheck fails because `gbcam-extract` types aren't seen by `gbcam-extract-web`'s `tsc`, build the dependency first: `pnpm --filter gbcam-extract build`.

- [ ] **Step 3: Commit**

```bash
git add packages/gbcam-extract-web/src/hooks/useFrameCatalog.ts
git commit -m "Add useFrameCatalog hook"
```

---

## Task 8 — `AppSettings.defaultFrame` and `ProcessingResult.frameOverride`

**Files:**
- Modify: `packages/gbcam-extract-web/src/hooks/useAppSettings.ts`
- Modify: `packages/gbcam-extract-web/src/hooks/useProcessing.ts`
- Modify: `packages/gbcam-extract-web/src/hooks/useImageHistory.ts`
- Create: `packages/gbcam-extract-web/src/types/frame-selection.ts`

- [ ] **Step 1: Create the shared `FrameSelection` type**

```ts
// packages/gbcam-extract-web/src/types/frame-selection.ts
/**
 * A frame choice. The discriminated union keeps the per-result override
 * semantics clean: "default" means follow the global default, "none" is
 * an explicit "no frame" override, and "frame" pins a specific frame ID.
 *
 * Persisted shape — only `id` is stored, never pixel data.
 */
export type FrameSelection =
  | { kind: "default" }
  | { kind: "none" }
  | { kind: "frame"; id: string };

/** Default for new per-result overrides. */
export const FRAME_SELECTION_DEFAULT: FrameSelection = { kind: "default" };

/** Default for the global default-frame setting (no global frame). */
export const FRAME_SELECTION_NONE: FrameSelection = { kind: "none" };
```

- [ ] **Step 2: Extend `AppSettings`**

Modify `packages/gbcam-extract-web/src/hooks/useAppSettings.ts`. Add a `defaultFrame` field; default to `FRAME_SELECTION_NONE`. The full file should read:

```ts
import { useCallback } from "react";
import { useLocalStorage } from "./useLocalStorage.js";
import type { PaletteEntry } from "../data/palettes.js";
import {
  type FrameSelection,
  FRAME_SELECTION_NONE,
} from "../types/frame-selection.js";

const STORAGE_KEY = "gbcam-app-settings";

export interface AppSettings {
  debug: boolean;
  clipboardEnabled: boolean;
  outputScale: number;
  previewScale: number;
  paletteSelection?: PaletteEntry;
  defaultFrame?: FrameSelection;
}

const DEFAULTS: AppSettings = {
  debug: false,
  clipboardEnabled: false,
  outputScale: 1,
  previewScale: 2,
  defaultFrame: FRAME_SELECTION_NONE,
};

export function useAppSettings() {
  const [settings, setSettings] = useLocalStorage<AppSettings>(
    STORAGE_KEY,
    DEFAULTS,
  );

  const updateSetting = useCallback(
    <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
      setSettings((prev) => ({ ...prev, [key]: value }));
    },
    [setSettings],
  );

  return { settings: { ...DEFAULTS, ...settings }, updateSetting };
}
```

- [ ] **Step 3: Extend `ProcessingResult` in `useProcessing.ts`**

Add a `frameOverride?: FrameSelection` field. The `...item` spread in
`loadResultsFromStorage` already round-trips unknown fields, so old data
without this field deserializes as `frameOverride: undefined` — which the
UI treats as `{ kind: "default" }`.

Modify the imports and `ProcessingResult`:

```ts
// near the existing imports in packages/gbcam-extract-web/src/hooks/useProcessing.ts
import type { FrameSelection } from "../types/frame-selection.js";
```

Replace the existing `ProcessingResult` interface:

```ts
export interface ProcessingResult {
  result: PipelineResult;
  filename: string;
  processingTime: number;
  /** Per-image frame override. Undefined = follow global default. */
  frameOverride?: FrameSelection;
}
```

No other changes are needed in this file — `saveResultsToStorage` and
`loadResultsFromStorage` use spread-and-override pattern that preserves
extra properties.

- [ ] **Step 4: Mirror in `useImageHistory.ts`**

Add the same import and rely on the existing `ProcessingResult` re-import
already used by `useImageHistory.ts`. Confirm the file already imports
`ProcessingResult` from `useProcessing.ts` (it does, line 10). No code
changes are needed in `useImageHistory.ts` itself — the field flows
through via the shared type.

Verify by re-reading the current import at the top of
`useImageHistory.ts`. If the file does not currently import
`ProcessingResult`, add:

```ts
import type { ProcessingResult } from "./useProcessing.js";
```

- [ ] **Step 5: Typecheck**

Run from repo root: `pnpm typecheck`
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add packages/gbcam-extract-web/src/types/frame-selection.ts \
        packages/gbcam-extract-web/src/hooks/useAppSettings.ts \
        packages/gbcam-extract-web/src/hooks/useProcessing.ts \
        packages/gbcam-extract-web/src/hooks/useImageHistory.ts
git commit -m "Add FrameSelection type, defaultFrame setting, and per-result frameOverride"
```

---

## Task 9 — `<FramePicker>` component

**Files:**
- Create: `packages/gbcam-extract-web/src/components/FramePicker.tsx`

- [ ] **Step 1: Confirm shadcn primitives are installed**

Read `packages/gbcam-extract-web/src/shadcn/components/popover.tsx` to
confirm it exists. If it does not, install it:

```bash
cd packages/gbcam-extract-web && pnpm shadcn add popover
```

(Each shadcn add gets its own commit; if you ran the install, commit those
changes before proceeding.)

- [ ] **Step 2: Write the component**

Create `packages/gbcam-extract-web/src/components/FramePicker.tsx`:

```tsx
import { useEffect, useMemo, useRef } from "react";
import type { Frame } from "gbcam-extract";
import { composeFrame, applyPalette } from "gbcam-extract";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/shadcn/components/popover";
import { Button } from "@/shadcn/components/button";
import { ChevronDown, Frame as FrameIcon } from "lucide-react";
import { cn } from "@/shadcn/utils/utils";
import type { FrameSelection } from "../types/frame-selection.js";

const HOLE_W = 128;
const HOLE_H = 112;

interface FramePickerProps {
  value: FrameSelection;
  onChange: (next: FrameSelection) => void;
  palette: [string, string, string, string];
  frames: Frame[];
  /** "result" includes a "Default — …" tile; "default" omits it. */
  mode: "default" | "result";
  /** Display label for the global default (used in "result" mode). */
  defaultFrameLabel?: string;
  disabled?: boolean;
}

/** Build a dummy 128×112 lightest-color image for picker thumbnails. */
function buildEmptyImage(): { data: Uint8ClampedArray; width: number; height: number } {
  const data = new Uint8ClampedArray(HOLE_W * HOLE_H * 4);
  for (let i = 0; i < HOLE_W * HOLE_H; i++) {
    data[i * 4 + 0] = 255;
    data[i * 4 + 1] = 255;
    data[i * 4 + 2] = 255;
    data[i * 4 + 3] = 255;
  }
  return { data, width: HOLE_W, height: HOLE_H };
}

const EMPTY_IMAGE = buildEmptyImage();

/** Render a frame (or solid lightest color when no frame) onto a canvas. */
function FrameCanvas({
  frame,
  palette,
  width,
  height,
  className,
}: {
  frame: Frame | null;
  palette: [string, string, string, string];
  width: number;
  height: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    let rendered;
    if (frame) {
      try {
        rendered = composeFrame(EMPTY_IMAGE, frame, palette);
      } catch {
        rendered = applyPalette(EMPTY_IMAGE, palette);
      }
    } else {
      rendered = applyPalette(EMPTY_IMAGE, palette);
    }
    const tmp = document.createElement("canvas");
    tmp.width = rendered.width;
    tmp.height = rendered.height;
    tmp
      .getContext("2d")!
      .putImageData(
        new ImageData(new Uint8ClampedArray(rendered.data), rendered.width, rendered.height),
        0,
        0,
      );
    ctx.drawImage(tmp, 0, 0, width, height);
  }, [frame, palette, width, height]);
  return <canvas ref={ref} className={className} style={{ imageRendering: "pixelated" }} />;
}

function frameDisplayName(frame: Frame): string {
  return `${frame.sheetStem} — ${frame.type} #${frame.index}`;
}

function selectionLabel(
  value: FrameSelection,
  framesById: Map<string, Frame>,
  defaultLabel: string | undefined,
): string {
  if (value.kind === "default") return `Default${defaultLabel ? ` — ${defaultLabel}` : ""}`;
  if (value.kind === "none") return "No frame";
  const f = framesById.get(value.id);
  return f ? frameDisplayName(f) : value.id;
}

export function FramePicker({
  value,
  onChange,
  palette,
  frames,
  mode,
  defaultFrameLabel,
  disabled,
}: FramePickerProps) {
  const framesById = useMemo(() => new Map(frames.map((f) => [f.id, f] as const)), [frames]);
  const triggerFrame: Frame | null =
    value.kind === "frame" ? framesById.get(value.id) ?? null : null;
  const triggerLabel = selectionLabel(value, framesById, defaultFrameLabel);

  const normals = useMemo(() => frames.filter((f) => f.type === "normal"), [frames]);
  const wilds = useMemo(() => frames.filter((f) => f.type === "wild"), [frames]);

  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button variant="secondary" disabled={disabled} className="gap-2">
            {triggerFrame || value.kind === "default" || value.kind === "none" ? (
              <span className="inline-flex size-4 items-center justify-center overflow-hidden rounded border border-border">
                {triggerFrame ? (
                  <FrameCanvas
                    frame={triggerFrame}
                    palette={palette}
                    width={16}
                    height={16}
                    className="size-4"
                  />
                ) : (
                  <FrameIcon className="size-3 text-muted-foreground" />
                )}
              </span>
            ) : null}
            <span className="truncate max-w-[12em]">{triggerLabel}</span>
            <ChevronDown data-icon="inline-end" />
          </Button>
        }
      />
      <PopoverContent className="w-[min(90vw,640px)] max-h-[70vh] overflow-auto p-3">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {mode === "result" && (
            <FrameTile
              label={`Default${defaultFrameLabel ? ` — ${defaultFrameLabel}` : ""}`}
              selected={value.kind === "default"}
              onClick={() => onChange({ kind: "default" })}
              palette={palette}
              frame={null}
              previewW={160}
              previewH={144}
            />
          )}
          <FrameTile
            label="No frame"
            selected={value.kind === "none"}
            onClick={() => onChange({ kind: "none" })}
            palette={palette}
            frame={null}
            previewW={160}
            previewH={144}
          />
        </div>
        {normals.length > 0 && (
          <>
            <h4 className="mt-3 mb-2 text-sm font-semibold">Normal frames</h4>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {normals.map((f) => (
                <FrameTile
                  key={f.id}
                  label={frameDisplayName(f)}
                  selected={value.kind === "frame" && value.id === f.id}
                  onClick={() => onChange({ kind: "frame", id: f.id })}
                  palette={palette}
                  frame={f}
                  previewW={160}
                  previewH={144}
                />
              ))}
            </div>
          </>
        )}
        {wilds.length > 0 && (
          <>
            <h4 className="mt-3 mb-2 text-sm font-semibold">Wild frames</h4>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {wilds.map((f) => (
                <FrameTile
                  key={f.id}
                  label={frameDisplayName(f)}
                  selected={value.kind === "frame" && value.id === f.id}
                  onClick={() => onChange({ kind: "frame", id: f.id })}
                  palette={palette}
                  frame={f}
                  previewW={f.width}
                  previewH={f.height}
                />
              ))}
            </div>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}

function FrameTile({
  label,
  selected,
  onClick,
  palette,
  frame,
  previewW,
  previewH,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
  palette: [string, string, string, string];
  frame: Frame | null;
  previewW: number;
  previewH: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-center gap-1 rounded border bg-card p-2 text-xs hover:bg-accent",
        selected && "ring-2 ring-primary",
      )}
    >
      <FrameCanvas
        frame={frame}
        palette={palette}
        width={previewW}
        height={previewH}
        className="max-w-full h-auto rounded border border-border"
      />
      <span className="truncate w-full text-center">{label}</span>
    </button>
  );
}
```

- [ ] **Step 3: Typecheck**

Run from repo root: `pnpm typecheck`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add packages/gbcam-extract-web/src/components/FramePicker.tsx
git commit -m "Add FramePicker component"
```

---

## Task 10 — `ResultCard`: layout swap + frame composition

**Files:**
- Modify: `packages/gbcam-extract-web/src/components/ResultCard.tsx`

- [ ] **Step 1: Rewrite `ResultCard` with the new props and layout**

Replace the contents of `packages/gbcam-extract-web/src/components/ResultCard.tsx`:

```tsx
import { useRef, useEffect, useState, useCallback } from "react";
import type { PipelineResult, Frame } from "gbcam-extract";
import { applyPalette, composeFrame } from "gbcam-extract";
import {
  canShare,
  shareImage,
  copyImageToClipboard,
} from "../utils/shareImage.js";
import { sanitizePaletteName } from "../utils/filenames.js";
import { Button } from "@/shadcn/components/button";
import { Badge } from "@/shadcn/components/badge";
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
} from "@/shadcn/components/card";
import { toast } from "sonner";
import { X, Download, Share2, Copy as CopyIcon } from "lucide-react";
import { FramePicker } from "./FramePicker.js";
import type { FrameSelection } from "../types/frame-selection.js";

interface ResultCardProps {
  result: PipelineResult;
  filename: string;
  processingTime: number;
  palette: [string, string, string, string];
  paletteName: string;
  outputScale?: number;
  previewScale?: number;
  /** All frames available in the catalog (for the picker). */
  frames: Frame[];
  /** The frame selection chosen on this result. Undefined = follow default. */
  frameOverride: FrameSelection;
  onFrameOverrideChange: (next: FrameSelection) => void;
  /** Already resolved (effective) frame to render — null = no frame. */
  effectiveFrame: Frame | null;
  /** Display label for the "Default — …" picker tile. */
  defaultFrameLabel: string;
  onDelete?: () => void;
}

/** Build an off-screen canvas at the given scale for download/share/copy. */
function buildOutputCanvas(
  result: PipelineResult,
  palette: [string, string, string, string],
  effectiveFrame: Frame | null,
  scale: number,
): HTMLCanvasElement | null {
  try {
    if (!result.grayscale?.data) return null;
    let rendered;
    if (effectiveFrame) {
      try {
        rendered = composeFrame(result.grayscale, effectiveFrame, palette);
      } catch {
        rendered = applyPalette(result.grayscale, palette);
      }
    } else {
      rendered = applyPalette(result.grayscale, palette);
    }
    if (!rendered?.data?.length) return null;

    const canvas = document.createElement("canvas");
    canvas.width = rendered.width * scale;
    canvas.height = rendered.height * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;

    const tmp = document.createElement("canvas");
    tmp.width = rendered.width;
    tmp.height = rendered.height;
    tmp
      .getContext("2d")!
      .putImageData(
        new ImageData(
          new Uint8ClampedArray(rendered.data),
          rendered.width,
          rendered.height,
        ),
        0,
        0,
      );
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    return canvas;
  } catch {
    return null;
  }
}

export function ResultCard({
  result,
  filename,
  processingTime,
  palette,
  paletteName,
  outputScale = 1,
  previewScale = 2,
  frames,
  frameOverride,
  onFrameOverrideChange,
  effectiveFrame,
  defaultFrameLabel,
  onDelete,
}: ResultCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [shareSupported, setShareSupported] = useState(false);

  useEffect(() => {
    setShareSupported(canShare());
  }, []);

  // Render preview at previewScale, applying the effective frame if any.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      if (!result.grayscale?.data) return;
      let rendered;
      if (effectiveFrame) {
        try {
          rendered = composeFrame(result.grayscale, effectiveFrame, palette);
        } catch (err) {
          console.error("composeFrame failed; falling back to bare image", err);
          rendered = applyPalette(result.grayscale, palette);
        }
      } else {
        rendered = applyPalette(result.grayscale, palette);
      }
      if (!rendered?.data?.length) return;

      const scale = previewScale;
      canvas.width = rendered.width * scale;
      canvas.height = rendered.height * scale;
      const ctx = canvas.getContext("2d")!;
      ctx.imageSmoothingEnabled = false;

      const tmp = document.createElement("canvas");
      tmp.width = rendered.width;
      tmp.height = rendered.height;
      tmp
        .getContext("2d")!
        .putImageData(
          new ImageData(
            new Uint8ClampedArray(rendered.data),
            rendered.width,
            rendered.height,
          ),
          0,
          0,
        );
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    } catch (err) {
      console.error("Error rendering image:", err);
    }
  }, [result, palette, previewScale, effectiveFrame]);

  const handleDownload = useCallback(() => {
    const outputCanvas = buildOutputCanvas(result, palette, effectiveFrame, outputScale);
    if (!outputCanvas) return;
    const basename = filename.replace(/\.[^.]+$/, "");
    const sanitized = sanitizePaletteName(paletteName);
    const link = document.createElement("a");
    link.download = sanitized
      ? `${basename}_${sanitized}_gb.png`
      : `${basename}_gb.png`;
    link.href = outputCanvas.toDataURL("image/png");
    link.click();
  }, [result, palette, effectiveFrame, outputScale, filename, paletteName]);

  const handleShare = useCallback(async () => {
    const outputCanvas = buildOutputCanvas(result, palette, effectiveFrame, outputScale);
    if (!outputCanvas) return;
    try {
      await shareImage(
        outputCanvas,
        filename.replace(/\.[^.]+$/, "") + "_gb.png",
      );
    } catch (err) {
      console.error("Failed to share image:", err);
    }
  }, [result, palette, effectiveFrame, outputScale, filename]);

  const handleCopy = useCallback(async () => {
    const outputCanvas = buildOutputCanvas(result, palette, effectiveFrame, outputScale);
    if (!outputCanvas) return;
    try {
      await copyImageToClipboard(outputCanvas);
      toast.success("Image copied to clipboard");
    } catch (err) {
      const errorMsg = (err as Error).message || "Failed to copy";
      toast.error(`Copy failed: ${errorMsg}`);
      console.error("Failed to copy image:", err);
    }
  }, [result, palette, effectiveFrame, outputScale]);

  return (
    <Card className="p-3 sm:p-4">
      <CardHeader className="p-0 mb-3">
        <div className="min-w-0">
          <p className="text-sm font-medium truncate" title={filename}>
            {filename}
          </p>
          <Badge variant="secondary" className="mt-0.5">
            {processingTime.toFixed(0)}ms
          </Badge>
        </div>
        {onDelete && (
          <CardAction>
            <Button
              variant="destructive"
              size="icon"
              onClick={onDelete}
              aria-label="Delete result"
              className="size-7"
            >
              <X />
            </Button>
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="flex flex-col sm:flex-row gap-3 p-0">
        <div className="flex flex-col gap-2 items-start order-2 sm:order-1">
          <div className="flex flex-wrap gap-2 items-start content-start">
            <Button onClick={handleDownload}>
              <Download data-icon="inline-start" />
              Download PNG
            </Button>
            {shareSupported && (
              <Button variant="secondary" onClick={handleShare}>
                <Share2 data-icon="inline-start" />
                Share
              </Button>
            )}
            <Button variant="secondary" onClick={handleCopy} aria-label="Copy image">
              <CopyIcon data-icon="inline-start" />
              Copy
            </Button>
          </div>
          <FramePicker
            value={frameOverride}
            onChange={onFrameOverrideChange}
            palette={palette}
            frames={frames}
            mode="result"
            defaultFrameLabel={defaultFrameLabel}
          />
        </div>
        <canvas
          ref={canvasRef}
          className="rounded border self-start order-1 sm:order-2"
          style={{ imageRendering: "pixelated", maxWidth: "100%" }}
        />
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Typecheck**

Run from repo root: `pnpm typecheck`
Expected: success against current call sites in `App.tsx`. Existing
`App.tsx` will fail typecheck because it doesn't pass the new
required props yet — that is fixed in Task 11. **Skip** this typecheck
step's "expected pass" if you are running tasks in order; just verify
the *file you just wrote* compiles in isolation (i.e. errors only point
to call sites in `App.tsx`).

- [ ] **Step 3: Commit (will fail compile until Task 11)**

```bash
git add packages/gbcam-extract-web/src/components/ResultCard.tsx
git commit -m "Refactor ResultCard for frame composition and add per-result picker"
```

---

## Task 11 — `App.tsx` wiring

**Files:**
- Modify: `packages/gbcam-extract-web/src/App.tsx`

- [ ] **Step 1: Wire the catalog hook, default-frame picker, per-result resolution, and updated `ResultCard` calls**

In `packages/gbcam-extract-web/src/App.tsx`:

Add imports near the top (alongside existing imports):

```ts
import { useFrameCatalog } from "./hooks/useFrameCatalog.js";
import { FramePicker } from "./components/FramePicker.js";
import {
  type FrameSelection,
  FRAME_SELECTION_DEFAULT,
  FRAME_SELECTION_NONE,
} from "./types/frame-selection.js";
import type { Frame } from "gbcam-extract";
```

Inside `App()`, after the existing `paletteEntry` line, add:

```ts
const catalog = useFrameCatalog();
const framesById = catalog.frames;
const defaultFrame: FrameSelection =
  settings.defaultFrame ?? FRAME_SELECTION_NONE;

function setDefaultFrame(next: FrameSelection) {
  // Global default may not itself be "default".
  if (next.kind === "default") return;
  updateSetting("defaultFrame", next);
}

function frameLabelFor(selection: FrameSelection): string {
  if (selection.kind === "none") return "No frame";
  if (selection.kind === "default") return "Default";
  const f = catalog.getFrameById(selection.id);
  return f ? `${f.sheetStem} — ${f.type} #${f.index}` : selection.id;
}

function resolveEffective(override: FrameSelection): Frame | null {
  const effective = override.kind === "default" ? defaultFrame : override;
  if (effective.kind === "frame") {
    return catalog.getFrameById(effective.id) ?? null;
  }
  return null;
}

const defaultFrameLabel = frameLabelFor(defaultFrame);
```

Add a per-result override state map (lifted from the cards), keyed by
filename — this matches the existing `handleDeleteResult` pattern:

```ts
function setResultFrameOverride(filename: string, next: FrameSelection) {
  setCurrentResults((prev) =>
    prev.map((r) =>
      r.filename === filename ? { ...r, frameOverride: next } : r,
    ),
  );
}

function setHistoryFrameOverride(
  batchId: string,
  resultIndex: number,
  next: FrameSelection,
) {
  // History updates require modifying the parent batch in place. We dispatch
  // through useImageHistory's setHistory by piggy-backing on existing state
  // — implemented inline by wrapping `archiveResults`/`deleteFromHistory`
  // is unnecessary; we add a small helper to useImageHistory in this task.
  updateHistoryFrameOverride(batchId, resultIndex, next);
}
```

Then add a new helper to `useImageHistory.ts` so history overrides
persist:

```ts
// At the bottom of useImageHistory.ts, before the return statement.
const updateFrameOverride = useCallback(
  (batchId: string, resultIndex: number, override: FrameSelection) => {
    setHistory((prev) =>
      prev.map((batch) =>
        batch.id === batchId
          ? {
              ...batch,
              results: batch.results.map((r, i) =>
                i === resultIndex ? { ...r, frameOverride: override } : r,
              ),
            }
          : batch,
      ),
    );
  },
  [],
);
```

…and add `updateFrameOverride` to the returned object.

Add `import type { FrameSelection } from "../types/frame-selection.js";`
at the top of `useImageHistory.ts`.

Back in `App.tsx`, destructure `updateFrameOverride` from
`useImageHistory()`:

```ts
const {
  history,
  isHistoryExpanded,
  setIsHistoryExpanded,
  archiveResults,
  deleteFromHistory,
  deleteBatch,
  deleteAllHistory,
  updateSettings: updateHistorySettings,
  settings: historySettings,
  updateFrameOverride: updateHistoryFrameOverride,
} = useImageHistory();
```

In the results-bar `<div className="mb-4 flex flex-wrap …">`, append a
new field after Preview Scale:

```tsx
<Field orientation="horizontal" className="w-auto gap-2">
  <FieldLabel>Default Frame:</FieldLabel>
  <FramePicker
    value={defaultFrame}
    onChange={setDefaultFrame}
    palette={paletteEntry.colors}
    frames={catalog.frames}
    mode="default"
    disabled={catalog.status !== "ready"}
  />
</Field>
```

Replace each `<ResultCard …>` call with the expanded prop set:

```tsx
{results.map((r) => (
  <div key={r.filename}>
    <ResultCard
      result={r.result}
      filename={r.filename}
      processingTime={r.processingTime}
      palette={paletteEntry.colors}
      paletteName={paletteEntry.name}
      outputScale={outputScale}
      previewScale={previewScale}
      frames={catalog.frames}
      frameOverride={r.frameOverride ?? FRAME_SELECTION_DEFAULT}
      onFrameOverrideChange={(next) => setResultFrameOverride(r.filename, next)}
      effectiveFrame={resolveEffective(r.frameOverride ?? FRAME_SELECTION_DEFAULT)}
      defaultFrameLabel={defaultFrameLabel}
      onDelete={() => handleDeleteResult(r.filename)}
    />
    {(r.result.intermediates || r.result.debug) && (
      <PipelineDebugViewer
        intermediates={r.result.intermediates}
        debug={r.result.debug}
      />
    )}
  </div>
))}
```

Replace each history `<ResultCard …>` similarly:

```tsx
{batch.results.map((result, idx) => (
  <ResultCard
    key={`${batch.id}-${idx}`}
    result={result.result}
    filename={result.filename}
    processingTime={result.processingTime}
    palette={paletteEntry.colors}
    paletteName={paletteEntry.name}
    outputScale={outputScale}
    previewScale={previewScale}
    frames={catalog.frames}
    frameOverride={result.frameOverride ?? FRAME_SELECTION_DEFAULT}
    onFrameOverrideChange={(next) =>
      updateHistoryFrameOverride(batch.id, idx, next)
    }
    effectiveFrame={resolveEffective(result.frameOverride ?? FRAME_SELECTION_DEFAULT)}
    defaultFrameLabel={defaultFrameLabel}
    onDelete={() => deleteFromHistory(batch.id, idx)}
  />
))}
```

Also delete the now-unused `downloadResult` helper at the bottom of
`App.tsx` and the corresponding `Download All` button's `forEach` body
needs updating to also pass `effectiveFrame` and use `composeFrame`. Inline
the logic by reusing `buildOutputCanvas` — but `buildOutputCanvas` is
defined inside `ResultCard.tsx`. The simplest fix is to extract it as a
shared util. Add a utility module:

Create `packages/gbcam-extract-web/src/utils/buildOutputCanvas.ts`:

```ts
import type { PipelineResult, Frame } from "gbcam-extract";
import { applyPalette, composeFrame } from "gbcam-extract";

export function buildOutputCanvas(
  result: PipelineResult,
  palette: [string, string, string, string],
  effectiveFrame: Frame | null,
  scale: number,
): HTMLCanvasElement | null {
  try {
    if (!result.grayscale?.data) return null;
    let rendered;
    if (effectiveFrame) {
      try {
        rendered = composeFrame(result.grayscale, effectiveFrame, palette);
      } catch {
        rendered = applyPalette(result.grayscale, palette);
      }
    } else {
      rendered = applyPalette(result.grayscale, palette);
    }
    if (!rendered?.data?.length) return null;

    const canvas = document.createElement("canvas");
    canvas.width = rendered.width * scale;
    canvas.height = rendered.height * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;

    const tmp = document.createElement("canvas");
    tmp.width = rendered.width;
    tmp.height = rendered.height;
    tmp
      .getContext("2d")!
      .putImageData(
        new ImageData(
          new Uint8ClampedArray(rendered.data),
          rendered.width,
          rendered.height,
        ),
        0,
        0,
      );
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    return canvas;
  } catch {
    return null;
  }
}
```

Update `ResultCard.tsx` to import this helper and remove the local copy:

```ts
import { buildOutputCanvas } from "../utils/buildOutputCanvas.js";
```

…and delete the local `buildOutputCanvas` function definition.

Update the `Download All (…)` button click handler in `App.tsx` to:

```tsx
<Button
  onClick={() => {
    results.forEach((r) => {
      const override = r.frameOverride ?? FRAME_SELECTION_DEFAULT;
      const effective = resolveEffective(override);
      const canvas = buildOutputCanvas(
        r.result,
        paletteEntry.colors,
        effective,
        outputScale,
      );
      if (!canvas) return;
      const baseName = r.filename.replace(/\.[^.]+$/, "");
      const sanitizedPaletteName = sanitizePaletteName(paletteEntry.name);
      const link = document.createElement("a");
      link.download = `${baseName}_${sanitizedPaletteName}_gb.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    });
  }}
>
  Download All ({results.length})
</Button>
```

Add the import at the top of `App.tsx`:

```ts
import { buildOutputCanvas } from "./utils/buildOutputCanvas.js";
```

Delete the `downloadResult` helper at the bottom of `App.tsx` entirely.

- [ ] **Step 2: Typecheck**

Run from repo root: `pnpm typecheck`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add packages/gbcam-extract-web/src/App.tsx \
        packages/gbcam-extract-web/src/hooks/useImageHistory.ts \
        packages/gbcam-extract-web/src/utils/buildOutputCanvas.ts \
        packages/gbcam-extract-web/src/components/ResultCard.tsx
git commit -m "Wire FramePicker + default frame setting + per-result override into App"
```

---

## Task 12 — Final verification

- [ ] **Step 1: Run all unit tests**

Run from repo root: `pnpm test`
Expected: all tests pass (existing + new frame tests).

- [ ] **Step 2: Typecheck across all packages**

Run from repo root: `pnpm typecheck`
Expected: success.

- [ ] **Step 3: Build all packages**

Run from repo root: `pnpm build`
Expected: success. The `web` build runs `copy-frames` + `tsc -b` + `vite build`.

- [ ] **Step 4: Smoke-test the dev server (optional but strongly preferred)**

Run from repo root: `pnpm dev`
Expected: the dev server starts without errors. The website serves
`/frames/the-spriters-resource/Frames_USA.png` and
`/frames/the-spriters-resource/Frames_JPN.png`. The Default Frame
picker appears in the results bar after processing an image, and
selecting a frame causes the preview, downloaded PNG, and copied image
to be wrapped in the chosen frame at native dimensions.

If the smoke test reveals a real bug, file the bug and (if obvious) fix
it before declaring this task complete.

- [ ] **Step 5: Final commit (if step 4 yielded any fixes)**

If step 4 caused changes, commit them with a `Fix: …` style message.
Otherwise skip.

---

## Self-review

**Spec coverage:**

| Spec section | Plan coverage |
|---|---|
| File location & gitignore | Task 6 (build script + .gitignore) |
| Frame splitter algorithm | Task 2 |
| Dedup with alphabetical tiebreak | Task 3 |
| Frame composer (image + frame + palette) | Task 4 |
| Composer failure → fall back to image | Task 10/11 (call sites) |
| Frame URL list generated, not hardcoded | Task 6 (`FrameSheets.ts` emitted) |
| Frame catalog hook | Task 7 |
| Selection persists frame ID only | Task 8 (`FrameSelection` shape) |
| Backward-compat localStorage | Task 8 (`...item` spread + `?? FRAME_SELECTION_DEFAULT`) |
| FramePicker component (popover, normal/wild, current palette) | Task 9 |
| Default frame setting near Preview Scale | Task 11 |
| Per-result picker under buttons, left of canvas (desktop) | Task 10 (`order-1`/`order-2`) |
| Preview / download / share / copy use the frame | Task 10 (`buildOutputCanvas`) |
| Tests in gbcam-extract | Tasks 2–4 |

**Placeholder scan:** none.

**Type consistency:** `FrameSelection` matches across types module, AppSettings, ProcessingResult, FramePicker props, App.tsx wiring. Frame is a single shape used everywhere from `gbcam-extract`. `composeFrame` signature stable across uses.

**Plan complete.**
