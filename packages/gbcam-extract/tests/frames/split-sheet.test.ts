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
    expect(summary).toMatchInlineSnapshot(`
      {
        "normal": 18,
        "shapes": [
          "Frames_USA:normal:1 160x144@16,16",
          "Frames_USA:normal:2 160x144@16,16",
          "Frames_USA:normal:3 160x144@16,16",
          "Frames_USA:normal:4 160x144@16,16",
          "Frames_USA:normal:5 160x144@16,16",
          "Frames_USA:normal:6 160x144@16,16",
          "Frames_USA:normal:7 160x144@16,16",
          "Frames_USA:normal:8 160x144@16,16",
          "Frames_USA:normal:9 160x144@16,16",
          "Frames_USA:normal:10 160x144@16,16",
          "Frames_USA:normal:11 160x144@16,16",
          "Frames_USA:normal:12 160x144@16,16",
          "Frames_USA:normal:13 160x144@16,16",
          "Frames_USA:normal:14 160x144@16,16",
          "Frames_USA:normal:15 160x144@16,16",
          "Frames_USA:normal:16 160x144@16,16",
          "Frames_USA:normal:17 160x144@16,16",
          "Frames_USA:normal:18 160x144@16,16",
          "Frames_USA:wild:1 160x224@16,40",
          "Frames_USA:wild:2 160x224@16,40",
          "Frames_USA:wild:3 160x224@16,40",
          "Frames_USA:wild:4 160x224@16,40",
          "Frames_USA:wild:5 160x224@16,40",
          "Frames_USA:wild:6 160x224@16,40",
          "Frames_USA:wild:7 160x224@16,40",
        ],
        "total": 25,
        "wild": 7,
      }
    `);
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
    expect(summary).toMatchInlineSnapshot(`
      {
        "normal": 18,
        "shapes": [
          "Frames_JPN:normal:1 160x144@16,16",
          "Frames_JPN:normal:2 160x144@16,16",
          "Frames_JPN:normal:3 160x144@16,16",
          "Frames_JPN:normal:4 160x144@16,16",
          "Frames_JPN:normal:5 160x144@16,16",
          "Frames_JPN:normal:6 160x144@16,16",
          "Frames_JPN:normal:7 160x144@16,16",
          "Frames_JPN:normal:8 160x144@16,16",
          "Frames_JPN:normal:9 160x144@16,16",
          "Frames_JPN:normal:10 160x144@16,16",
          "Frames_JPN:normal:11 160x144@16,16",
          "Frames_JPN:normal:12 160x144@16,16",
          "Frames_JPN:normal:13 160x144@16,16",
          "Frames_JPN:normal:14 160x144@16,16",
          "Frames_JPN:normal:15 160x144@16,16",
          "Frames_JPN:normal:16 160x144@16,16",
          "Frames_JPN:normal:17 160x144@16,16",
          "Frames_JPN:normal:18 160x144@16,16",
          "Frames_JPN:wild:1 160x224@16,40",
          "Frames_JPN:wild:2 160x224@16,40",
          "Frames_JPN:wild:3 160x224@16,40",
          "Frames_JPN:wild:4 160x224@16,40",
          "Frames_JPN:wild:5 160x224@16,40",
          "Frames_JPN:wild:6 160x224@16,40",
          "Frames_JPN:wild:7 160x224@16,40",
          "Frames_JPN:wild:8 160x224@16,40",
        ],
        "total": 26,
        "wild": 8,
      }
    `);
  });
});
