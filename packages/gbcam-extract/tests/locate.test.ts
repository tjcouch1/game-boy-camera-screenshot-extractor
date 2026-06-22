import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { locate } from "../src/locate.js";
import { initOpenCV } from "../src/init-opencv.js";
import { createDebugCollector } from "../src/debug.js";
import { loadImage as loadHelperImage, repoRoot } from "./helpers/load-image.js";

beforeAll(async () => {
  await initOpenCV();
}, 30_000);

describe("locate (synthetic)", () => {
  it("throws a clear no-candidate error when given a uniformly dark image", () => {
    const w = 800, h = 600;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 30; data[i + 1] = 30; data[i + 2] = 30; data[i + 3] = 255;
    }
    expect(() => locate({ data, width: w, height: h })).toThrow(/locate/);
  });
});

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
const CORNERS_TOLERANCE_PX = 70;

function loadCornersFixture(): CornersFixture {
  const path = repoRoot("supporting-materials", "hand-edited-rectangles", "corners.json");
  return JSON.parse(readFileSync(path, "utf-8"));
}

describe("locate (real photos vs corners.json)", () => {
  const fixture = loadCornersFixture();
  const stems = Object.keys(fixture.images);

  for (const stem of stems) {
    it(`${stem}: output corners within ${CORNERS_TOLERANCE_PX}px of corners.json`, async () => {
      const inputPath = repoRoot("test-input-full", `${stem}.jpg`);
      const input = await loadHelperImage(inputPath);
      const dbg = createDebugCollector();

      // Run locate; compare its detected screen corners against the
      // hand-marked rectangle in corners.json. corners.json marks the
      // screen edges (160:144 aspect), not the post-margin output, so we
      // compare against `chosenCandidate.corners` rather than `outputCorners`.
      locate(input, { debug: dbg });
      const m = dbg.data.metrics.locate;
      expect(m, `metrics.locate should be populated`).toBeDefined();

      const chosen = m.chosenCandidate as { corners?: [number, number][] } | undefined;
      const screenCorners = chosen?.corners;
      expect(screenCorners, `metrics.locate.chosenCandidate.corners should be set`).toBeDefined();
      expect(screenCorners!.length).toBe(4);

      const expected = fixture.images[stem].corners;
      const expectedOrdered: [number, number][] = [
        expected.topLeft,
        expected.topRight,
        expected.bottomRight,
        expected.bottomLeft,
      ];

      for (let i = 0; i < 4; i++) {
        const [ox, oy] = screenCorners![i];
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
