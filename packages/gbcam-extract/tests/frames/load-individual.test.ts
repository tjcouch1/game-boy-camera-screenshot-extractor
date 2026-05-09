import { describe, it, expect } from "vitest";
import { loadIndividualFrame } from "../../src/frames/load-individual.js";
import type { GBImageData } from "../../src/common.js";

/**
 * Build a synthetic individual-frame image:
 *   • W × H canvas, pre-filled with `frameValue` grayscale and full alpha.
 *   • A 128 × 112 block at (holeX, holeY) is set to either fully transparent
 *     or pure white, depending on `holeMode`.
 */
function buildIndividual(
  W: number,
  H: number,
  holeX: number,
  holeY: number,
  frameValue: number,
  holeMode: "transparent" | "white",
): GBImageData {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    data[i * 4 + 0] = frameValue;
    data[i * 4 + 1] = frameValue;
    data[i * 4 + 2] = frameValue;
    data[i * 4 + 3] = 255;
  }
  for (let y = holeY; y < holeY + 112; y++) {
    for (let x = holeX; x < holeX + 128; x++) {
      const i = (y * W + x) * 4;
      if (holeMode === "transparent") {
        data[i + 0] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
      } else {
        data[i + 0] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
        data[i + 3] = 255;
      }
    }
  }
  return { data, width: W, height: H };
}

describe("loadIndividualFrame", () => {
  it("classifies a 160 × 144 image as normal and finds a transparent hole", () => {
    const image = buildIndividual(160, 144, 16, 16, 0, "transparent");
    const f = loadIndividualFrame(image, "Test_Normal");
    expect(f.id).toBe("Test_Normal:normal:1");
    expect(f.sheetStem).toBe("Test_Normal");
    expect(f.aliasStems).toEqual(["Test_Normal"]);
    expect(f.type).toBe("normal");
    expect(f.kind).toBe("individual");
    expect(f.index).toBe(1);
    expect(f.width).toBe(160);
    expect(f.height).toBe(144);
    expect(f.holeX).toBe(16);
    expect(f.holeY).toBe(16);
    // Frame body pixel: black (0).
    expect(f.pixels[0]).toBe(0);
    // Hole pixel rendered as 255 (lightest palette colour).
    expect(f.pixels[16 * 160 + 16]).toBe(255);
  });

  it("classifies non-160×144 dimensions as wild and finds a white hole", () => {
    const image = buildIndividual(192, 224, 32, 80, 165, "white");
    const f = loadIndividualFrame(image, "wild-something");
    expect(f.id).toBe("wild-something:wild:1");
    expect(f.type).toBe("wild");
    expect(f.holeX).toBe(32);
    expect(f.holeY).toBe(80);
    // Frame body pixel snaps 165 → 165 (LG).
    expect(f.pixels[0]).toBe(165);
    // Hole pixel stored as 255.
    expect(f.pixels[80 * 192 + 32]).toBe(255);
    // Every pixel is one of the four GB grayscale values.
    for (let i = 0; i < f.pixels.length; i++) {
      expect([0, 82, 165, 255]).toContain(f.pixels[i]);
    }
  });

  it("throws when no 128×112 transparent or white region exists", () => {
    // Solid black image, no hole.
    const W = 160;
    const H = 144;
    const data = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      data[i * 4 + 3] = 255;
    }
    const image: GBImageData = { data, width: W, height: H };
    expect(() => loadIndividualFrame(image, "no-hole")).toThrow(/no .*hole/);
  });

  it("throws when the image is smaller than the hole", () => {
    const data = new Uint8ClampedArray(64 * 56 * 4);
    const image: GBImageData = { data, width: 64, height: 56 };
    expect(() => loadIndividualFrame(image, "tiny")).toThrow();
  });
});
