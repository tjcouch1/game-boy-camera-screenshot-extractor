import { describe, it, expect } from "vitest";
import {
  detectProcessedImage,
  WILD_W,
  WILD_H,
} from "../src/detect-processed.js";
import {
  createGBImageData,
  GB_COLORS,
  CAM_W,
  CAM_H,
  SCREEN_W,
  SCREEN_H,
  FRAME_THICK,
  type GBImageData,
} from "../src/common.js";
import type { Frame } from "../src/frames/types.js";

/** Deterministic PRNG so tests are reproducible. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

/** Build a random 4-value GB grayscale image at the given size. */
function makeGBImage(w: number, h: number, seed = 42): GBImageData {
  const rng = makeRng(seed);
  const img = createGBImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = GB_COLORS[Math.floor(rng() * 4)];
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  return img;
}

/** Nearest-neighbor upscale by integer factor. */
function upscaleImage(img: GBImageData, k: number): GBImageData {
  const out = createGBImageData(img.width * k, img.height * k);
  for (let y = 0; y < out.height; y++) {
    for (let x = 0; x < out.width; x++) {
      const si = ((Math.floor(y / k) * img.width) + Math.floor(x / k)) * 4;
      const di = (y * out.width + x) * 4;
      out.data[di] = img.data[si];
      out.data[di + 1] = img.data[si + 1];
      out.data[di + 2] = img.data[si + 2];
      out.data[di + 3] = 255;
    }
  }
  return out;
}

/** Add bounded uniform noise to every channel (simulates lossy compression). */
function addNoise(img: GBImageData, amplitude: number, seed = 7): GBImageData {
  const rng = makeRng(seed);
  const out = createGBImageData(img.width, img.height);
  for (let i = 0; i < img.data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const n = Math.round((rng() * 2 - 1) * amplitude);
      out.data[i + c] = Math.max(0, Math.min(255, img.data[i + c] + n));
    }
    out.data[i + 3] = 255;
  }
  return out;
}

/** Apply a hex palette to a GB grayscale image ([255, 165, 82, 0] order). */
function applyHexPalette(
  img: GBImageData,
  colors: [string, string, string, string],
): GBImageData {
  const byValue = new Map<number, [number, number, number]>();
  const parse = (hex: string): [number, number, number] => [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
  byValue.set(255, parse(colors[0]));
  byValue.set(165, parse(colors[1]));
  byValue.set(82, parse(colors[2]));
  byValue.set(0, parse(colors[3]));
  const out = createGBImageData(img.width, img.height);
  for (let i = 0; i < img.data.length; i += 4) {
    const [r, g, b] = byValue.get(img.data[i])!;
    out.data[i] = r;
    out.data[i + 1] = g;
    out.data[i + 2] = b;
    out.data[i + 3] = 255;
  }
  return out;
}

function expectSameGrayscale(actual: GBImageData, expected: GBImageData) {
  expect(actual.width).toBe(expected.width);
  expect(actual.height).toBe(expected.height);
  let diffs = 0;
  for (let i = 0; i < actual.data.length; i += 4) {
    if (actual.data[i] !== expected.data[i]) diffs++;
  }
  expect(diffs).toBe(0);
}

describe("detectProcessedImage", () => {
  it("detects a bare 128x112 grayscale image at 1x", () => {
    const img = makeGBImage(CAM_W, CAM_H);
    const det = detectProcessedImage(img);
    expect(det).not.toBeNull();
    expect(det!.layout).toBe("bare");
    expect(det!.scale).toBe(1);
    expectSameGrayscale(det!.grayscale, img);
  });

  it("detects an integer-upscaled bare image and downsamples it", () => {
    const base = makeGBImage(CAM_W, CAM_H, 3);
    const det = detectProcessedImage(upscaleImage(base, 3));
    expect(det).not.toBeNull();
    expect(det!.layout).toBe("bare");
    expect(det!.scale).toBe(3);
    expectSameGrayscale(det!.grayscale, base);
  });

  it("survives compression-like noise", () => {
    const base = makeGBImage(CAM_W, CAM_H, 5);
    const noisy = addNoise(upscaleImage(base, 2), 12);
    const det = detectProcessedImage(noisy);
    expect(det).not.toBeNull();
    expectSameGrayscale(det!.grayscale, base);
  });

  it("detects a palette-rendered image and maps colors back", () => {
    const base = makeGBImage(CAM_W, CAM_H, 9);
    // The "Down" palette used by the app's RGB output.
    const colored = applyHexPalette(base, [
      "#FFFFA5",
      "#FF9494",
      "#9494FF",
      "#000000",
    ]);
    const det = detectProcessedImage(upscaleImage(colored, 4));
    expect(det).not.toBeNull();
    expect(det!.scale).toBe(4);
    expectSameGrayscale(det!.grayscale, base);
  });

  it("maps an unknown 4-color palette via luminance rank", () => {
    const base = makeGBImage(CAM_W, CAM_H, 11);
    const colored = applyHexPalette(base, [
      "#F0E8D0",
      "#B08858",
      "#605040",
      "#181008",
    ]);
    const det = detectProcessedImage(colored);
    expect(det).not.toBeNull();
    expectSameGrayscale(det!.grayscale, base);
  });

  it("crops the camera image out of a normal-framed (160x144) output", () => {
    const cam = makeGBImage(CAM_W, CAM_H, 13);
    const framed = makeGBImage(SCREEN_W, SCREEN_H, 17);
    for (let y = 0; y < CAM_H; y++) {
      for (let x = 0; x < CAM_W; x++) {
        const si = (y * CAM_W + x) * 4;
        const di = ((y + FRAME_THICK) * SCREEN_W + (x + FRAME_THICK)) * 4;
        for (let c = 0; c < 4; c++) framed.data[di + c] = cam.data[si + c];
      }
    }
    const det = detectProcessedImage(upscaleImage(framed, 2));
    expect(det).not.toBeNull();
    expect(det!.layout).toBe("normal-frame");
    expectSameGrayscale(det!.grayscale, cam);
  });

  it("locates the hole in a wild-framed output via a known frame", () => {
    const holeX = 24;
    const holeY = 80;
    const framePixels = new Uint8ClampedArray(WILD_W * WILD_H);
    const rng = makeRng(23);
    for (let i = 0; i < framePixels.length; i++) {
      framePixels[i] = GB_COLORS[Math.floor(rng() * 4)];
    }
    const frame: Frame = {
      id: "test:wild:1",
      sheetStem: "test",
      aliasStems: ["test"],
      type: "wild",
      kind: "sheet",
      index: 1,
      width: WILD_W,
      height: WILD_H,
      pixels: framePixels,
      holeX,
      holeY,
    };

    const cam = makeGBImage(CAM_W, CAM_H, 29);
    const composed = createGBImageData(WILD_W, WILD_H);
    for (let y = 0; y < WILD_H; y++) {
      for (let x = 0; x < WILD_W; x++) {
        const inHole =
          x >= holeX && x < holeX + CAM_W && y >= holeY && y < holeY + CAM_H;
        const v = inHole
          ? cam.data[((y - holeY) * CAM_W + (x - holeX)) * 4]
          : framePixels[y * WILD_W + x];
        const o = (y * WILD_W + x) * 4;
        composed.data[o] = v;
        composed.data[o + 1] = v;
        composed.data[o + 2] = v;
        composed.data[o + 3] = 255;
      }
    }

    const det = detectProcessedImage(composed, { knownFrames: [frame] });
    expect(det).not.toBeNull();
    expect(det!.layout).toBe("wild-frame");
    expect(det!.matchedFrameId).toBe("test:wild:1");
    expectSameGrayscale(det!.grayscale, cam);
  });

  it("rejects images whose dimensions don't match any layout", () => {
    expect(detectProcessedImage(makeGBImage(130, 112))).toBeNull();
    expect(detectProcessedImage(makeGBImage(CAM_W * 2, CAM_H * 3))).toBeNull();
    expect(detectProcessedImage(makeGBImage(127, 111))).toBeNull();
  });

  it("rejects a many-colored photo even at matching dimensions", () => {
    const rng = makeRng(31);
    const img = createGBImageData(CAM_W, CAM_H);
    for (let i = 0; i < img.data.length; i += 4) {
      img.data[i] = Math.floor(rng() * 256);
      img.data[i + 1] = Math.floor(rng() * 256);
      img.data[i + 2] = Math.floor(rng() * 256);
      img.data[i + 3] = 255;
    }
    expect(detectProcessedImage(img)).toBeNull();
  });

  it("rejects a smooth-gradient image at matching dimensions", () => {
    const img = createGBImageData(CAM_W, CAM_H);
    for (let y = 0; y < CAM_H; y++) {
      for (let x = 0; x < CAM_W; x++) {
        const i = (y * CAM_W + x) * 4;
        const v = Math.round((x / (CAM_W - 1)) * 255);
        img.data[i] = v;
        img.data[i + 1] = v;
        img.data[i + 2] = v;
        img.data[i + 3] = 255;
      }
    }
    expect(detectProcessedImage(img)).toBeNull();
  });

  it("rejects a grayscale image quantized to non-GB values", () => {
    const rng = makeRng(37);
    const img = createGBImageData(CAM_W, CAM_H);
    // 60 and 100 both snap to GB value 82 — ambiguous, so not a GB image.
    const values = [60, 100, 200, 240];
    for (let i = 0; i < img.data.length; i += 4) {
      const v = values[Math.floor(rng() * 4)];
      img.data[i] = v;
      img.data[i + 1] = v;
      img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
    expect(detectProcessedImage(img)).toBeNull();
  });
});
