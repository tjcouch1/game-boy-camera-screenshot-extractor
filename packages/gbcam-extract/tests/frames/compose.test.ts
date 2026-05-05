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
