import { describe, it, expect } from "vitest";
import type { Frame } from "gbcam-extract";
import { frameDisplayName } from "./frame-display.js";

function makeFrame(overrides: Partial<Frame> = {}): Frame {
  return {
    id: "Frames_USA:normal:1",
    sheetStem: "Frames_USA",
    aliasStems: ["Frames_USA"],
    type: "normal",
    kind: "sheet",
    index: 1,
    width: 160,
    height: 144,
    pixels: new Uint8ClampedArray(160 * 144),
    holeX: 16,
    holeY: 16,
    ...overrides,
  };
}

describe("frameDisplayName — regional tagging", () => {
  it("tags a USA-exclusive frame with (USA)", () => {
    const f = makeFrame({ sheetStem: "Frames_USA", aliasStems: ["Frames_USA"], index: 3 });
    expect(frameDisplayName(f)).toBe("Frame 3 (USA)");
  });

  it("tags a JPN-exclusive frame with (JPN)", () => {
    const f = makeFrame({ sheetStem: "Frames_JPN", aliasStems: ["Frames_JPN"], index: 5 });
    expect(frameDisplayName(f)).toBe("Frame 5 (JPN)");
  });

  it("gives a shared frame (both regions) no tag, regardless of upload order", () => {
    const usaWins = makeFrame({
      sheetStem: "Frames_USA",
      aliasStems: ["Frames_USA", "Frames_JPN"],
      index: 7,
    });
    const jpnWins = makeFrame({
      sheetStem: "Frames_JPN",
      aliasStems: ["Frames_JPN", "Frames_USA"],
      index: 7,
    });
    expect(frameDisplayName(usaWins)).toBe("Frame 7");
    expect(frameDisplayName(jpnWins)).toBe("Frame 7");
  });

  it("uses the 'Wild Frame' prefix for wild frames", () => {
    const f = makeFrame({
      type: "wild",
      sheetStem: "Frames_JPN",
      aliasStems: ["Frames_JPN"],
      index: 2,
      height: 224,
    });
    expect(frameDisplayName(f)).toBe("Wild Frame 2 (JPN)");
  });

  it("uses the cleaned stem for individual frames", () => {
    const f = makeFrame({
      kind: "individual",
      sheetStem: "standard-matrix",
      aliasStems: ["standard-matrix"],
    });
    expect(frameDisplayName(f)).toBe("Standard matrix");
  });
});
