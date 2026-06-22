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
  it("tags both regions when they have a DIFFERENT frame at the same index", () => {
    const usa = makeFrame({
      id: "Frames_USA:normal:3",
      sheetStem: "Frames_USA",
      aliasStems: ["Frames_USA"],
      index: 3,
    });
    const jpn = makeFrame({
      id: "Frames_JPN:normal:3",
      sheetStem: "Frames_JPN",
      aliasStems: ["Frames_JPN"],
      index: 3,
    });
    const catalog = [usa, jpn];
    expect(frameDisplayName(usa, catalog)).toBe("Frame 3 (USA)");
    expect(frameDisplayName(jpn, catalog)).toBe("Frame 3 (JPN)");
  });

  it("does NOT tag a region-exclusive frame with no other-region collision", () => {
    // Only the USA sheet is present (or no JPN frame shares this index).
    const usa = makeFrame({
      id: "Frames_USA:normal:4",
      sheetStem: "Frames_USA",
      aliasStems: ["Frames_USA"],
      index: 4,
    });
    expect(frameDisplayName(usa, [usa])).toBe("Frame 4");
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
    expect(frameDisplayName(usaWins, [usaWins])).toBe("Frame 7");
    expect(frameDisplayName(jpnWins, [jpnWins])).toBe("Frame 7");
  });

  it("uses the 'Wild Frame' prefix and tags on a cross-region collision", () => {
    const usa = makeFrame({
      id: "Frames_USA:wild:2",
      type: "wild",
      sheetStem: "Frames_USA",
      aliasStems: ["Frames_USA"],
      index: 2,
      height: 224,
    });
    const jpn = makeFrame({
      id: "Frames_JPN:wild:2",
      type: "wild",
      sheetStem: "Frames_JPN",
      aliasStems: ["Frames_JPN"],
      index: 2,
      height: 224,
    });
    expect(frameDisplayName(jpn, [usa, jpn])).toBe("Wild Frame 2 (JPN)");
  });

  it("does not let a same-index frame of a different TYPE force a tag", () => {
    const normal = makeFrame({
      id: "Frames_USA:normal:1",
      sheetStem: "Frames_USA",
      aliasStems: ["Frames_USA"],
      index: 1,
    });
    const wild = makeFrame({
      id: "Frames_JPN:wild:1",
      type: "wild",
      sheetStem: "Frames_JPN",
      aliasStems: ["Frames_JPN"],
      index: 1,
      height: 224,
    });
    // Different types never collide, so neither is tagged.
    expect(frameDisplayName(normal, [normal, wild])).toBe("Frame 1");
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
