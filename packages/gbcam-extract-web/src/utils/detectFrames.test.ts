import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the underlying gbcam-extract loaders so we can drive
// detectAndLoadFrames's dispatch logic without fabricating real GBImageData
// for every code path. The actual loaders have their own dedicated tests in
// the gbcam-extract package.
vi.mock("gbcam-extract", () => {
  return {
    splitSheet: vi.fn(),
    loadIndividualFrame: vi.fn(),
  };
});

import { splitSheet, loadIndividualFrame } from "gbcam-extract";
import type { Frame, GBImageData } from "gbcam-extract";
import {
  detectAndLoadFrames,
  sanitizeFilenameStem,
  disambiguateStem,
} from "./detectFrames.js";

const splitSheetMock = vi.mocked(splitSheet);
const loadIndividualFrameMock = vi.mocked(loadIndividualFrame);

function makeFrame(
  partial: Partial<Frame> & { id: string; sheetStem: string },
): Frame {
  return {
    aliasStems: [partial.sheetStem],
    type: "normal",
    kind: "sheet",
    index: 1,
    width: 160,
    height: 144,
    pixels: new Uint8ClampedArray(160 * 144),
    holeX: 16,
    holeY: 16,
    ...partial,
  };
}

function makeImage(): GBImageData {
  return { data: new Uint8ClampedArray(4), width: 1, height: 1 };
}

describe("detectAndLoadFrames", () => {
  beforeEach(() => {
    splitSheetMock.mockReset();
    loadIndividualFrameMock.mockReset();
  });

  it("returns sheet frames when splitSheet yields >= 2 frames", () => {
    splitSheetMock.mockReturnValue([
      makeFrame({ id: "test:normal:1", sheetStem: "test", index: 1 }),
      makeFrame({ id: "test:normal:2", sheetStem: "test", index: 2 }),
    ]);
    const result = detectAndLoadFrames(makeImage(), "test");
    expect(result).toHaveLength(2);
    // detectAndLoadFrames forces kind:"individual" on every result so storage
    // entries don't carry sheet provenance.
    expect(result.every((f) => f.kind === "individual")).toBe(true);
    expect(loadIndividualFrameMock).not.toHaveBeenCalled();
  });

  it("prefers loadIndividualFrame when splitSheet returns a single frame", () => {
    // splitSheet's tight-bbox recomputation clips individual frames whose
    // bezel narrows below the hole (e.g. Game Boy Pocket frames with a
    // label area separated by background). loadIndividualFrame uses the
    // full image dimensions, so we prefer it whenever both succeed.
    splitSheetMock.mockReturnValue([
      makeFrame({
        id: "pocket:wild:1",
        sheetStem: "pocket",
        type: "wild",
        width: 160,
        height: 130, // clipped — bottom of the frame is missing
      }),
    ]);
    loadIndividualFrameMock.mockReturnValue(
      makeFrame({
        id: "pocket:wild:1",
        sheetStem: "pocket",
        type: "wild",
        kind: "individual",
        width: 160,
        height: 180, // full source dimensions
      }),
    );
    const result = detectAndLoadFrames(makeImage(), "pocket");
    expect(result).toHaveLength(1);
    expect(result[0].height).toBe(180);
    expect(loadIndividualFrameMock).toHaveBeenCalledOnce();
  });

  it("falls back to loadIndividualFrame when splitSheet returns no frames", () => {
    splitSheetMock.mockReturnValue([]);
    loadIndividualFrameMock.mockReturnValue(
      makeFrame({
        id: "lone:normal:1",
        sheetStem: "lone",
        kind: "individual",
      }),
    );
    const result = detectAndLoadFrames(makeImage(), "lone");
    expect(result).toHaveLength(1);
    expect(result[0].kind).toBe("individual");
    expect(loadIndividualFrameMock).toHaveBeenCalledOnce();
  });

  it("falls back to loadIndividualFrame when splitSheet throws", () => {
    splitSheetMock.mockImplementation(() => {
      throw new Error("malformed sheet");
    });
    loadIndividualFrameMock.mockReturnValue(
      makeFrame({
        id: "lone:normal:1",
        sheetStem: "lone",
        kind: "individual",
      }),
    );
    const result = detectAndLoadFrames(makeImage(), "lone");
    expect(result).toHaveLength(1);
  });

  it("uses splitSheet's single result if loadIndividualFrame fails", () => {
    // Defensive: a true single-frame sheet whose body has a non-white,
    // non-transparent background can defeat loadIndividualFrame's
    // hole search. In that case we still want to return something.
    splitSheetMock.mockReturnValue([
      makeFrame({
        id: "weird:normal:1",
        sheetStem: "weird",
        kind: "sheet",
      }),
    ]);
    loadIndividualFrameMock.mockImplementation(() => {
      throw new Error("no hole found");
    });
    const result = detectAndLoadFrames(makeImage(), "weird");
    expect(result).toHaveLength(1);
    expect(result[0].kind).toBe("individual");
  });

  it("throws when neither splitSheet nor loadIndividualFrame succeed", () => {
    splitSheetMock.mockReturnValue([]);
    loadIndividualFrameMock.mockImplementation(() => {
      throw new Error("no hole found");
    });
    expect(() => detectAndLoadFrames(makeImage(), "garbage")).toThrow(
      /Couldn't detect a frame/i,
    );
  });
});

describe("sanitizeFilenameStem", () => {
  it("strips extensions and replaces unsafe chars", () => {
    expect(sanitizeFilenameStem("My Frame!.png")).toBe("My-Frame");
    expect(sanitizeFilenameStem("frame_01.PNG")).toBe("frame_01");
    expect(sanitizeFilenameStem("a/b/c.jpg")).toBe("a-b-c");
  });

  it("collapses runs of dashes and trims edges", () => {
    expect(sanitizeFilenameStem("--weird---name--.png")).toBe("weird-name");
  });
});

describe("disambiguateStem", () => {
  it("returns the input when it's not taken", () => {
    expect(disambiguateStem("foo", new Set())).toBe("foo");
  });

  it("appends -2 on first collision and increments thereafter", () => {
    expect(disambiguateStem("foo", new Set(["foo"]))).toBe("foo-2");
    expect(disambiguateStem("foo", new Set(["foo", "foo-2"]))).toBe("foo-3");
  });
});
