import { describe, it, expect } from "vitest";
import { splitSheet } from "../../src/frames/split-sheet.js";
import { dedupeFrames, appendDeduped } from "../../src/frames/dedupe.js";
import type { Frame } from "../../src/frames/types.js";
import { loadImage, repoRoot } from "../helpers/load-image.js";

function makeSyntheticFrame(stem: string, index: number, fillByte: number): Frame {
  const w = 160;
  const h = 144;
  const pixels = new Uint8ClampedArray(w * h).fill(fillByte);
  return {
    id: `${stem}:normal:${index}`,
    sheetStem: stem,
    aliasStems: [stem],
    type: "normal",
    kind: "sheet",
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

  it("returns one entry per distinct frame when there are no duplicates", () => {
    const a = makeSyntheticFrame("A", 1, 0);
    const b = makeSyntheticFrame("B", 1, 82);
    const out = dedupeFrames([a, b]);
    expect(out.map((f) => f.id).sort()).toEqual([
      "A:normal:1",
      "B:normal:1",
    ]);
  });

  it("keeps the alphabetically later sheet's frame when duplicates exist", () => {
    // Same pixels, different stems. USA > JPN alphabetically, so USA wins.
    const usa = makeSyntheticFrame("Frames_USA", 1, 0);
    const jpn = makeSyntheticFrame("Frames_JPN", 1, 0);
    const out = dedupeFrames([usa, jpn]);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("Frames_USA:normal:1");
    // Both stems should appear in aliasStems so callers can detect that this
    // frame is shared across sheets.
    expect(out[0].aliasStems.sort()).toEqual(["Frames_JPN", "Frames_USA"]);
  });

  it("does not mutate the input frames' aliasStems arrays", () => {
    const usa = makeSyntheticFrame("Frames_USA", 1, 0);
    const jpn = makeSyntheticFrame("Frames_JPN", 1, 0);
    dedupeFrames([usa, jpn]);
    expect(usa.aliasStems).toEqual(["Frames_USA"]);
    expect(jpn.aliasStems).toEqual(["Frames_JPN"]);
  });

  it("treats different dimensions as distinct even when pixel arrays would match in prefix", () => {
    const a = makeSyntheticFrame("A", 1, 0);
    const b: Frame = { ...makeSyntheticFrame("B", 1, 0), width: 160, height: 100 };
    b.pixels = new Uint8ClampedArray(160 * 100).fill(0);
    const out = dedupeFrames([a, b]);
    expect(out).toHaveLength(2);
  });

  it("appendDeduped keeps base order, appends only new fingerprints, and merges aliases", () => {
    const sheetA = makeSyntheticFrame("Frames_USA", 1, 0);
    const sheetB = makeSyntheticFrame("Frames_USA", 2, 82);
    const indDup = makeSyntheticFrame("ind-dup", 1, 0); // same pixels as sheetA
    const indNew = makeSyntheticFrame("ind-new", 1, 165);
    const out = appendDeduped([sheetA, sheetB], [indDup, indNew]);
    // Sheets stay first, in their original order. Only the new individual frame is appended.
    expect(out.map((f) => f.id)).toEqual([
      "Frames_USA:normal:1",
      "Frames_USA:normal:2",
      "ind-new:normal:1",
    ]);
    // Duplicate individual's stem merged onto the matching sheet frame.
    expect(out[0].aliasStems.sort()).toEqual(["Frames_USA", "ind-dup"]);
    // Inputs should not be mutated.
    expect(sheetA.aliasStems).toEqual(["Frames_USA"]);
    expect(indDup.aliasStems).toEqual(["ind-dup"]);
  });

  it("deduplicates real sheets and yields fewer frames than the sum", async () => {
    const usaSheet = await loadImage(
      repoRoot("supporting-materials/frames/sheets/the-spriters-resource/Frames_USA.png"),
    );
    const jpnSheet = await loadImage(
      repoRoot("supporting-materials/frames/sheets/the-spriters-resource/Frames_JPN.png"),
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
      sharedAcrossSheets: out.filter((f) => f.aliasStems.length > 1).length,
    };
    expect(summary).toMatchInlineSnapshot(`
      {
        "combined": 51,
        "deduped": 36,
        "jpn": 26,
        "jpnWinners": 11,
        "sharedAcrossSheets": 15,
        "usa": 25,
        "usaWinners": 25,
      }
    `);
  });
});
