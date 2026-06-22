import type { Frame } from "./types.js";

/**
 * Remove pixel-identical duplicates across frames.
 *
 * Tiebreaker: the alphabetically-latest `sheetStem` wins (so `Frames_USA`
 * beats `Frames_JPN`). We sort the input descending by stem first, then walk
 * and keep first-seen unique fingerprints. When a duplicate is encountered
 * its `aliasStems` are merged into the winner so callers can tell whether a
 * given frame appeared in multiple sheets.
 *
 * Fingerprint = `<width>x<height>:<type>:<FNV-1a hash of pixels>`.
 */
export function dedupeFrames(frames: Frame[]): Frame[] {
  const sorted = [...frames].sort((a, b) =>
    b.sheetStem.localeCompare(a.sheetStem),
  );
  const byFp = new Map<string, Frame>();
  const order: string[] = [];
  for (const f of sorted) {
    const fp = frameFingerprint(f);
    const winner = byFp.get(fp);
    if (winner) {
      mergeAliases(winner, f);
    } else {
      byFp.set(fp, { ...f, aliasStems: [...f.aliasStems] });
      order.push(fp);
    }
  }
  return order.map((fp) => byFp.get(fp)!);
}

/**
 * Append `more` to `base`, dropping any frame in `more` that duplicates a
 * frame already in `base` (matched by fingerprint), and dropping duplicates
 * within `more` (first occurrence wins). Aliases are merged onto whichever
 * frame ends up in the result.
 *
 * Order: every frame from `base` keeps its position, then unique frames
 * from `more` follow in input order. Use this when you need to layer a new
 * source of frames after a previously-deduplicated set without resorting.
 */
export function appendDeduped(base: Frame[], more: Frame[]): Frame[] {
  const byFp = new Map<string, Frame>();
  const baseOrder: string[] = [];
  for (const f of base) {
    const fp = frameFingerprint(f);
    if (!byFp.has(fp)) {
      byFp.set(fp, { ...f, aliasStems: [...f.aliasStems] });
      baseOrder.push(fp);
    } else {
      mergeAliases(byFp.get(fp)!, f);
    }
  }
  const moreOrder: string[] = [];
  for (const f of more) {
    const fp = frameFingerprint(f);
    const winner = byFp.get(fp);
    if (winner) {
      mergeAliases(winner, f);
    } else {
      byFp.set(fp, { ...f, aliasStems: [...f.aliasStems] });
      moreOrder.push(fp);
    }
  }
  return [...baseOrder, ...moreOrder].map((fp) => byFp.get(fp)!);
}

/** Stable identity key — same dimensions, type, and pixels → same fingerprint. */
export function frameFingerprint(frame: Frame): string {
  return `${frame.width}x${frame.height}:${frame.type}:${fnv1a(frame.pixels)}`;
}

function mergeAliases(winner: Frame, other: Frame): void {
  for (const stem of other.aliasStems) {
    if (!winner.aliasStems.includes(stem)) winner.aliasStems.push(stem);
  }
}

/** FNV-1a 32-bit on a byte stream — fast and good enough for exact dedup. */
function fnv1a(bytes: Uint8ClampedArray): string {
  let h = 0x811c9dc5; // FNV-1a 32-bit offset basis
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193); // FNV-1a 32-bit prime
  }
  // Convert to unsigned hex.
  return (h >>> 0).toString(16);
}
