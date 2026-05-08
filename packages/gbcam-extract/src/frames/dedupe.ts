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
    const fp = `${f.width}x${f.height}:${f.type}:${fnv1a(f.pixels)}`;
    const winner = byFp.get(fp);
    if (winner) {
      for (const stem of f.aliasStems) {
        if (!winner.aliasStems.includes(stem)) winner.aliasStems.push(stem);
      }
    } else {
      byFp.set(fp, { ...f, aliasStems: [...f.aliasStems] });
      order.push(fp);
    }
  }
  return order.map((fp) => byFp.get(fp)!);
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
