import type { Frame } from "./types.js";

/**
 * Remove pixel-identical duplicates across frames.
 *
 * Tiebreaker: the alphabetically-earliest `sheetStem` wins. We sort the
 * input by stem first, then walk and keep first-seen unique fingerprints.
 *
 * Fingerprint = `<width>x<height>:<type>:<FNV-1a hash of pixels>`.
 */
export function dedupeFrames(frames: Frame[]): Frame[] {
  const sorted = [...frames].sort((a, b) =>
    a.sheetStem.localeCompare(b.sheetStem),
  );
  const seen = new Set<string>();
  const out: Frame[] = [];
  for (const f of sorted) {
    const fp = `${f.width}x${f.height}:${f.type}:${fnv1a(f.pixels)}`;
    if (seen.has(fp)) continue;
    seen.add(fp);
    out.push(f);
  }
  return out;
}

/** FNV-1a 32-bit on a byte stream — fast and good enough for exact dedup. */
function fnv1a(bytes: Uint8ClampedArray): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193);
  }
  // Convert to unsigned hex.
  return (h >>> 0).toString(16);
}
