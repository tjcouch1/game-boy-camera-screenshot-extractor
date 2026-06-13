import type { Frame } from "gbcam-extract";

/**
 * Extract a short region tag from a sheet stem. e.g. "Frames_USA" → "USA",
 * "Frames_JPN" → "JPN". Falls back to the full stem if no trailing tag is
 * found so unknown sheets still surface useful information.
 */
function regionFromStem(stem: string): string {
  const match = stem.match(/_([A-Za-z0-9]+)$/);
  return match ? match[1] : stem;
}

/**
 * Cosmetic clean-up for an individual frame's file stem: replace hyphens
 * with spaces and uppercase the first character. Existing capitalisation in
 * the rest of the stem is preserved so proper-noun-style names (e.g.
 * `wild-megaman-BOICHOT`) keep their casing.
 */
function prettifyStem(stem: string): string {
  const spaced = stem.replace(/-/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * Human-readable name for a frame, used in the picker UI and download
 * filenames.
 *
 * Examples:
 *   - `Frame 3` — unique index or shared across regional sheets
 *   - `Frame 3 (USA)` — colliding index (e.g. USA and JPN both have a Frame 3
 *     and they are visually different)
 *   - `Standard matrix` — individual frame; just the cleaned file stem
 */
export function frameDisplayName(frame: Frame, allFrames?: Frame[]): string {
  if (frame.kind === "individual") return prettifyStem(frame.sheetStem);
  const prefix = frame.type === "wild" ? "Wild Frame" : "Frame";

  const isRegional =
    frame.sheetStem === "Frames_USA" || frame.sheetStem === "Frames_JPN";

  // If it's shared across sheets (USA + JPN are identical), no region needed.
  if (frame.aliasStems.length > 1) return `${prefix} ${frame.index}`;

  // If it's not a regional frame, always show the stem/region.
  if (!isRegional)
    return `${prefix} ${frame.index} (${regionFromStem(frame.sheetStem)})`;

  // If we don't have the context of other frames, assume collision and show region.
  if (!allFrames)
    return `${prefix} ${frame.index} (${regionFromStem(frame.sheetStem)})`;

  // Only show region if another regional frame exists with the same index.
  const hasCollision = allFrames.some(
    (f) =>
      f.id !== frame.id &&
      f.type === frame.type &&
      f.index === frame.index &&
      (f.sheetStem === "Frames_USA" || f.sheetStem === "Frames_JPN"),
  );

  if (!hasCollision) return `${prefix} ${frame.index}`;
  return `${prefix} ${frame.index} (${regionFromStem(frame.sheetStem)})`;
}

/**
 * Sanitize a frame display name for use inside a filename. Mirrors
 * {@link sanitizePaletteName} but kept separate because frame names contain
 * parentheses and spaces that we want to flatten to underscores rather than
 * drop entirely (so "Frame 3 (USA)" → "Frame_3_USA" rather than "Frame3USA").
 */
export function sanitizeFrameName(name: string): string {
  return name
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}
