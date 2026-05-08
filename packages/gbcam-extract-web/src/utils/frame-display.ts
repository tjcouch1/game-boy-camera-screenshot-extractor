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
 * Human-readable name for a frame, used in the picker UI and download
 * filenames.
 *
 * Examples:
 *   - `Frame 3 (USA)` — normal frame from a single sheet
 *   - `Frame 3` — frame deduplicated across sheets (region dropped because
 *     the same image appeared in multiple sheets)
 *   - `Wild Frame 1 (JPN)` — wild frame from a single sheet
 */
export function frameDisplayName(frame: Frame): string {
  const prefix = frame.type === "wild" ? "Wild Frame" : "Frame";
  const sharedAcrossSheets = frame.aliasStems.length > 1;
  if (sharedAcrossSheets) return `${prefix} ${frame.index}`;
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
