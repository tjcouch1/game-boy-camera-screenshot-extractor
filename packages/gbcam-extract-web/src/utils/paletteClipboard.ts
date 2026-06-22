/**
 * Utility functions for copying and pasting palettes to/from clipboard.
 */

export interface SerializedPalette {
  name: string;
  colors: [string, string, string, string];
}

/**
 * Serialize a palette to a JSON string for clipboard storage.
 * Format: JSON string with name and colors array
 */
export function serializePaletteToClipboard(
  palette: SerializedPalette,
): string {
  return JSON.stringify({
    type: "gbcam-palette",
    version: "1",
    data: palette,
  });
}

/**
 * Deserialize a palette from clipboard contents.
 * Validates format and returns the palette, or null if invalid.
 */
export function deserializePaletteFromClipboard(
  text: string,
): SerializedPalette | null {
  try {
    const parsed = JSON.parse(text);

    // Check for our format
    if (parsed.type === "gbcam-palette" && parsed.version === "1") {
      const data = parsed.data;
      if (
        data &&
        typeof data === "object" &&
        typeof data.name === "string" &&
        Array.isArray(data.colors) &&
        data.colors.length === 4 &&
        data.colors.every(
          (c: unknown) => typeof c === "string" && /^#[0-9A-F]{6}$/i.test(c),
        )
      ) {
        return data as SerializedPalette;
      }
    }
  } catch {
    // Invalid JSON or parse error - not a palette
  }

  return null;
}

/**
 * Check if clipboard contents are formatted as a palette.
 * This is async because we need to read from the clipboard API.
 * Gracefully handles permission denied errors.
 */
export async function isPaletteInClipboard(): Promise<boolean> {
  try {
    // Check if clipboard API is available
    if (!navigator.clipboard || !navigator.clipboard.readText) {
      return false;
    }

    // On some mobile browsers, readText() requires user permission
    // Try to read, catch permission errors gracefully
    try {
      const text = await navigator.clipboard.readText();
      return deserializePaletteFromClipboard(text) !== null;
    } catch (err) {
      // NotAllowedError means permission denied, which is expected on some mobile browsers
      // Throw it so the hook can stop polling.
      if (err instanceof Error && err.name === "NotAllowedError") {
        throw err;
      }
      return false;
    }
  } catch (err) {
    // Clipboard API not available
    if (err instanceof Error && err.name === "NotAllowedError") {
      throw err;
    }
    return false;
  }
}

/**
 * Read a palette from the clipboard.
 * Requires user permission on mobile browsers.
 */
export async function readPaletteFromClipboard(): Promise<SerializedPalette | null> {
  try {
    // Check if clipboard API is available
    if (!navigator.clipboard || !navigator.clipboard.readText) {
      return null;
    }

    const text = await navigator.clipboard.readText();
    return deserializePaletteFromClipboard(text);
  } catch (err) {
    // NotAllowedError (permission denied) and other errors
    // Log for debugging but don't throw to user
    console.debug("Failed to read from clipboard:", (err as Error).name);
    return null;
  }
}

/**
 * Write a palette to the clipboard.
 * Returns true if successful, false otherwise.
 */
export async function writePaletteToClipboard(
  palette: SerializedPalette,
): Promise<boolean> {
  try {
    // Check if clipboard API is available
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      return false;
    }

    const serialized = serializePaletteToClipboard(palette);
    await navigator.clipboard.writeText(serialized);
    return true;
  } catch (err) {
    // NotAllowedError or other write errors
    // Log for debugging but return false gracefully
    console.debug("Failed to write to clipboard:", (err as Error).name);
    return false;
  }
}
