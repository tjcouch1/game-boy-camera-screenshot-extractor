import { useCallback } from "react";
import { useLocalStorage } from "./useLocalStorage.js";

const STORAGE_KEY = "gbcam-user-palettes";
const STORAGE_VERSION = "1";
const STORAGE_VERSION_KEY = "gbcam-user-palettes-version";

export interface UserPaletteEntry {
  id: string;
  name: string;
  colors: [string, string, string, string];
  isEditing?: boolean;
  savedColors?: [string, string, string, string];
  savedName?: string;
}

function generateId(): string {
  return `palette-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// On module load: if the stored schema version doesn't match, drop stored
// palettes so useLocalStorage falls back to its initial value. Then write
// the current version so future loads round-trip cleanly.
try {
  if (localStorage.getItem(STORAGE_VERSION_KEY) !== STORAGE_VERSION) {
    localStorage.removeItem(STORAGE_KEY);
  }
  localStorage.setItem(STORAGE_VERSION_KEY, STORAGE_VERSION);
} catch {
  // localStorage may be unavailable (private mode, etc.) — ignore.
}

export function useUserPalettes() {
  const [palettes, setPalettes] = useLocalStorage<UserPaletteEntry[]>(
    STORAGE_KEY,
    [],
  );

  // Create a new palette in edit mode with auto-generated unique name
  const createPaletteInEditMode = useCallback(
    (fromName: string, fromColors: [string, string, string, string]) => {
      // Strip any existing " custom <N>" suffix so successive creates from the
      // same source (e.g. pasting the same palette twice) don't collide.
      const customPatternMatch = fromName.match(/ custom (\d+)$/);
      const baseName = customPatternMatch
        ? fromName.substring(
            0,
            fromName.length - customPatternMatch[0].length,
          )
        : fromName;

      const escapedBase = baseName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const numberPattern = new RegExp(`^${escapedBase} custom (\\d+)$`);
      const existingNumbers = palettes
        .map((p) => {
          const m = p.name.match(numberPattern);
          return m ? parseInt(m[1], 10) : 0;
        })
        .filter((n) => n > 0);
      const nextNumber =
        existingNumbers.length > 0 ? Math.max(...existingNumbers) + 1 : 1;

      const newPalette: UserPaletteEntry = {
        id: generateId(),
        name: `${baseName} custom ${nextNumber}`,
        colors: [...fromColors],
        isEditing: true,
      };

      setPalettes((prev) => [...prev, newPalette]);
      return newPalette;
    },
    [palettes, setPalettes],
  );

  // Update a palette's properties
  const updatePalette = useCallback(
    (id: string, changes: Partial<UserPaletteEntry>) => {
      setPalettes((prev) =>
        prev.map((p) => (p.id === id ? { ...p, ...changes } : p)),
      );
    },
    [setPalettes],
  );

  // Save a palette and exit edit mode
  const savePalette = useCallback(
    (id: string) => {
      setPalettes((prev) =>
        prev.map((p) =>
          p.id === id
            ? {
                ...p,
                isEditing: false,
                savedColors: undefined,
                savedName: undefined,
              }
            : p,
        ),
      );
    },
    [setPalettes],
  );

  // Cancel editing and restore previous values (only for previously saved palettes)
  const cancelPaletteEdit = useCallback(
    (id: string) => {
      setPalettes((prev) =>
        prev.map((p) => {
          if (p.id === id && p.savedName && p.savedColors) {
            return {
              ...p,
              name: p.savedName,
              colors: p.savedColors,
              isEditing: false,
              savedName: undefined,
              savedColors: undefined,
            };
          }
          return p;
        }),
      );
    },
    [setPalettes],
  );

  // Delete a palette permanently
  const deletePalette = useCallback(
    (id: string) => {
      setPalettes((prev) => prev.filter((p) => p.id !== id));
    },
    [setPalettes],
  );

  // Start editing a palette (used when clicking edit button)
  const startEditingPalette = useCallback(
    (id: string) => {
      setPalettes((prev) =>
        prev.map((p) => {
          if (p.id === id) {
            return {
              ...p,
              isEditing: true,
              savedName: p.name,
              savedColors: [...p.colors],
            };
          }
          return p;
        }),
      );
    },
    [setPalettes],
  );

  return {
    palettes,
    createPaletteInEditMode,
    updatePalette,
    savePalette,
    cancelPaletteEdit,
    deletePalette,
    startEditingPalette,
  };
}
