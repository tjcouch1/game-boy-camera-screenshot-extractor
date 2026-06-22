import { useState, useEffect, useRef } from "react";
import { isPaletteInClipboard } from "../utils/paletteClipboard.js";

/**
 * Hook to track whether clipboard contains a valid palette.
 * To avoid permission loops, this checks only on mount and window focus.
 * If a permission error is detected, it stops checking automatically.
 */
export function useClipboardPaletteCheck(enabled: boolean = false) {
  const [hasClipboardPalette, setHasClipboardPalette] = useState(false);
  const permissionDenied = useRef(false);

  useEffect(() => {
    if (!enabled) {
      setHasClipboardPalette(false);
      return;
    }

    const checkClipboard = async () => {
      if (permissionDenied.current) return;
      
      try {
        const hasPalette = await isPaletteInClipboard();
        setHasClipboardPalette(hasPalette);
      } catch (err) {
        // If it throws an error (e.g., NotAllowedError), stop checking automatically
        permissionDenied.current = true;
        console.debug("Clipboard check failed, disabling auto-checks:", err);
      }
    };

    // Check immediately on enable
    checkClipboard();

    const handleFocus = () => {
      checkClipboard();
    };
    window.addEventListener("focus", handleFocus);

    return () => {
      window.removeEventListener("focus", handleFocus);
    };
  }, [enabled]);

  return { hasClipboardPalette };
}
