// packages/gbcam-extract-web/src/hooks/useFrameCatalog.ts
import { useEffect, useMemo, useState } from "react";
import type { Frame, GBImageData } from "gbcam-extract";
import {
  splitSheet,
  loadIndividualFrame,
  dedupeFrames,
  appendDeduped,
} from "gbcam-extract";
import { FRAME_SHEETS } from "../generated/FrameSheets.js";
import { useUserFrames } from "./useUserFrames.js";

export type FrameCatalogStatus = "loading" | "ready" | "error";

export interface FrameCatalog {
  status: FrameCatalogStatus;
  frames: Frame[];
  /** Map id -> Frame for O(1) lookup. Includes both built-in and user frames. */
  getFrameById(id: string): Frame | undefined;
  /** Set of frame IDs originating from user uploads (used to render delete buttons). */
  userFrameIds: Set<string>;
  /** addFrames from useUserFrames, exposed for the picker's upload flow. */
  addUserFrames: (frames: Frame[]) => { added: number };
  /** deleteFrame from useUserFrames, exposed for the picker's delete flow. */
  deleteUserFrame: (id: string) => void;
  error?: string;
}

interface BuiltIn {
  frames: Frame[];
  byId: Map<string, Frame>;
}

let cachedBuiltIns: BuiltIn | null = null;
let pendingBuiltIns: Promise<BuiltIn> | null = null;

async function fetchSheet(url: string): Promise<GBImageData> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error(`Failed to decode ${url}`));
      el.src = objectUrl;
    });
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2d context unavailable");
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    return { data: imageData.data, width: img.width, height: img.height };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function buildBuiltIns(): Promise<BuiltIn> {
  const sheetFrames: Frame[] = [];
  const individualFrames: Frame[] = [];
  for (const entry of FRAME_SHEETS) {
    const image = await fetchSheet(entry.url);
    if (entry.kind === "individual") {
      individualFrames.push(loadIndividualFrame(image, entry.stem));
    } else {
      sheetFrames.push(...splitSheet(image, entry.stem));
    }
  }
  // Dedupe sheets first so cross-sheet duplicates merge with the existing
  // alphabetical-stem tiebreaker, then append individuals after — pixel
  // duplicates of an existing sheet frame just merge their alias stem in
  // rather than introducing a fresh entry.
  const dedupedSheets = dedupeFrames(sheetFrames);
  const frames = appendDeduped(dedupedSheets, individualFrames);
  const byId = new Map(frames.map((f) => [f.id, f] as const));
  return { frames, byId };
}

export function useFrameCatalog(): FrameCatalog {
  const userFrames = useUserFrames();
  const [builtIns, setBuiltIns] = useState<{
    status: FrameCatalogStatus;
    value: BuiltIn;
    error?: string;
  }>(() =>
    cachedBuiltIns
      ? { status: "ready", value: cachedBuiltIns }
      : {
          status: "loading",
          value: { frames: [], byId: new Map() },
        },
  );

  useEffect(() => {
    if (cachedBuiltIns) return;
    let mounted = true;
    if (!pendingBuiltIns) pendingBuiltIns = buildBuiltIns();
    pendingBuiltIns
      .then((result) => {
        cachedBuiltIns = result;
        if (mounted) setBuiltIns({ status: "ready", value: result });
      })
      .catch((err) => {
        pendingBuiltIns = null;
        if (mounted) {
          setBuiltIns({
            status: "error",
            value: { frames: [], byId: new Map() },
            error: err instanceof Error ? err.message : String(err),
          });
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  // Merge user frames into the built-ins on every render. appendDeduped
  // preserves the built-in order and drops any user upload that duplicates an
  // existing built-in by fingerprint.
  const merged = useMemo(() => {
    const baseFrames = builtIns.value.frames;
    if (userFrames.decodedFrames.length === 0) {
      return {
        frames: baseFrames,
        byId: builtIns.value.byId,
        userFrameIds: new Set<string>(),
      };
    }
    const userIds = new Set(userFrames.decodedFrames.map((f) => f.id));
    const frames = appendDeduped(baseFrames, userFrames.decodedFrames);
    const byId = new Map(frames.map((f) => [f.id, f] as const));
    return { frames, byId, userFrameIds: userIds };
  }, [
    builtIns.value.frames,
    builtIns.value.byId,
    userFrames.decodedFrames,
  ]);

  // Composite status: built-ins must be loaded; user frames decoding while we
  // already have built-ins shouldn't block the picker (the merged view is
  // valid without the not-yet-decoded uploads).
  const status: FrameCatalogStatus =
    builtIns.status === "error"
      ? "error"
      : builtIns.status === "loading"
        ? "loading"
        : "ready";

  return {
    status,
    frames: merged.frames,
    error: builtIns.error,
    getFrameById: (id) => merged.byId.get(id),
    userFrameIds: merged.userFrameIds,
    addUserFrames: userFrames.addFrames,
    deleteUserFrame: userFrames.deleteFrame,
  };
}
