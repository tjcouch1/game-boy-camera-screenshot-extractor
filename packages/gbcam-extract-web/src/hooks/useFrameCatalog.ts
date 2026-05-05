// packages/gbcam-extract-web/src/hooks/useFrameCatalog.ts
import { useEffect, useState } from "react";
import type { Frame, GBImageData } from "gbcam-extract";
import { splitSheet, dedupeFrames } from "gbcam-extract";
import { FRAME_SHEETS } from "../generated/FrameSheets.js";

export type FrameCatalogStatus = "loading" | "ready" | "error";

export interface FrameCatalog {
  status: FrameCatalogStatus;
  frames: Frame[];
  /** Map id -> Frame for O(1) lookup. */
  getFrameById(id: string): Frame | undefined;
  error?: string;
}

let cached: { frames: Frame[]; byId: Map<string, Frame> } | null = null;
let pending: Promise<{ frames: Frame[]; byId: Map<string, Frame> }> | null = null;

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

async function buildCatalog(): Promise<{ frames: Frame[]; byId: Map<string, Frame> }> {
  const all: Frame[] = [];
  for (const entry of FRAME_SHEETS) {
    const sheet = await fetchSheet(entry.url);
    all.push(...splitSheet(sheet, entry.stem));
  }
  const frames = dedupeFrames(all);
  const byId = new Map(frames.map((f) => [f.id, f] as const));
  return { frames, byId };
}

export function useFrameCatalog(): FrameCatalog {
  const [state, setState] = useState<{
    status: FrameCatalogStatus;
    frames: Frame[];
    byId: Map<string, Frame>;
    error?: string;
  }>(() =>
    cached
      ? { status: "ready", frames: cached.frames, byId: cached.byId }
      : { status: "loading", frames: [], byId: new Map() },
  );

  useEffect(() => {
    if (cached) return;
    let mounted = true;
    if (!pending) pending = buildCatalog();
    pending
      .then((result) => {
        cached = result;
        if (mounted) {
          setState({
            status: "ready",
            frames: result.frames,
            byId: result.byId,
          });
        }
      })
      .catch((err) => {
        pending = null;
        if (mounted) {
          setState({
            status: "error",
            frames: [],
            byId: new Map(),
            error: err instanceof Error ? err.message : String(err),
          });
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  return {
    status: state.status,
    frames: state.frames,
    error: state.error,
    getFrameById: (id) => state.byId.get(id),
  };
}
