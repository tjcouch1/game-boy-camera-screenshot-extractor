import { useEffect, useMemo, useRef } from "react";
import type { Frame } from "gbcam-extract";
import { composeFrame, applyPalette } from "gbcam-extract";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/shadcn/components/popover";
import { Button } from "@/shadcn/components/button";
import { ChevronDown, Frame as FrameIcon } from "lucide-react";
import { cn } from "@/shadcn/utils/utils";
import type { FrameSelection } from "../types/frame-selection.js";

const HOLE_W = 128;
const HOLE_H = 112;

interface FramePickerProps {
  value: FrameSelection;
  onChange: (next: FrameSelection) => void;
  palette: [string, string, string, string];
  frames: Frame[];
  /** "result" includes a "Default — …" tile; "default" omits it. */
  mode: "default" | "result";
  /** Display label for the global default (used in "result" mode). */
  defaultFrameLabel?: string;
  disabled?: boolean;
}

/** Build a dummy 128×112 lightest-color image for picker thumbnails. */
function buildEmptyImage(): { data: Uint8ClampedArray; width: number; height: number } {
  const data = new Uint8ClampedArray(HOLE_W * HOLE_H * 4);
  for (let i = 0; i < HOLE_W * HOLE_H; i++) {
    data[i * 4 + 0] = 255;
    data[i * 4 + 1] = 255;
    data[i * 4 + 2] = 255;
    data[i * 4 + 3] = 255;
  }
  return { data, width: HOLE_W, height: HOLE_H };
}

const EMPTY_IMAGE = buildEmptyImage();

/** Render a frame (or solid lightest color when no frame) onto a canvas. */
function FrameCanvas({
  frame,
  palette,
  width,
  height,
  className,
}: {
  frame: Frame | null;
  palette: [string, string, string, string];
  width: number;
  height: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    let rendered;
    if (frame) {
      try {
        rendered = composeFrame(EMPTY_IMAGE, frame, palette);
      } catch {
        rendered = applyPalette(EMPTY_IMAGE, palette);
      }
    } else {
      rendered = applyPalette(EMPTY_IMAGE, palette);
    }
    const tmp = document.createElement("canvas");
    tmp.width = rendered.width;
    tmp.height = rendered.height;
    tmp
      .getContext("2d")!
      .putImageData(
        new ImageData(new Uint8ClampedArray(rendered.data), rendered.width, rendered.height),
        0,
        0,
      );
    ctx.drawImage(tmp, 0, 0, width, height);
  }, [frame, palette, width, height]);
  return <canvas ref={ref} className={className} style={{ imageRendering: "pixelated" }} />;
}

function frameDisplayName(frame: Frame): string {
  return `${frame.sheetStem} — ${frame.type} #${frame.index}`;
}

function selectionLabel(
  value: FrameSelection,
  framesById: Map<string, Frame>,
  defaultLabel: string | undefined,
): string {
  if (value.kind === "default") return `Default${defaultLabel ? ` — ${defaultLabel}` : ""}`;
  if (value.kind === "none") return "No frame";
  const f = framesById.get(value.id);
  return f ? frameDisplayName(f) : value.id;
}

export function FramePicker({
  value,
  onChange,
  palette,
  frames,
  mode,
  defaultFrameLabel,
  disabled,
}: FramePickerProps) {
  const framesById = useMemo(() => new Map(frames.map((f) => [f.id, f] as const)), [frames]);
  const triggerFrame: Frame | null =
    value.kind === "frame" ? framesById.get(value.id) ?? null : null;
  const triggerLabel = selectionLabel(value, framesById, defaultFrameLabel);

  const normals = useMemo(() => frames.filter((f) => f.type === "normal"), [frames]);
  const wilds = useMemo(() => frames.filter((f) => f.type === "wild"), [frames]);

  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button variant="secondary" disabled={disabled} className="gap-2">
            {triggerFrame || value.kind === "default" || value.kind === "none" ? (
              <span className="inline-flex size-4 items-center justify-center overflow-hidden rounded border border-border">
                {triggerFrame ? (
                  <FrameCanvas
                    frame={triggerFrame}
                    palette={palette}
                    width={16}
                    height={16}
                    className="size-4"
                  />
                ) : (
                  <FrameIcon className="size-3 text-muted-foreground" />
                )}
              </span>
            ) : null}
            <span className="truncate max-w-[12em]">{triggerLabel}</span>
            <ChevronDown data-icon="inline-end" />
          </Button>
        }
      />
      <PopoverContent className="w-[min(90vw,640px)] max-h-[70vh] overflow-auto p-3">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {mode === "result" && (
            <FrameTile
              label={`Default${defaultFrameLabel ? ` — ${defaultFrameLabel}` : ""}`}
              selected={value.kind === "default"}
              onClick={() => onChange({ kind: "default" })}
              palette={palette}
              frame={null}
              previewW={160}
              previewH={144}
            />
          )}
          <FrameTile
            label="No frame"
            selected={value.kind === "none"}
            onClick={() => onChange({ kind: "none" })}
            palette={palette}
            frame={null}
            previewW={160}
            previewH={144}
          />
        </div>
        {normals.length > 0 && (
          <>
            <h4 className="mt-3 mb-2 text-sm font-semibold">Normal frames</h4>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {normals.map((f) => (
                <FrameTile
                  key={f.id}
                  label={frameDisplayName(f)}
                  selected={value.kind === "frame" && value.id === f.id}
                  onClick={() => onChange({ kind: "frame", id: f.id })}
                  palette={palette}
                  frame={f}
                  previewW={160}
                  previewH={144}
                />
              ))}
            </div>
          </>
        )}
        {wilds.length > 0 && (
          <>
            <h4 className="mt-3 mb-2 text-sm font-semibold">Wild frames</h4>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {wilds.map((f) => (
                <FrameTile
                  key={f.id}
                  label={frameDisplayName(f)}
                  selected={value.kind === "frame" && value.id === f.id}
                  onClick={() => onChange({ kind: "frame", id: f.id })}
                  palette={palette}
                  frame={f}
                  previewW={f.width}
                  previewH={f.height}
                />
              ))}
            </div>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}

function FrameTile({
  label,
  selected,
  onClick,
  palette,
  frame,
  previewW,
  previewH,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
  palette: [string, string, string, string];
  frame: Frame | null;
  previewW: number;
  previewH: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-center gap-1 rounded border bg-card p-2 text-xs hover:bg-accent",
        selected && "ring-2 ring-primary",
      )}
    >
      <FrameCanvas
        frame={frame}
        palette={palette}
        width={previewW}
        height={previewH}
        className="max-w-full h-auto rounded border border-border"
      />
      <span className="truncate w-full text-center">{label}</span>
    </button>
  );
}
