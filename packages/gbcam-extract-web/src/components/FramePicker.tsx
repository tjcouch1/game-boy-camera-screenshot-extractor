import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Frame, GBImageData } from "gbcam-extract";
import { composeFrame, applyPalette, appendDeduped } from "gbcam-extract";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/shadcn/components/popover";
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/shadcn/components/drawer";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shadcn/components/dialog";
import { Button } from "@/shadcn/components/button";
import {
  ChevronDown,
  Frame as FrameIcon,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/shadcn/utils/utils";
import { useIsMobile } from "../hooks/useIsMobile.js";
import type { FrameSelection } from "../types/frame-selection.js";
import { frameDisplayName } from "../utils/frame-display.js";
import {
  detectAndLoadFrames,
  disambiguateStem,
  fileToGBImageData,
  sanitizeFilenameStem,
} from "../utils/detectFrames.js";

const HOLE_W = 128;
const HOLE_H = 112;
/** Display size (in CSS pixels) of the trigger button's corner thumbnail. */
const TRIGGER_THUMB_PX = 24;

interface FramePickerProps {
  value: FrameSelection;
  onChange: (next: FrameSelection) => void;
  palette: [string, string, string, string];
  frames: Frame[];
  /** "result" includes a "Default — …" tile; "default" omits it. */
  mode: "default" | "result";
  /** Display label for the global default (used in "result" mode). */
  defaultFrameLabel?: string;
  /**
   * Currently-resolved default frame, shown in the "Default — …" tile when
   * `mode === "result"`. Null means the default is "no frame".
   */
  defaultFrame?: Frame | null;
  /**
   * Optional 128×112 grayscale image to compose into every thumbnail. When
   * omitted, thumbnails render with the lightest palette color in the hole.
   */
  image?: GBImageData;
  disabled?: boolean;
  /** IDs of frames originating from user uploads. Tiles for these get a delete button. */
  userFrameIds?: Set<string>;
  /** Persist a batch of new user-uploaded frames. Called from the upload flow. */
  onAddUserFrames?: (frames: Frame[]) => { added: number };
  /** Remove a previously-uploaded frame by ID. */
  onDeleteUserFrame?: (id: string) => void;
}

/** Build a dummy 128×112 lightest-color image for picker thumbnails. */
function buildEmptyImage(): GBImageData {
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
const EMPTY_USER_IDS: Set<string> = new Set();

/** Render a frame (or solid lightest color when no frame) onto a canvas. */
function FrameCanvas({
  frame,
  palette,
  image,
  width,
  height,
  className,
}: {
  frame: Frame | null;
  palette: [string, string, string, string];
  image: GBImageData;
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
        rendered = composeFrame(image, frame, palette);
      } catch {
        rendered = applyPalette(image, palette);
      }
    } else {
      rendered = applyPalette(image, palette);
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
  }, [frame, palette, image, width, height]);
  return <canvas ref={ref} className={className} style={{ imageRendering: "pixelated" }} />;
}

/**
 * Render the largest top-left square of a frame that doesn't overlap the
 * 128×112 hole. Side length = max(holeX, holeY). Shown in the picker's
 * trigger button so the selected frame's distinctive corner art is visible.
 */
function FrameCornerCanvas({
  frame,
  palette,
  displaySize,
  className,
}: {
  frame: Frame;
  palette: [string, string, string, string];
  displaySize: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const side = Math.max(frame.holeX, frame.holeY);
    canvas.width = displaySize;
    canvas.height = displaySize;
    const ctx = canvas.getContext("2d");
    if (!ctx || side <= 0) return;
    ctx.imageSmoothingEnabled = false;

    const W = frame.width;
    const gray = new Uint8ClampedArray(side * side * 4);
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        const v = frame.pixels[y * W + x];
        const i = (y * side + x) * 4;
        gray[i] = v;
        gray[i + 1] = v;
        gray[i + 2] = v;
        gray[i + 3] = 255;
      }
    }
    const rendered = applyPalette(
      { data: gray, width: side, height: side },
      palette,
    );

    const tmp = document.createElement("canvas");
    tmp.width = side;
    tmp.height = side;
    tmp
      .getContext("2d")!
      .putImageData(
        new ImageData(new Uint8ClampedArray(rendered.data), side, side),
        0,
        0,
      );
    ctx.drawImage(tmp, 0, 0, displaySize, displaySize);
  }, [frame, palette, displaySize]);
  return (
    <canvas
      ref={ref}
      className={className}
      style={{ imageRendering: "pixelated" }}
    />
  );
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
  defaultFrame,
  image,
  disabled,
  userFrameIds,
  onAddUserFrames,
  onDeleteUserFrame,
}: FramePickerProps) {
  const framesById = useMemo(() => new Map(frames.map((f) => [f.id, f] as const)), [frames]);
  const triggerFrame: Frame | null =
    value.kind === "frame"
      ? framesById.get(value.id) ?? null
      : value.kind === "default"
        ? defaultFrame ?? null
        : null;
  const triggerLabel = selectionLabel(value, framesById, defaultFrameLabel);
  const thumbnailImage = image ?? EMPTY_IMAGE;

  // Custom-frames support is enabled when the parent passes both the upload
  // and delete handlers. The picker still renders fine without them — the
  // section just doesn't appear (used in places where uploading isn't
  // appropriate, e.g. nested pickers).
  const customEnabled = Boolean(onAddUserFrames && onDeleteUserFrame);
  const userIds = userFrameIds ?? EMPTY_USER_IDS;

  const normals = useMemo(
    () => frames.filter((f) => f.type === "normal" && !userIds.has(f.id)),
    [frames, userIds],
  );
  const wilds = useMemo(
    () => frames.filter((f) => f.type === "wild" && !userIds.has(f.id)),
    [frames, userIds],
  );
  const customs = useMemo(
    () => frames.filter((f) => userIds.has(f.id)),
    [frames, userIds],
  );

  const [open, setOpen] = useState(false);
  const isMobile = useIsMobile();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const pendingDeleteFrame = pendingDeleteId
    ? framesById.get(pendingDeleteId) ?? null
    : null;

  const select = useCallback(
    (next: FrameSelection) => {
      onChange(next);
      setOpen(false);
    },
    [onChange],
  );

  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFiles = useCallback(
    async (fileList: FileList | null) => {
      if (!fileList || fileList.length === 0 || !onAddUserFrames) return;
      const files = Array.from(fileList);
      // Reset the input so re-uploading the same filename re-triggers onChange.
      if (fileInputRef.current) fileInputRef.current.value = "";

      // Build a running set of "taken" stems so multi-file uploads disambiguate
      // against each other and against the existing catalog.
      const takenStems = new Set(frames.map((f) => f.sheetStem));
      const survivors: Frame[] = [];
      let totalDuplicates = 0;
      let runningCatalog = frames;

      for (const file of files) {
        try {
          const image = await fileToGBImageData(file);
          const baseStem = sanitizeFilenameStem(file.name) || "custom-frame";
          const stem = disambiguateStem(baseStem, takenStems);
          takenStems.add(stem);
          const detected = detectAndLoadFrames(image, stem);
          // Dedup against the running catalog (built-in + previously-uploaded
          // + earlier survivors in this batch). appendDeduped drops by
          // fingerprint, so pixel-identical re-uploads are detected.
          const before = runningCatalog.length;
          const merged = appendDeduped(runningCatalog, detected);
          const newlyKept = merged.slice(before);
          totalDuplicates += detected.length - newlyKept.length;
          survivors.push(...newlyKept);
          runningCatalog = merged;
        } catch (err) {
          const reason =
            err instanceof Error ? err.message : "Unknown error.";
          toast.error(`${file.name}: ${reason}`);
        }
      }

      if (survivors.length > 0) {
        try {
          const { added } = onAddUserFrames(survivors);
          const parts: string[] = [];
          parts.push(`Added ${added} frame${added === 1 ? "" : "s"}.`);
          if (totalDuplicates > 0) {
            parts.push(
              `Skipped ${totalDuplicates} duplicate${totalDuplicates === 1 ? "" : "s"}.`,
            );
          }
          toast.success(parts.join(" "));
        } catch (err) {
          console.error("FramePicker: addUserFrames failed", err);
          if (err instanceof DOMException && err.name === "QuotaExceededError") {
            toast.error(
              "Out of storage. Delete some frames or images and try again.",
            );
          } else {
            toast.error("Failed to save frames.");
          }
        }
      } else if (totalDuplicates > 0) {
        toast.info(
          `Skipped ${totalDuplicates} duplicate${totalDuplicates === 1 ? "" : "s"}.`,
        );
      }
    },
    [frames, onAddUserFrames],
  );

  const confirmDelete = useCallback(() => {
    if (pendingDeleteId && onDeleteUserFrame) {
      onDeleteUserFrame(pendingDeleteId);
    }
    setPendingDeleteId(null);
  }, [pendingDeleteId, onDeleteUserFrame]);

  const triggerButton = (
    <Button variant="secondary" disabled={disabled} className="gap-2">
      <span
        className="inline-flex items-center justify-center overflow-hidden rounded border border-border"
        style={{ width: TRIGGER_THUMB_PX, height: TRIGGER_THUMB_PX }}
      >
        {triggerFrame ? (
          <FrameCornerCanvas
            frame={triggerFrame}
            palette={palette}
            displaySize={TRIGGER_THUMB_PX}
          />
        ) : (
          <FrameIcon className="size-4 text-muted-foreground" />
        )}
      </span>
      <span className="truncate max-w-[12em]">{triggerLabel}</span>
      <ChevronDown data-icon="inline-end" />
    </Button>
  );

  const body = (
    <>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {mode === "result" && (
          <FrameTile
            label={`Default${defaultFrameLabel ? ` — ${defaultFrameLabel}` : ""}`}
            selected={value.kind === "default"}
            onClick={() => select({ kind: "default" })}
            palette={palette}
            image={thumbnailImage}
            frame={defaultFrame ?? null}
            previewW={defaultFrame?.width ?? 160}
            previewH={defaultFrame?.height ?? 144}
          />
        )}
        <FrameTile
          label="No frame"
          selected={value.kind === "none"}
          onClick={() => select({ kind: "none" })}
          palette={palette}
          image={thumbnailImage}
          frame={null}
          previewW={HOLE_W}
          previewH={HOLE_H}
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
                onClick={() => select({ kind: "frame", id: f.id })}
                palette={palette}
                image={thumbnailImage}
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
                onClick={() => select({ kind: "frame", id: f.id })}
                palette={palette}
                image={thumbnailImage}
                frame={f}
                previewW={f.width}
                previewH={f.height}
              />
            ))}
          </div>
        </>
      )}
      {customEnabled && (
        <>
          <div className="mt-3 mb-2 flex items-center justify-between gap-2">
            <h4 className="text-sm font-semibold">Custom frames</h4>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleUploadClick}
            >
              <Upload data-icon="inline-start" />
              Upload
            </Button>
          </div>
          {customs.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              Upload a Game Boy Camera frame PNG. Sheets and individual frames
              are both supported.
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {customs.map((f) => (
                <FrameTile
                  key={f.id}
                  label={frameDisplayName(f)}
                  selected={value.kind === "frame" && value.id === f.id}
                  onClick={() => select({ kind: "frame", id: f.id })}
                  palette={palette}
                  image={thumbnailImage}
                  frame={f}
                  previewW={f.width}
                  previewH={f.height}
                  onDelete={() => setPendingDeleteId(f.id)}
                />
              ))}
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            multiple
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </>
      )}
    </>
  );

  // The delete confirmation Dialog is rendered as a sibling so it survives the
  // popover/drawer close (closing the picker shouldn't dismiss the dialog).
  const deleteDialog = (
    <Dialog
      open={pendingDeleteId !== null}
      onOpenChange={(o) => {
        if (!o) setPendingDeleteId(null);
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete this frame?</DialogTitle>
          <DialogDescription>
            {pendingDeleteFrame
              ? `"${frameDisplayName(pendingDeleteFrame)}" will be removed. This can't be undone.`
              : "This can't be undone."}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose render={<Button variant="secondary" />}>
            Cancel
          </DialogClose>
          <DialogClose
            render={<Button variant="destructive" onClick={confirmDelete} />}
          >
            Delete
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  if (isMobile) {
    return (
      <>
        <Drawer open={open} onOpenChange={setOpen}>
          <DrawerTrigger asChild>{triggerButton}</DrawerTrigger>
          <DrawerContent>
            <DrawerHeader>
              <DrawerTitle>Select a frame</DrawerTitle>
            </DrawerHeader>
            <div className="flex-1 overflow-y-auto px-4 pb-4">{body}</div>
          </DrawerContent>
        </Drawer>
        {deleteDialog}
      </>
    );
  }

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger render={triggerButton} />
        <PopoverContent className="w-[min(90vw,640px)] max-h-[70vh] overflow-auto p-3">
          {body}
        </PopoverContent>
      </Popover>
      {deleteDialog}
    </>
  );
}

function FrameTile({
  label,
  selected,
  onClick,
  palette,
  image,
  frame,
  previewW,
  previewH,
  onDelete,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
  palette: [string, string, string, string];
  image: GBImageData;
  frame: Frame | null;
  previewW: number;
  previewH: number;
  /** When set, render a small Trash2 button at top-end that calls this. */
  onDelete?: () => void;
}) {
  // The tile uses a wrapper <div> rather than a single <button> when a delete
  // affordance is needed: nesting buttons is invalid HTML, so the selection
  // click target becomes an inner button and the trash sits as its sibling.
  if (onDelete) {
    return (
      <div
        className={cn(
          "relative flex flex-col items-center justify-end gap-1 rounded border bg-card p-2 text-xs",
          selected && "ring-2 ring-primary",
        )}
      >
        <button
          type="button"
          onClick={onClick}
          className="flex flex-col items-center justify-end gap-1 w-full hover:bg-accent rounded"
        >
          <FrameCanvas
            frame={frame}
            palette={palette}
            image={image}
            width={previewW}
            height={previewH}
            className="max-w-full h-auto rounded border border-border"
          />
          <span className="truncate w-full text-center">{label}</span>
        </button>
        <Button
          type="button"
          variant="secondary"
          size="icon"
          aria-label="Delete frame"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="absolute top-2 end-2 size-7"
        >
          <Trash2 />
        </Button>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-center justify-end gap-1 rounded border bg-card p-2 text-xs hover:bg-accent",
        selected && "ring-2 ring-primary",
      )}
    >
      <FrameCanvas
        frame={frame}
        palette={palette}
        image={image}
        width={previewW}
        height={previewH}
        className="max-w-full h-auto rounded border border-border"
      />
      <span className="truncate w-full text-center">{label}</span>
    </button>
  );
}
