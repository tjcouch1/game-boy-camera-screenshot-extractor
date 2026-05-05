import { useRef, useEffect, useState, useCallback } from "react";
import type { PipelineResult, Frame } from "gbcam-extract";
import { applyPalette, composeFrame } from "gbcam-extract";
import {
  canShare,
  shareImage,
  copyImageToClipboard,
} from "../utils/shareImage.js";
import { sanitizePaletteName } from "../utils/filenames.js";
import { Button } from "@/shadcn/components/button";
import { Badge } from "@/shadcn/components/badge";
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
} from "@/shadcn/components/card";
import { toast } from "sonner";
import { X, Download, Share2, Copy as CopyIcon } from "lucide-react";
import { FramePicker } from "./FramePicker.js";
import type { FrameSelection } from "../types/frame-selection.js";

interface ResultCardProps {
  result: PipelineResult;
  filename: string;
  processingTime: number;
  palette: [string, string, string, string];
  paletteName: string;
  outputScale?: number;
  previewScale?: number;
  /** All frames available in the catalog (for the picker). */
  frames: Frame[];
  /** The frame selection chosen on this result. Undefined = follow default. */
  frameOverride: FrameSelection;
  onFrameOverrideChange: (next: FrameSelection) => void;
  /** Already resolved (effective) frame to render — null = no frame. */
  effectiveFrame: Frame | null;
  /** Display label for the "Default — …" picker tile. */
  defaultFrameLabel: string;
  onDelete?: () => void;
}

/** Build an off-screen canvas at the given scale for download/share/copy. */
function buildOutputCanvas(
  result: PipelineResult,
  palette: [string, string, string, string],
  effectiveFrame: Frame | null,
  scale: number,
): HTMLCanvasElement | null {
  try {
    if (!result.grayscale?.data) return null;
    let rendered;
    if (effectiveFrame) {
      try {
        rendered = composeFrame(result.grayscale, effectiveFrame, palette);
      } catch {
        rendered = applyPalette(result.grayscale, palette);
      }
    } else {
      rendered = applyPalette(result.grayscale, palette);
    }
    if (!rendered?.data?.length) return null;

    const canvas = document.createElement("canvas");
    canvas.width = rendered.width * scale;
    canvas.height = rendered.height * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;

    const tmp = document.createElement("canvas");
    tmp.width = rendered.width;
    tmp.height = rendered.height;
    tmp
      .getContext("2d")!
      .putImageData(
        new ImageData(
          new Uint8ClampedArray(rendered.data),
          rendered.width,
          rendered.height,
        ),
        0,
        0,
      );
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    return canvas;
  } catch {
    return null;
  }
}

export function ResultCard({
  result,
  filename,
  processingTime,
  palette,
  paletteName,
  outputScale = 1,
  previewScale = 2,
  frames,
  frameOverride,
  onFrameOverrideChange,
  effectiveFrame,
  defaultFrameLabel,
  onDelete,
}: ResultCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [shareSupported, setShareSupported] = useState(false);

  useEffect(() => {
    setShareSupported(canShare());
  }, []);

  // Render preview at previewScale, applying the effective frame if any.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      if (!result.grayscale?.data) return;
      let rendered;
      if (effectiveFrame) {
        try {
          rendered = composeFrame(result.grayscale, effectiveFrame, palette);
        } catch (err) {
          console.error("composeFrame failed; falling back to bare image", err);
          rendered = applyPalette(result.grayscale, palette);
        }
      } else {
        rendered = applyPalette(result.grayscale, palette);
      }
      if (!rendered?.data?.length) return;

      const scale = previewScale;
      canvas.width = rendered.width * scale;
      canvas.height = rendered.height * scale;
      const ctx = canvas.getContext("2d")!;
      ctx.imageSmoothingEnabled = false;

      const tmp = document.createElement("canvas");
      tmp.width = rendered.width;
      tmp.height = rendered.height;
      tmp
        .getContext("2d")!
        .putImageData(
          new ImageData(
            new Uint8ClampedArray(rendered.data),
            rendered.width,
            rendered.height,
          ),
          0,
          0,
        );
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    } catch (err) {
      console.error("Error rendering image:", err);
    }
  }, [result, palette, previewScale, effectiveFrame]);

  const handleDownload = useCallback(() => {
    const outputCanvas = buildOutputCanvas(result, palette, effectiveFrame, outputScale);
    if (!outputCanvas) return;
    const basename = filename.replace(/\.[^.]+$/, "");
    const sanitized = sanitizePaletteName(paletteName);
    const link = document.createElement("a");
    link.download = sanitized
      ? `${basename}_${sanitized}_gb.png`
      : `${basename}_gb.png`;
    link.href = outputCanvas.toDataURL("image/png");
    link.click();
  }, [result, palette, effectiveFrame, outputScale, filename, paletteName]);

  const handleShare = useCallback(async () => {
    const outputCanvas = buildOutputCanvas(result, palette, effectiveFrame, outputScale);
    if (!outputCanvas) return;
    try {
      await shareImage(
        outputCanvas,
        filename.replace(/\.[^.]+$/, "") + "_gb.png",
      );
    } catch (err) {
      console.error("Failed to share image:", err);
    }
  }, [result, palette, effectiveFrame, outputScale, filename]);

  const handleCopy = useCallback(async () => {
    const outputCanvas = buildOutputCanvas(result, palette, effectiveFrame, outputScale);
    if (!outputCanvas) return;
    try {
      await copyImageToClipboard(outputCanvas);
      toast.success("Image copied to clipboard");
    } catch (err) {
      const errorMsg = (err as Error).message || "Failed to copy";
      toast.error(`Copy failed: ${errorMsg}`);
      console.error("Failed to copy image:", err);
    }
  }, [result, palette, effectiveFrame, outputScale]);

  return (
    <Card className="p-3 sm:p-4">
      <CardHeader className="p-0 mb-3">
        <div className="min-w-0">
          <p className="text-sm font-medium truncate" title={filename}>
            {filename}
          </p>
          <Badge variant="secondary" className="mt-0.5">
            {processingTime.toFixed(0)}ms
          </Badge>
        </div>
        {onDelete && (
          <CardAction>
            <Button
              variant="destructive"
              size="icon"
              onClick={onDelete}
              aria-label="Delete result"
              className="size-7"
            >
              <X />
            </Button>
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="flex flex-col sm:flex-row gap-3 p-0">
        <div className="flex flex-col gap-2 items-start order-2 sm:order-1">
          <div className="flex flex-wrap gap-2 items-start content-start">
            <Button onClick={handleDownload}>
              <Download data-icon="inline-start" />
              Download PNG
            </Button>
            {shareSupported && (
              <Button variant="secondary" onClick={handleShare}>
                <Share2 data-icon="inline-start" />
                Share
              </Button>
            )}
            <Button variant="secondary" onClick={handleCopy} aria-label="Copy image">
              <CopyIcon data-icon="inline-start" />
              Copy
            </Button>
          </div>
          <FramePicker
            value={frameOverride}
            onChange={onFrameOverrideChange}
            palette={palette}
            frames={frames}
            mode="result"
            defaultFrameLabel={defaultFrameLabel}
          />
        </div>
        <canvas
          ref={canvasRef}
          className="rounded border self-start order-1 sm:order-2"
          style={{ imageRendering: "pixelated", maxWidth: "100%" }}
        />
      </CardContent>
    </Card>
  );
}
