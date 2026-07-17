import { useState, useCallback } from "react";
import { flushSync } from "react-dom";
import type { GBImageData, Frame } from "gbcam-extract";
import { processPicture } from "gbcam-extract";
import { toast } from "sonner";
import type { ProcessedImage } from "./useImageHistory.js";

export interface CurrentImageProgress {
  filename: string;
  currentStep: string;
  index: number; // current image position (0-based)
  total: number; // total images to process
}

export interface ProcessingProgress {
  totalImages: number;
  completedImages: number;
  currentImageProgress: CurrentImageProgress | null;
  overallProgress: number; // 0-100, smooth across all images and steps
}

export interface ProcessFilesOptions {
  debug?: boolean;
  /**
   * Frames whose artwork may surround an already-processed input; passed to
   * the pipeline so wild-framed pipeline outputs fed back in can be cropped
   * at the right hole position.
   */
  knownFrames?: Frame[];
  /** Called once per successfully processed image, in file order. */
  onResult: (processed: ProcessedImage) => void;
}

function fileToGBImageData(file: File): Promise<GBImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      resolve({
        data: imageData.data,
        width: img.width,
        height: img.height,
      });
      URL.revokeObjectURL(img.src);
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error(`Failed to load image: ${file.name}`));
    };
    img.src = URL.createObjectURL(file);
  });
}

export function useProcessing() {
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState<ProcessingProgress>({
    totalImages: 0,
    completedImages: 0,
    currentImageProgress: null,
    overallProgress: 0,
  });

  // Pipeline steps for progress tracking
  const PIPELINE_STEPS = ["warp", "correct", "crop", "sample", "quantize"];
  const STEPS_COUNT = PIPELINE_STEPS.length;

  const calculateOverallProgress = (
    completedImages: number,
    currentStep: string,
    pct: number,
    totalImages: number,
  ): number => {
    if (totalImages === 0) return 0;
    // stepIndex < 0 means no pipeline step is in progress (Loading or done).
    // For an in-progress step, stepProgress = stepIndex + pct/100, so
    // onProgress(step, 100) counts that step as fully completed (1 unit) — the
    // previous version ignored pct and stalled at the step's start value.
    const stepIndex = PIPELINE_STEPS.indexOf(currentStep);
    const stepProgress = stepIndex >= 0 ? stepIndex + pct / 100 : 0;
    const totalSteps = totalImages * STEPS_COUNT;
    const progress =
      (completedImages * STEPS_COUNT + stepProgress) / totalSteps;
    return Math.max(0, Math.min(100, Math.round(progress * 100)));
  };

  const processFiles = useCallback(
    async (files: File[], options: ProcessFilesOptions) => {
      const { debug = false, knownFrames, onResult } = options;
      setProcessing(true);
      setProgress({
        totalImages: files.length,
        completedImages: 0,
        currentImageProgress: null,
        overallProgress: 0,
      });

      for (let fileIndex = 0; fileIndex < files.length; fileIndex++) {
        const file = files[fileIndex];
        try {
          setProgress((prev) => ({
            ...prev,
            currentImageProgress: {
              filename: file.name,
              currentStep: "Loading",
              index: fileIndex,
              total: files.length,
            },
            overallProgress: calculateOverallProgress(
              fileIndex,
              "",
              0,
              files.length,
            ),
          }));

          const gbImage = await fileToGBImageData(file);

          const start = performance.now();
          const result = await processPicture(gbImage, {
            debug,
            knownFrames,
            onProgress: (step, pct) => {
              flushSync(() => {
                setProgress((prev) => ({
                  ...prev,
                  currentImageProgress: prev.currentImageProgress
                    ? {
                        ...prev.currentImageProgress,
                        currentStep: step,
                      }
                    : null,
                  overallProgress: calculateOverallProgress(
                    fileIndex,
                    step,
                    pct,
                    files.length,
                  ),
                }));
              });
              // Yield to the event loop so the browser can repaint between
              // synchronous pipeline steps. Without this the bar appears frozen
              // during a single image because warp/correct/etc. all run inside
              // one JS turn. processPicture awaits this Promise.
              return new Promise<void>((resolve) => setTimeout(resolve, 0));
            },
          });
          const processingTime = performance.now() - start;

          onResult({ result, filename: file.name, processingTime });
        } catch (err) {
          const errorMsg = err instanceof Error ? err.message : String(err);
          console.error(`Failed to process ${file.name}:`, err);
          toast.error(`Failed to process ${file.name}: ${errorMsg}`);
        }

        const completedCount = fileIndex + 1;
        const nextProgressValue = calculateOverallProgress(
          completedCount,
          "",
          0,
          files.length,
        );
        setProgress({
          totalImages: files.length,
          completedImages: completedCount,
          currentImageProgress:
            completedCount < files.length
              ? {
                  filename: "",
                  currentStep: "",
                  index: completedCount,
                  total: files.length,
                }
              : null,
          overallProgress: nextProgressValue,
        });
      }

      setProcessing(false);
      setProgress({
        totalImages: files.length,
        completedImages: files.length,
        currentImageProgress: null,
        overallProgress: 100,
      });
    },
    [],
  );

  return { processFiles, processing, progress };
}
