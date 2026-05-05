import { useState, useCallback, useEffect } from "react";
import { flushSync } from "react-dom";
import type { PipelineResult, GBImageData } from "gbcam-extract";
import { processPicture } from "gbcam-extract";
import type { FrameSelection } from "../types/frame-selection.js";
import {
  serializePipelineResult,
  deserializePipelineResult,
  isSerializedPipelineResult,
} from "../utils/serialization.js";

export interface ProcessingResult {
  result: PipelineResult;
  filename: string;
  processingTime: number;
  /** Per-image frame override. Undefined = follow global default. */
  frameOverride?: FrameSelection;
}

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

const RESULTS_STORAGE_KEY = "gbcam-current-results";

async function loadResultsFromStorage(): Promise<ProcessingResult[]> {
  try {
    const stored = localStorage.getItem(RESULTS_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      // Deserialize PipelineResult objects from PNG format
      // Map over results and await deserialization for each
      const deserialized = await Promise.all(
        parsed.map(async (item: any) => ({
          ...item,
          result: isSerializedPipelineResult(item.result)
            ? await deserializePipelineResult(item.result)
            : item.result,
        })),
      );
      return deserialized;
    }
  } catch (e) {
    console.error("Error parsing results from storage:", e);
    throw e;
  }
  return [];
}

function saveResultsToStorage(results: ProcessingResult[]) {
  // Serialize before storing to use compact base64 representation
  const serialized = results.map((item) => ({
    ...item,
    result: serializePipelineResult(item.result),
  }));
  localStorage.setItem(RESULTS_STORAGE_KEY, JSON.stringify(serialized));
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
  const [results, setResults] = useState<ProcessingResult[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load results from storage on mount
  useEffect(() => {
    let isMounted = true;
    loadResultsFromStorage()
      .then((loaded) => {
        if (isMounted) {
          setResults(loaded);
          setIsLoaded(true);
          // Save the loaded results back to storage to ensure consistency
          saveResultsToStorage(loaded);
        }
      })
      .catch(() => {
        // If loading fails, mark as loaded and clear storage
        if (isMounted) {
          setResults([]);
          setIsLoaded(true);
          localStorage.removeItem(RESULTS_STORAGE_KEY);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // Save results whenever they change (only after initial load)
  useEffect(() => {
    if (isLoaded) {
      saveResultsToStorage(results);
    }
  }, [results, isLoaded]);

  // Pipeline steps for progress tracking
  const PIPELINE_STEPS = ["warp", "correct", "crop", "sample", "quantize"];
  const STEPS_COUNT = PIPELINE_STEPS.length;

  const calculateOverallProgress = (
    completedImages: number,
    currentImageIndex: number,
    currentStep: string,
    totalImages: number,
  ): number => {
    // Each image has STEPS_COUNT steps
    // Progress is: (completed images * STEPS_COUNT + current step index) / (total images * STEPS_COUNT)
    // Clamp to 0-100 to avoid negative values
    const currentStepIndex = Math.max(0, PIPELINE_STEPS.indexOf(currentStep));
    const stepsForCompletedImages = completedImages * STEPS_COUNT;
    const stepsForCurrentImage = currentStepIndex;
    const totalSteps = totalImages * STEPS_COUNT;
    const progress =
      (stepsForCompletedImages + stepsForCurrentImage) / totalSteps;
    return Math.max(0, Math.min(100, Math.round(progress * 100)));
  };

  const processFiles = useCallback(async (files: File[], debug = false) => {
    setProcessing(true);
    setProgress({
      totalImages: files.length,
      completedImages: 0,
      currentImageProgress: null,
      overallProgress: 0,
    });
    setResults([]);

    const newResults: ProcessingResult[] = [];

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
            fileIndex,
            "",
            files.length,
          ),
        }));

        const gbImage = await fileToGBImageData(file);

        const start = performance.now();
        const result = await processPicture(gbImage, {
          debug,
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
                  fileIndex,
                  step,
                  files.length,
                ),
              }));
            });
          },
        });
        const processingTime = performance.now() - start;

        newResults.push({ result, filename: file.name, processingTime });
        setResults([...newResults]);

        const completedCount = fileIndex + 1;
        const nextProgressValue = calculateOverallProgress(
          completedCount,
          completedCount,
          "",
          files.length,
        );
        setProgress((prev) => ({
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
        }));
      } catch (err) {
        console.error(`Failed to process ${file.name}:`, err);
        const completedCount = fileIndex + 1;
        const nextProgressValue = calculateOverallProgress(
          completedCount,
          completedCount,
          "",
          files.length,
        );
        setProgress((prev) => ({
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
        }));
      }
    }

    setProcessing(false);
    setProgress({
      totalImages: files.length,
      completedImages: files.length,
      currentImageProgress: null,
      overallProgress: 100,
    });
  }, []);

  return { processFiles, processing, progress, results, setResults };
}
