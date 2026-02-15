#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2
from PIL import Image

TARGET_W = 128
TARGET_H = 112

def find_camera_region(img):
    """
    Detect the non-black rectangle approximately centered in the image.
    Assumes the border around it is mostly black.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to separate black border from content
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not detect camera image region.")

    # Choose the contour closest to center and reasonably large
    h, w = gray.shape
    center = np.array([w // 2, h // 2])

    best_rect = None
    best_score = None

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw < w * 0.2 or ch < h * 0.2:
            continue  # too small

        rect_center = np.array([x + cw // 2, y + ch // 2])
        dist = np.linalg.norm(rect_center - center)

        if best_score is None or dist < best_score:
            best_score = dist
            best_rect = (x, y, cw, ch)

    if best_rect is None:
        raise ValueError("No suitable camera rectangle found.")

    x, y, cw, ch = best_rect
    return img[y:y+ch, x:x+cw]


def average_downscale(img, target_w, target_h):
    """
    Downscale by averaging pixel blocks.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    src_h, src_w = gray.shape

    result = np.zeros((target_h, target_w), dtype=np.float32)

    for ty in range(target_h):
        for tx in range(target_w):
            x0 = int(tx * src_w / target_w)
            x1 = int((tx + 1) * src_w / target_w)
            y0 = int(ty * src_h / target_h)
            y1 = int((ty + 1) * src_h / target_h)

            block = gray[y0:y1, x0:x1]
            result[ty, tx] = np.mean(block)

    return result


def quantize_2bit(img):
    """
    Convert grayscale image to 4 evenly spaced brightness levels.
    """
    # Compute thresholds based on quartiles for better matching
    thresholds = np.percentile(img, [25, 50, 75])

    out = np.zeros_like(img)

    out[img <= thresholds[0]] = 0
    out[(img > thresholds[0]) & (img <= thresholds[1])] = 85
    out[(img > thresholds[1]) & (img <= thresholds[2])] = 170
    out[img > thresholds[2]] = 255

    return out.astype(np.uint8)


def process_file(path):
    print(f"Processing: {path}")

    img = cv2.imread(path)
    if img is None:
        print(f"Skipping (could not load): {path}")
        return

    camera_region = find_camera_region(img)
    downscaled = average_downscale(camera_region, TARGET_W, TARGET_H)
    quantized = quantize_2bit(downscaled)

    output_img = Image.fromarray(quantized, mode='L')

    base, _ = os.path.splitext(path)
    out_path = base + "_gbcam.png"
    output_img.save(out_path)

    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Drag image(s) onto this script or pass file paths as arguments.")
        sys.exit(1)

    for path in sys.argv[1:]:
        process_file(path)


if __name__ == "__main__":
    main()
