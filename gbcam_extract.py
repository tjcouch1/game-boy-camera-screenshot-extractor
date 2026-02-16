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
    Detect the actual Game Boy Camera image inside the gray bordered frame.
    Uses aspect ratio filtering and contour hierarchy.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    center = np.array([w // 2, h // 2])

    # Step 1: Remove very dark background
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Clean small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 2: Find contours with hierarchy (important!)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None:
        raise ValueError("No contours detected.")

    best_rect = None
    best_score = None

    TARGET_RATIO = 128 / 112

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw < w * 0.15 or ch < h * 0.15:
            continue  # too small

        aspect = cw / ch
        aspect_error = abs(aspect - TARGET_RATIO)

        if aspect_error > 0.2:
            continue  # reject shapes far from 128x112 ratio

        rect_center = np.array([x + cw // 2, y + ch // 2])
        center_dist = np.linalg.norm(rect_center - center)

        # Score favors:
        # - correct aspect ratio
        # - closeness to center
        score = aspect_error * 500 + center_dist

        if best_score is None or score < best_score:
            best_score = score
            best_rect = (x, y, cw, ch)

    if best_rect is None:
        raise ValueError("Could not isolate Game Boy Camera image region.")

    x, y, cw, ch = best_rect

    # Small inward trim to avoid gray border bleed
    trim_x = int(cw * 0.02)
    trim_y = int(ch * 0.02)

    return img[
        y + trim_y : y + ch - trim_y,
        x + trim_x : x + cw - trim_x
    ]


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
