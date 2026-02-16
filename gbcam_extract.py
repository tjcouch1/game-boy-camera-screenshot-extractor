#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import cv2
from PIL import Image

TARGET_W = 128
TARGET_H = 112


def debug_print(debug, *args):
    if debug:
        print("[DEBUG]", *args)


def find_camera_region(img, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = np.array([w // 2, h // 2])

    debug_print(debug, f"Input image size: {w}x{h}")

    # Threshold to remove black outer region
    threshold_value = 25
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    debug_print(debug, f"Threshold value used: {threshold_value}")
    debug_print(debug, f"Nonzero mask pixels: {np.count_nonzero(mask)}")

    # Morphological close
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or len(contours) == 0:
        raise ValueError("No contours detected.")

    debug_print(debug, f"Total contours found: {len(contours)}")

    TARGET_RATIO = 128 / 112
    debug_print(debug, f"Target aspect ratio: {TARGET_RATIO:.4f}")

    best_rect = None
    best_score = None

    for i, cnt in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw < w * 0.15 or ch < h * 0.15:
            debug_print(debug, f"Contour {i}: Rejected (too small) {cw}x{ch}")
            continue

        aspect = cw / ch
        aspect_error = abs(aspect - TARGET_RATIO)

        rect_center = np.array([x + cw // 2, y + ch // 2])
        center_dist = np.linalg.norm(rect_center - center)

        score = aspect_error * 500 + center_dist

        debug_print(
            debug,
            f"Contour {i}:",
            f"Rect=({x},{y},{cw},{ch})",
            f"Aspect={aspect:.4f}",
            f"AspectError={aspect_error:.4f}",
            f"CenterDist={center_dist:.2f}",
            f"Score={score:.2f}"
        )

        if aspect_error > 0.25:
            debug_print(debug, f"Contour {i}: Rejected (aspect mismatch)")
            continue

        if best_score is None or score < best_score:
            best_score = score
            best_rect = (x, y, cw, ch)
            debug_print(debug, f"Contour {i}: NEW BEST")

    if best_rect is None:
        raise ValueError("Could not isolate Game Boy Camera image region.")

    x, y, cw, ch = best_rect

    debug_print(debug, f"Selected rectangle: ({x},{y},{cw},{ch})")
    
    cropped = img[y:y+ch, x:x+cw]
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    debug_print(debug, "Refining crop using frame color detection...")

    h, w = cropped_gray.shape

    # --- Step 1: Estimate frame gray color ---
    # Sample a 5-pixel border around the outside
    sample_strip = np.concatenate([
        cropped_gray[0:5, :].flatten(),
        cropped_gray[h-5:h, :].flatten(),
        cropped_gray[:, 0:5].flatten(),
        cropped_gray[:, w-5:w].flatten()
    ])

    frame_gray = np.median(sample_strip)
    debug_print(debug, f"Estimated frame gray level: {frame_gray}")

    # Pixel considered frame if close to frame gray
    FRAME_TOL = 18

    def is_frame_pixel(val):
        return abs(int(val) - int(frame_gray)) < FRAME_TOL

    def row_frame_fraction(row):
        return np.mean([is_frame_pixel(v) for v in row])

    def col_frame_fraction(col):
        return np.mean([is_frame_pixel(v) for v in col])

    # --- Step 2: Trim top ---
    top = 0
    while top < h:
        frac = row_frame_fraction(cropped_gray[top])
        debug_print(debug, f"Top row {top} frame fraction: {frac:.3f}")
        if frac < 0.6:  # less than 60% frame → real image
            break
        top += 1

    # --- Step 3: Trim bottom ---
    bottom = h - 1
    while bottom > top:
        frac = row_frame_fraction(cropped_gray[bottom])
        debug_print(debug, f"Bottom row {bottom} frame fraction: {frac:.3f}")
        if frac < 0.6:
            break
        bottom -= 1

    # --- Step 4: Trim left ---
    left = 0
    while left < w:
        frac = col_frame_fraction(cropped_gray[:, left])
        debug_print(debug, f"Left col {left} frame fraction: {frac:.3f}")
        if frac < 0.6:
            break
        left += 1

    # --- Step 5: Trim right ---
    right = w - 1
    while right > left:
        frac = col_frame_fraction(cropped_gray[:, right])
        debug_print(debug, f"Right col {right} frame fraction: {frac:.3f}")
        if frac < 0.6:
            break
        right -= 1

    debug_print(debug, f"Final frame-trim box: top={top}, bottom={bottom}, left={left}, right={right}")

    cropped = cropped[top:bottom+1, left:right+1]

    debug_print(debug, f"Final cropped size: {cropped.shape[1]}x{cropped.shape[0]}")

    return cropped


def average_downscale(img, target_w, target_h, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_h, src_w = gray.shape

    debug_print(debug, f"Downscaling from {src_w}x{src_h} to {target_w}x{target_h}")

    result = np.zeros((target_h, target_w), dtype=np.float32)

    for ty in range(target_h):
        for tx in range(target_w):
            x0 = int(tx * src_w / target_w)
            x1 = int((tx + 1) * src_w / target_w)
            y0 = int(ty * src_h / target_h)
            y1 = int((ty + 1) * src_h / target_h)

            block = gray[y0:y1, x0:x1]
            result[ty, tx] = np.mean(block)

    debug_print(debug, f"Downscale complete.")

    return result


def quantize_2bit(img, debug=False):
    thresholds = np.percentile(img, [25, 50, 75])

    debug_print(debug, f"Quantization thresholds: {thresholds}")

    out = np.zeros_like(img)

    out[img <= thresholds[0]] = 0
    out[(img > thresholds[0]) & (img <= thresholds[1])] = 85
    out[(img > thresholds[1]) & (img <= thresholds[2])] = 170
    out[img > thresholds[2]] = 255

    return out.astype(np.uint8)


def process_file(path, debug=False):
    print(f"Processing: {path}")

    img = cv2.imread(path)
    if img is None:
        print(f"Skipping (could not load): {path}")
        return

    camera_region = find_camera_region(img, debug=debug)
    downscaled = average_downscale(camera_region, TARGET_W, TARGET_H, debug=debug)
    quantized = quantize_2bit(downscaled, debug=debug)

    output_img = Image.fromarray(quantized, mode='L')

    base, _ = os.path.splitext(path)
    out_path = base + "_gbcam.png"
    output_img.save(out_path)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract Game Boy Camera image from screenshot.")
    parser.add_argument("files", nargs="+", help="Input image files")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")

    args = parser.parse_args()

    for path in args.files:
        process_file(path, debug=args.debug)


if __name__ == "__main__":
    main()
