"""Detect the hand-drawn green rotated rectangle in each *-rectangle.png and
record the four corner positions (in original-image pixel coords) to corners.json."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).parent


def find_green_corners(img_bgr: np.ndarray) -> list[list[float]]:
    # Isolate strong green pixels (the hand-drawn rectangle).
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 150, 80), (85, 255, 255))

    # Close gaps in the 2px rectangle outline so it forms one connected component.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Keep only the largest connected component — discards stray green pixels
    # from the background (cables, fabric, etc.) that confuse minAreaRect.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        raise RuntimeError("no green pixels found")
    # stats[0] is the background; pick the largest non-background component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    component = (labels == biggest).astype(np.uint8) * 255

    pts = cv2.findNonZero(component)
    if pts is None:
        raise RuntimeError("no green pixels found")

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)  # 4x2 float32

    # Order corners as TL, TR, BR, BL (top-left first, going clockwise).
    pts4 = box.astype(float)
    s = pts4.sum(axis=1)
    d = np.diff(pts4, axis=1).ravel()
    tl = pts4[np.argmin(s)]
    br = pts4[np.argmax(s)]
    tr = pts4[np.argmin(d)]
    bl = pts4[np.argmax(d)]

    return [
        [round(float(tl[0]), 2), round(float(tl[1]), 2)],
        [round(float(tr[0]), 2), round(float(tr[1]), 2)],
        [round(float(br[0]), 2), round(float(br[1]), 2)],
        [round(float(bl[0]), 2), round(float(bl[1]), 2)],
    ]


CORNER_COLORS_BGR = {
    "TL": (0, 0, 255),      # red
    "TR": (0, 255, 255),    # yellow
    "BR": (255, 0, 255),    # magenta
    "BL": (255, 255, 0),    # cyan
}


def draw_overlay(img_bgr: np.ndarray, corners: list[list[float]]) -> np.ndarray:
    overlay = img_bgr.copy()
    pts = np.array(corners, dtype=np.int32)

    # Outline the detected quadrilateral.
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    # Distinct dot + label per corner so the ordering is visible.
    radius = max(6, min(img_bgr.shape[:2]) // 200)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), corners):
        color = CORNER_COLORS_BGR[label]
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, color, thickness=-1)
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, (0, 0, 0), thickness=2)
        cv2.putText(
            overlay,
            label,
            (int(round(x)) + radius + 4, int(round(y)) - radius - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    return overlay


def main() -> int:
    results: dict[str, dict] = {}
    files = sorted(HERE.glob("*-rectangle.png"))
    if not files:
        print("no *-rectangle.png files found", file=sys.stderr)
        return 1

    overlay_dir = HERE / "detected-corners"
    overlay_dir.mkdir(exist_ok=True)

    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  {f.name}: failed to load", file=sys.stderr)
            continue
        h, w = img.shape[:2]
        corners = find_green_corners(img)
        # Strip the "-rectangle" suffix to key by original test-image stem.
        stem = f.stem.removesuffix("-rectangle")
        results[stem] = {
            "imageSize": [w, h],
            "corners": {
                "topLeft": corners[0],
                "topRight": corners[1],
                "bottomRight": corners[2],
                "bottomLeft": corners[3],
            },
        }

        overlay_path = overlay_dir / f"{stem}-detected.png"
        cv2.imwrite(str(overlay_path), draw_overlay(img, corners))

        print(f"  {stem}: TL={corners[0]} TR={corners[1]} BR={corners[2]} BL={corners[3]}")

    out = HERE / "corners.json"
    out.write_text(
        json.dumps(
            {
                "description": (
                    "Corner positions of the hand-drawn green rectangle around the "
                    "Game Boy screen in each *-rectangle.png. Coordinates are in the "
                    "original image's pixel space (origin at top-left)."
                ),
                "images": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
