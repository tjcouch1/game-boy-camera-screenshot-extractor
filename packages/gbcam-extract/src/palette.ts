import { type GBImageData, GB_COLORS } from "./common.js";

export function parseHex(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/**
 * Apply a color palette to a grayscale GBImageData.
 * Input should contain only the 4 GB Camera grayscale values (0, 82, 165, 255).
 * Palette is 4 hex colors ordered lightest to darkest:
 *   [0] = color for 255 (white), [1] = color for 165, [2] = color for 82, [3] = color for 0 (black)
 */
export function applyPalette(
  grayscale: GBImageData,
  palette: [string, string, string, string],
): GBImageData {
  const colors = palette.map(parseHex);
  const colorMap = new Map<number, [number, number, number]>();
  colorMap.set(255, colors[0]);
  colorMap.set(165, colors[1]);
  colorMap.set(82, colors[2]);
  colorMap.set(0, colors[3]);

  const data = new Uint8ClampedArray(grayscale.data.length);
  for (let i = 0; i < grayscale.data.length; i += 4) {
    const gray = grayscale.data[i];
    let nearest = 0;
    let minDist = Infinity;
    for (const gv of GB_COLORS) {
      const d = Math.abs(gray - gv);
      if (d < minDist) { minDist = d; nearest = gv; }
    }
    const [r, g, b] = colorMap.get(nearest)!;
    data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
  }
  return { data, width: grayscale.width, height: grayscale.height };
}
