/**
 * Bivariate polynomial surface fitting + least-squares solve.
 *
 * Extracted from correct.ts so other steps (e.g. frame-aware quantize) can
 * reuse the same fit-evaluate cycle without duplicating the math.
 */

/** Build Vandermonde design matrix: each column is x^dx * y^dy, dx+dy ≤ degree. */
function buildDesignMatrix(
  yn: Float64Array,
  xn: Float64Array,
  degree: number,
): Float64Array {
  const n = yn.length;
  let numTerms = 0;
  for (let dy = 0; dy <= degree; dy++) {
    for (let dx = 0; dx <= degree - dy; dx++) numTerms++;
  }
  const A = new Float64Array(n * numTerms);
  let col = 0;
  for (let dy = 0; dy <= degree; dy++) {
    for (let dx = 0; dx <= degree - dy; dx++) {
      for (let i = 0; i < n; i++) {
        A[i * numTerms + col] = Math.pow(yn[i], dy) * Math.pow(xn[i], dx);
      }
      col++;
    }
  }
  return A;
}

/** Solve A·c = b via normal equations (A^T A) c = A^T b with Gauss elim + partial pivot. */
function solveLeastSquares(
  A: Float64Array,
  b: Float64Array,
  rows: number,
  cols: number,
): Float64Array {
  const AtA = new Float64Array(cols * cols);
  for (let i = 0; i < cols; i++) {
    for (let j = i; j < cols; j++) {
      let sum = 0;
      for (let k = 0; k < rows; k++) sum += A[k * cols + i] * A[k * cols + j];
      AtA[i * cols + j] = sum;
      AtA[j * cols + i] = sum;
    }
  }
  const Atb = new Float64Array(cols);
  for (let i = 0; i < cols; i++) {
    let sum = 0;
    for (let k = 0; k < rows; k++) sum += A[k * cols + i] * b[k];
    Atb[i] = sum;
  }
  const stride = cols + 1;
  const aug = new Float64Array(cols * stride);
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < cols; j++) aug[i * stride + j] = AtA[i * cols + j];
    aug[i * stride + cols] = Atb[i];
  }
  for (let k = 0; k < cols; k++) {
    let maxVal = Math.abs(aug[k * stride + k]);
    let maxRow = k;
    for (let i = k + 1; i < cols; i++) {
      const v = Math.abs(aug[i * stride + k]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = i;
      }
    }
    if (maxRow !== k) {
      for (let j = k; j < stride; j++) {
        const tmp = aug[k * stride + j];
        aug[k * stride + j] = aug[maxRow * stride + j];
        aug[maxRow * stride + j] = tmp;
      }
    }
    const pivot = aug[k * stride + k];
    if (Math.abs(pivot) < 1e-12) continue;
    for (let i = k + 1; i < cols; i++) {
      const factor = aug[i * stride + k] / pivot;
      for (let j = k; j < stride; j++) aug[i * stride + j] -= factor * aug[k * stride + j];
    }
  }
  const x = new Float64Array(cols);
  for (let i = cols - 1; i >= 0; i--) {
    let sum = aug[i * stride + cols];
    for (let j = i + 1; j < cols; j++) sum -= aug[i * stride + j] * x[j];
    const diag = aug[i * stride + i];
    x[i] = Math.abs(diag) > 1e-12 ? sum / diag : 0;
  }
  return x;
}

/**
 * Fit a bivariate polynomial of given degree to (xs, ys → vs) and evaluate on
 * an HxW grid. Coordinates are normalised to [-1, 1] before fitting.
 *
 * Returns the H*W evaluated surface as a Float32Array (row-major).
 * If there are too few samples (< numTerms) or all samples are identical,
 * returns a flat surface at the sample mean.
 */
export function fitBivariateSurface(
  ys: number[],
  xs: number[],
  vs: number[],
  H: number,
  W: number,
  degree: number,
): Float32Array {
  const n = ys.length;
  let numTerms = 0;
  for (let dy = 0; dy <= degree; dy++) {
    for (let _dx = 0; _dx <= degree - dy; _dx++) numTerms++;
  }
  const surface = new Float32Array(H * W);
  if (n === 0) return surface;
  if (n < numTerms) {
    let s = 0;
    for (const v of vs) s += v;
    surface.fill(s / n);
    return surface;
  }
  const ynS = new Float64Array(n);
  const xnS = new Float64Array(n);
  const vS = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    ynS[i] = (ys[i] / H) * 2 - 1;
    xnS[i] = (xs[i] / W) * 2 - 1;
    vS[i] = vs[i];
  }
  const A = buildDesignMatrix(ynS, xnS, degree);
  const coeffs = solveLeastSquares(A, vS, n, numTerms);
  for (let y = 0; y < H; y++) {
    const yn = (y / H) * 2 - 1;
    for (let x = 0; x < W; x++) {
      const xn = (x / W) * 2 - 1;
      let val = 0;
      let col = 0;
      for (let dy = 0; dy <= degree; dy++) {
        for (let dx = 0; dx <= degree - dy; dx++) {
          val += coeffs[col] * Math.pow(yn, dy) * Math.pow(xn, dx);
          col++;
        }
      }
      surface[y * W + x] = val;
    }
  }
  return surface;
}
