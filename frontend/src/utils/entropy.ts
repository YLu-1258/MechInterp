/* ── Attention entropy computation ──────────────────────────────────────
 *
 * Shannon entropy of attention distributions. Higher entropy = more
 * diffuse (uniform) attention, lower = more focused (peaked).
 * ──────────────────────────────────────────────────────────────────────── */

const EPS = 1e-10;

/**
 * Compute Shannon entropy for a single attention distribution (one query position).
 * H(p) = -Σ p_i * log(p_i)
 */
export function computeEntropy(distribution: number[]): number {
  let h = 0;
  for (const p of distribution) {
    if (p > EPS) {
      h -= p * Math.log(p);
    }
  }
  return h;
}

/**
 * Compute the mean entropy of an attention head across all query positions.
 * matrix shape: [seq_len, seq_len] where each row is a probability distribution.
 */
export function meanHeadEntropy(matrix: number[][]): number {
  if (matrix.length === 0) return 0;
  let total = 0;
  for (const row of matrix) {
    total += computeEntropy(row);
  }
  return total / matrix.length;
}
