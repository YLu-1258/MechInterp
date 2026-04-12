/* ── Color scale utilities for the instrument aesthetic ─────────────────
 *
 * All color functions return CSS color strings and optional glow metadata.
 * Colors encode information — they are not decorative.
 * ──────────────────────────────────────────────────────────────────────── */

/** Interpolate between two hex colors. t ∈ [0, 1]. */
function lerpColor(a: string, b: string, t: number): string {
  const parse = (hex: string) => {
    const h = hex.replace('#', '');
    return [
      parseInt(h.slice(0, 2), 16),
      parseInt(h.slice(2, 4), 16),
      parseInt(h.slice(4, 6), 16),
    ];
  };
  const [r1, g1, b1] = parse(a);
  const [r2, g2, b2] = parse(b);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const bl = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}

/** Clamp value to [0, 1]. */
function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

/**
 * Map an attribution score to an amber color + glow intensity.
 * score=0 → dark base (#1a1a1f), score=max → full amber (#ffb74d).
 */
export function attributionToAmber(
  score: number,
  maxAbsScore: number
): { color: string; glowIntensity: number; textColor: string } {
  const t = maxAbsScore > 0 ? clamp01(Math.abs(score) / maxAbsScore) : 0;
  const color = lerpColor('#1a1a1f', '#ffb74d', t);
  const glowIntensity = t;
  // White text when background is dark enough (t < 0.55)
  const textColor = t > 0.55 ? '#0e0e10' : '#e4e4e7';
  return { color, glowIntensity, textColor };
}

/**
 * Map a ΔLogit value to blue (positive) or red (negative) with glow.
 * Near zero → dark base. High positive → electric blue. High negative → red.
 */
export function deltaLogitToColor(
  value: number,
  maxAbs: number
): { color: string; glowIntensity: number; glowType: 'blue' | 'red' | 'none'; textColor: string } {
  if (maxAbs === 0) {
    return { color: '#1a1a1f', glowIntensity: 0, glowType: 'none', textColor: '#e4e4e7' };
  }
  const normalized = clamp01(Math.abs(value) / maxAbs);
  let color: string;
  let glowType: 'blue' | 'red' | 'none';

  if (value > 0) {
    color = lerpColor('#1a1a1f', '#42a5f5', normalized);
    glowType = normalized > 0.1 ? 'blue' : 'none';
  } else {
    color = lerpColor('#1a1a1f', '#ef5350', normalized);
    glowType = normalized > 0.1 ? 'red' : 'none';
  }

  const textColor = normalized > 0.45 ? '#ffffff' : '#e4e4e7';
  return { color, glowIntensity: normalized, glowType, textColor };
}

/**
 * Map an attention weight [0, 1] to an amber color for the attention matrix.
 */
export function attentionToAmber(weight: number): { color: string; glowIntensity: number } {
  const t = clamp01(weight);
  const color = lerpColor('#1a1a1f', '#ffb74d', t);
  return { color, glowIntensity: t };
}

/**
 * Map entropy to color for the attention head grid.
 * Low entropy (focused) → dim. High entropy (diffuse) → amber glow.
 */
export function entropyToColor(
  entropy: number,
  maxEntropy: number
): { color: string; glowIntensity: number } {
  const t = maxEntropy > 0 ? clamp01(entropy / maxEntropy) : 0;
  const color = lerpColor('#1a1a1f', '#ffb74d', t);
  return { color, glowIntensity: t };
}

/**
 * Get contrasting text color based on perceived luminance.
 */
export function getTextColor(bgR: number, bgG: number, bgB: number): string {
  const luminance = 0.299 * bgR + 0.587 * bgG + 0.114 * bgB;
  return luminance > 140 ? '#0e0e10' : '#e4e4e7';
}
