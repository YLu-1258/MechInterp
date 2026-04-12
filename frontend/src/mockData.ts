/* ── Realistic mock data for all four visualization views ───────────────
 *
 * Populated with data that would come from a GPT-2 small analysis of
 * "The capital of France is" — so the full visual design is visible
 * without a backend running.
 * ──────────────────────────────────────────────────────────────────────── */

import type { ModelInfo, LogitLensFrame } from './types';

// ── Model Info ────────────────────────────────────────────────────────────
export const MOCK_MODEL_INFO: ModelInfo = {
  name: 'gpt2-small',
  n_layers: 12,
  n_heads: 12,
  d_model: 768,
};

// ── Token Heatmap Data ────────────────────────────────────────────────────
export const MOCK_TOKENS = [
  'The', ' capital', ' of', ' France', ' is', ' Paris', ',', ' the',
  ' largest', ' city', ' in', ' the', ' country', '.',
];

export const MOCK_ATTRIBUTION_SCORES = [
  0.05, 0.12, 0.03, 0.87, 0.22, 0.91, 0.02, 0.08,
  0.15, 0.11, 0.04, 0.06, 0.19, 0.01,
];

// ── Attention Patterns ────────────────────────────────────────────────────
// Generate realistic autoregressive attention patterns for 12 layers × 12 heads
// Each pattern is seq_len × seq_len (14 × 14 for our mock tokens)
function generateAttentionPattern(seqLen: number, layer: number, head: number): number[][] {
  const pattern: number[][] = [];
  const seed = layer * 100 + head * 10;

  for (let q = 0; q < seqLen; q++) {
    const row = new Array(seqLen).fill(0);
    // Autoregressive: can only attend to positions ≤ q
    let sum = 0;
    for (let k = 0; k <= q; k++) {
      // Create varied patterns: some heads are more focused, some diffuse
      const focusFactor = ((seed + q * 7 + k * 13) % 17) / 17;
      let weight: number;

      if (head % 4 === 0) {
        // "Previous token" head: strong attention to q-1
        weight = k === q - 1 ? 3.0 + focusFactor : 0.1 + focusFactor * 0.3;
      } else if (head % 4 === 1) {
        // "Induction" head: attention to early positions
        weight = k < 3 ? 2.0 + focusFactor : 0.2 + focusFactor * 0.2;
      } else if (head % 4 === 2) {
        // "Positional" head: diagonal-ish attention
        weight = Math.exp(-Math.abs(q - k) * 0.5) + focusFactor * 0.3;
      } else {
        // "Diffuse" head: roughly uniform with some noise
        weight = 0.5 + focusFactor * 0.5;
      }

      // Layer depth affects sharpness
      if (layer > 6) {
        weight = Math.pow(weight, 1.5);
      }

      row[k] = weight;
      sum += weight;
    }
    // Normalize to probabilities
    for (let k = 0; k <= q; k++) {
      row[k] = sum > 0 ? row[k] / sum : 0;
    }
    pattern.push(row);
  }
  return pattern;
}

export const MOCK_ATTENTION_PATTERNS: number[][][][] = Array.from(
  { length: 12 },
  (_, l) => Array.from(
    { length: 12 },
    (_, h) => generateAttentionPattern(MOCK_TOKENS.length, l, h)
  )
);

// ── Logit Lens Data ───────────────────────────────────────────────────────
// 12 layers showing convergence to " Paris" around layer 8
export const MOCK_LOGIT_LENS_FRAMES: LogitLensFrame[] = [
  {
    type: 'layer', layer: 0,
    top_tokens: [' the', ' a', ' is', ' in', ' of'],
    top_probs: [0.08, 0.06, 0.05, 0.04, 0.03],
    top_token_ids: [262, 257, 318, 287, 286],
  },
  {
    type: 'layer', layer: 1,
    top_tokens: [' a', ' the', ' is', ' not', ' in'],
    top_probs: [0.09, 0.07, 0.05, 0.04, 0.03],
    top_token_ids: [257, 262, 318, 407, 287],
  },
  {
    type: 'layer', layer: 2,
    top_tokens: [' in', ' the', ' a', ' known', ' called'],
    top_probs: [0.11, 0.08, 0.06, 0.05, 0.04],
    top_token_ids: [287, 262, 257, 1900, 1444],
  },
  {
    type: 'layer', layer: 3,
    top_tokens: [' known', ' located', ' Par', ' a', ' the'],
    top_probs: [0.12, 0.09, 0.07, 0.06, 0.05],
    top_token_ids: [1900, 5140, 2547, 257, 262],
  },
  {
    type: 'layer', layer: 4,
    top_tokens: [' Par', ' located', ' known', ' a', ' the'],
    top_probs: [0.18, 0.10, 0.08, 0.05, 0.04],
    top_token_ids: [2547, 5140, 1900, 257, 262],
  },
  {
    type: 'layer', layer: 5,
    top_tokens: [' Paris', ' Par', ' Lyon', ' France', ' known'],
    top_probs: [0.22, 0.15, 0.08, 0.06, 0.04],
    top_token_ids: [6342, 2547, 17638, 4881, 1900],
  },
  {
    type: 'layer', layer: 6,
    top_tokens: [' Paris', ' Lyon', ' Par', ' France', ' Marseille'],
    top_probs: [0.35, 0.10, 0.07, 0.05, 0.04],
    top_token_ids: [6342, 17638, 2547, 4881, 26436],
  },
  {
    type: 'layer', layer: 7,
    top_tokens: [' Paris', ' Lyon', ' France', ' Marseille', ' the'],
    top_probs: [0.52, 0.08, 0.05, 0.04, 0.03],
    top_token_ids: [6342, 17638, 4881, 26436, 262],
  },
  {
    type: 'layer', layer: 8,
    top_tokens: [' Paris', ' Lyon', ' Marseille', ' France', ' Nice'],
    top_probs: [0.78, 0.04, 0.03, 0.02, 0.01],
    top_token_ids: [6342, 17638, 26436, 4881, 8695],
  },
  {
    type: 'layer', layer: 9,
    top_tokens: [' Paris', ' Lyon', ' Marseille', ' Nice', ' France'],
    top_probs: [0.85, 0.03, 0.02, 0.01, 0.01],
    top_token_ids: [6342, 17638, 26436, 8695, 4881],
  },
  {
    type: 'layer', layer: 10,
    top_tokens: [' Paris', ' Lyon', ' Marseille', ' Nice', ' Bordeaux'],
    top_probs: [0.91, 0.02, 0.01, 0.01, 0.01],
    top_token_ids: [6342, 17638, 26436, 8695, 42277],
  },
  {
    type: 'layer', layer: 11,
    top_tokens: [' Paris', ' Lyon', ' Marseille', ' Nice', ' Bordeaux'],
    top_probs: [0.95, 0.01, 0.01, 0.01, 0.00],
    top_token_ids: [6342, 17638, 26436, 8695, 42277],
  },
];

export const MOCK_CONVERGENCE_LAYER = 8;

// ── Patch Diff Data ───────────────────────────────────────────────────────
// IOI task: "When Mary and John went to the store, John/Mary gave a drink to"
export const MOCK_PATCH_SOURCE_PROMPT =
  'When Mary and John went to the store, John gave a drink to';
export const MOCK_PATCH_TARGET_PROMPT =
  'When Mary and John went to the store, Mary gave a drink to';
export const MOCK_PATCH_CORRECT_TOKEN = ' Mary';
export const MOCK_PATCH_INCORRECT_TOKEN = ' John';

// 12×12 ΔLogit matrix with known IOI circuit hot heads
function generateDeltaLogits(): number[][] {
  const matrix: number[][] = [];
  for (let l = 0; l < 12; l++) {
    const row: number[] = [];
    for (let h = 0; h < 12; h++) {
      // Base: small random noise
      let value = (Math.sin(l * 7 + h * 13) * 0.5) * 0.15;

      // IOI circuit heads get large positive values
      if (l === 9 && h === 6) value = 2.34;  // L9H6 — name mover
      if (l === 9 && h === 9) value = 1.87;  // L9H9 — name mover
      if (l === 10 && h === 0) value = 1.52; // L10H0 — name mover
      if (l === 7 && h === 3) value = 0.89;  // L7H3 — S-inhibition
      if (l === 7 && h === 9) value = 0.72;  // L7H9 — S-inhibition
      if (l === 8 && h === 6) value = 0.65;  // L8H6

      // Some negative heads (backup name movers)
      if (l === 10 && h === 7) value = -0.45;
      if (l === 11 && h === 10) value = -0.38;
      if (l === 10 && h === 10) value = -0.52;

      row.push(Math.round(value * 100) / 100);
    }
    matrix.push(row);
  }
  return matrix;
}

export const MOCK_DELTA_LOGITS = generateDeltaLogits();
export const MOCK_CLEAN_LOGIT_DIFF = 3.42;

// ── Source / Target tokens (IOI) ──────────────────────────────────────────
export const MOCK_SOURCE_TOKENS = [
  'When', ' Mary', ' and', ' John', ' went', ' to', ' the', ' store',
  ',', ' John', ' gave', ' a', ' drink', ' to',
];
export const MOCK_TARGET_TOKENS = [
  'When', ' Mary', ' and', ' John', ' went', ' to', ' the', ' store',
  ',', ' Mary', ' gave', ' a', ' drink', ' to',
];
