import type { TensorData } from '../types';

/**
 * Converts a 1D flattened TensorData object representing attention patterns
 * back into a nested 4D array: [n_layers][n_heads][seq_len][seq_len].
 */
export function unflattenAttention(tensor: TensorData): number[][][][] {
  const { shape, data } = tensor;

  if (shape.length !== 4) {
    throw new Error(`Expected expected 4D shape for attention patterns, got ${shape.length}D`);
  }

  const [nLayers, nHeads, seqLenQ, seqLenK] = shape;
  
  if (seqLenQ !== seqLenK) {
    throw new Error(`Expected square attention matrix per head, got ${seqLenQ}x${seqLenK}`);
  }

  const result: number[][][][] = [];
  let offset = 0;

  for (let l = 0; l < nLayers; l++) {
    const layerArr: number[][][] = [];
    for (let h = 0; h < nHeads; h++) {
      const headArr: number[][] = [];
      for (let q = 0; q < seqLenQ; q++) {
        const row: number[] = [];
        for (let k = 0; k < seqLenK; k++) {
          row.push(data[offset++]);
        }
        headArr.push(row);
      }
      layerArr.push(headArr);
    }
    result.push(layerArr);
  }

  // Sanity check
  if (offset !== data.length) {
    console.warn(`unflattenAttention: used ${offset} items but data length is ${data.length}`);
  }

  return result;
}
