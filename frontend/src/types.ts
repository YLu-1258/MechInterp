/* ── Shared TypeScript types matching backend API schemas ──────────────── */

export interface ModelInfo {
  name: string;
  n_layers: number;
  n_heads: number;
  d_model: number;
}

export interface TensorData {
  shape: number[];
  data: number[];
}

export interface LogitAttribution {
  by_component: Record<string, number>;
  by_token: number[];
}

export interface AnalyzeResponse {
  tokens: string[];
  logit_attribution: LogitAttribution;
  attention_patterns: TensorData;
  mlp_norms: TensorData;
  residual_stream: TensorData;
  model_info: ModelInfo;
}

export interface AnswerTokens {
  correct: string;
  incorrect: string;
}

export interface PatchResponse {
  delta_logits: TensorData;
  source_tokens: string[];
  target_tokens: string[];
  answer_token_info: Record<string, unknown> | null;
  model_info: ModelInfo;
}

export interface LogitLensFrame {
  type: 'layer';
  layer: number;
  top_tokens: string[];
  top_probs: number[];
  top_token_ids: number[];
}

export interface LogitLensDone {
  type: 'done';
  convergence_layer: number | null;
  final_prediction: string | null;
  total_layers: number;
}

export type SupportedModel = 'gpt2-small' | 'gpt2-medium' | 'qwen3-0.5b' | 'qwen3-1.5b';

export type TabId = 'tokens' | 'heads' | 'lens' | 'patch';

export interface SelectedHead {
  layer: number;
  head: number;
}
