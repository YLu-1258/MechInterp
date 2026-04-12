"""Activation patching logic for the Mechanistic Interpretability Toolkit.

Implements head-level activation patching via TransformerLens hooks. For each
(layer, head) pair in the model, we replace that head's output activation in
the *target* prompt's forward pass with the corresponding cached activation from
the *source* prompt, then measure the change in the answer-token logit (ΔLogit).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer

from models import AnswerTokens, ModelInfo, PatchResponse, TensorData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_logits(
    logits: torch.Tensor,
    correct_id: Optional[int],
    incorrect_id: Optional[int],
) -> float:
    """Extract a scalar score from the final-position logits.

    If answer token IDs are provided, returns the logit *difference*
    (correct − incorrect). Otherwise returns the maximum logit at the last
    position (a proxy for model confidence).

    Args:
        logits: Shape [batch=1, seq_len, vocab_size].
        correct_id: Token ID of the correct answer token, or None.
        incorrect_id: Token ID of the incorrect answer token, or None.

    Returns:
        Scalar float score.
    """
    last_logits = logits[0, -1, :]  # [vocab_size]
    if correct_id is not None and incorrect_id is not None:
        return (last_logits[correct_id] - last_logits[incorrect_id]).item()
    return last_logits.max().item()


def _align_activation(
    cached: torch.Tensor,
    target_seq_len: int,
) -> torch.Tensor:
    """Align a cached activation tensor to a target sequence length.

    TransformerLens hook_result tensors have shape [batch, seq, n_heads, d_head].
    When source and target prompts differ in length we must truncate or pad along
    the sequence dimension so that the shapes are compatible.

    Args:
        cached: Source activation, shape [batch, src_seq_len, n_heads, d_head].
        target_seq_len: Sequence length of the target prompt.

    Returns:
        Tensor of shape [batch, target_seq_len, n_heads, d_head].
    """
    src_seq_len = cached.shape[1]
    if src_seq_len == target_seq_len:
        return cached
    if src_seq_len > target_seq_len:
        # Truncate: keep the first target_seq_len positions
        return cached[:, :target_seq_len, :, :]
    # Pad: repeat the last source position for the remaining slots
    pad_len = target_seq_len - src_seq_len
    pad = cached[:, -1:, :, :].expand(-1, pad_len, -1, -1)
    return torch.cat([cached, pad], dim=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_source_activations(
    model: HookedTransformer,
    source_prompt: str,
) -> tuple[object, torch.Tensor]:
    """Run the source prompt and cache hook_z activations for all layers.

    We only cache ``hook_z`` (the per-head output before W_O), which is
    sufficient for head-level activation patching and avoids storing every
    intermediate tensor.

    Args:
        model: A loaded HookedTransformer instance.
        source_prompt: The source prompt string.

    Returns:
        A (cache, tokens) tuple where *cache* is the TransformerLens
        ActivationCache (filtered to hook_z entries) and *tokens* is the
        token-ID tensor.
    """
    with torch.no_grad():
        tokens = model.to_tokens(source_prompt)
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name.endswith("hook_z"),
        )
    return cache, tokens


def get_clean_logit(
    model: HookedTransformer,
    target_prompt: str,
    correct_id: Optional[int],
    incorrect_id: Optional[int],
) -> tuple[float, torch.Tensor]:
    """Run a clean (unpatched) forward pass on the target prompt.

    Args:
        model: A loaded HookedTransformer instance.
        target_prompt: The target prompt string.
        correct_id: Token ID of the correct answer token, or None.
        incorrect_id: Token ID of the incorrect answer token, or None.

    Returns:
        A (clean_score, tokens) tuple.
    """
    with torch.no_grad():
        tokens = model.to_tokens(target_prompt)
        logits = model(tokens)
    score = _score_logits(logits, correct_id, incorrect_id)
    return score, tokens


def patch_single_head(
    model: HookedTransformer,
    target_tokens: torch.Tensor,
    source_cache: object,
    layer: int,
    head_idx: int,
    correct_id: Optional[int],
    incorrect_id: Optional[int],
) -> float:
    """Run target forward pass with one attention head's output replaced.

    Installs a hook on ``blocks.{layer}.attn.hook_result`` that replaces the
    activation for ``head_idx`` with the corresponding source-cached value,
    aligned to the target sequence length.

    Args:
        model: A loaded HookedTransformer instance.
        target_tokens: Token IDs for the target prompt, shape [1, target_seq].
        source_cache: ActivationCache returned by ``cache_source_activations``.
        layer: Layer index.
        head_idx: Head index within that layer.
        correct_id: Token ID of the correct answer token, or None.
        incorrect_id: Token ID of the incorrect answer token, or None.

    Returns:
        Scalar patched logit score.
    """
    target_seq_len = target_tokens.shape[1]
    hook_name = f"blocks.{layer}.attn.hook_z"
    cached_act = source_cache[hook_name]  # [1, src_seq, n_heads, d_head]
    aligned = _align_activation(cached_act, target_seq_len)

    def _hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        # value: [batch, seq, n_heads, d_head]
        value[:, :, head_idx, :] = aligned[:, :, head_idx, :]
        return value

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            target_tokens,
            fwd_hooks=[(hook_name, _hook)],
        )

    return _score_logits(patched_logits, correct_id, incorrect_id)


def run_activation_patching(
    model: HookedTransformer,
    source_prompt: str,
    target_prompt: str,
    answer_tokens: Optional[AnswerTokens],
) -> PatchResponse:
    """Run full head-level activation patching and return a structured response.

    This is the main driver called by the /patch route handler. It:
      1. Resolves answer token IDs (if provided).
      2. Caches source activations.
      3. Runs a clean target pass to establish baseline logit.
      4. Iterates all (layer, head) pairs, patching one at a time.
      5. Computes ΔLogit = patched_score − clean_score for each head.
      6. Returns a PatchResponse with the full matrix and metadata.

    Args:
        model: A loaded HookedTransformer instance.
        source_prompt: Source prompt whose activations serve as the patch.
        target_prompt: Target prompt that is patched.
        answer_tokens: Optional correct/incorrect token pair; if None the
            maximum-logit heuristic is used.

    Returns:
        A fully populated PatchResponse.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # ------------------------------------------------------------------
    # 1. Resolve answer token IDs
    # ------------------------------------------------------------------
    correct_id: Optional[int] = None
    incorrect_id: Optional[int] = None
    answer_token_info: dict | None = None

    if answer_tokens is not None:
        correct_ids = model.to_single_token(answer_tokens.correct)
        incorrect_ids = model.to_single_token(answer_tokens.incorrect)
        correct_id = int(correct_ids)
        incorrect_id = int(incorrect_ids)
        answer_token_info = {
            "correct_token": answer_tokens.correct,
            "correct_id": correct_id,
            "incorrect_token": answer_tokens.incorrect,
            "incorrect_id": incorrect_id,
        }

    # ------------------------------------------------------------------
    # 2. Cache source activations
    # ------------------------------------------------------------------
    source_cache, source_token_ids = cache_source_activations(model, source_prompt)
    source_tokens = model.to_str_tokens(source_token_ids[0])

    # ------------------------------------------------------------------
    # 3. Clean target pass (baseline logit)
    # ------------------------------------------------------------------
    clean_score, target_token_ids = get_clean_logit(
        model, target_prompt, correct_id, incorrect_id
    )
    target_tokens = model.to_str_tokens(target_token_ids[0])

    if answer_token_info is not None:
        answer_token_info["clean_logit_diff"] = clean_score

    logger.info(
        "Activation patching | model=%s | source_tokens=%d | target_tokens=%d | "
        "layers=%d | heads=%d | total_patches=%d | clean_score=%.4f",
        model.cfg.model_name,
        len(source_tokens),
        len(target_tokens),
        n_layers,
        n_heads,
        n_layers * n_heads,
        clean_score,
    )

    # ------------------------------------------------------------------
    # 4. Patching loop
    # ------------------------------------------------------------------
    delta_matrix = np.zeros((n_layers, n_heads), dtype=np.float32)

    for layer in range(n_layers):
        for head in range(n_heads):
            patched_score = patch_single_head(
                model,
                target_token_ids,
                source_cache,
                layer,
                head,
                correct_id,
                incorrect_id,
            )
            delta_matrix[layer, head] = patched_score - clean_score

        # Reset after each layer just to be safe (hooks are already
        # cleaned up by run_with_hooks, but belt-and-suspenders)
        model.reset_hooks()

    # ------------------------------------------------------------------
    # 5. Build response
    # ------------------------------------------------------------------
    model_info = ModelInfo(
        name=model.cfg.model_name,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=model.cfg.d_model,
    )

    return PatchResponse(
        delta_logits=TensorData(
            shape=list(delta_matrix.shape),
            data=delta_matrix.tolist(),
        ),
        source_tokens=list(source_tokens),
        target_tokens=list(target_tokens),
        answer_token_info=answer_token_info,
        model_info=model_info,
    )
