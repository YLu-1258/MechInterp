"""Logit lens computation — projects intermediate residual streams through the unembedding matrix.

This module implements the logit lens technique (nostalgebraist, 2020), which reveals
what a transformer would predict if it stopped processing at any intermediate layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


def compute_logit_lens_at_layer(
    model: HookedTransformer,
    cache,
    layer_idx: int,
    position: int,
    top_k: int = 10,
) -> dict[str, Any]:
    """Project the residual stream at a given layer through the unembedding matrix.

    Extracts ``cache["resid_post", layer_idx]``, applies the final layer norm,
    multiplies by the unembedding matrix ``W_U``, and returns the top-K tokens
    and their softmax probabilities at the requested sequence position.

    Args:
        model: A loaded HookedTransformer instance.
        cache: Activation cache from ``model.run_with_cache()``.
        layer_idx: Which transformer layer's residual stream to project.
        position: Token position to read from (supports negative indexing).
        top_k: How many top tokens to return.

    Returns:
        Dict with keys ``layer``, ``top_tokens``, ``top_probs``, ``top_token_ids``.
    """
    # resid_post shape: [batch, seq_len, d_model] → take batch 0
    resid = cache["resid_post", layer_idx][0]  # [seq_len, d_model]

    # Resolve negative position
    seq_len = resid.shape[0]
    if position < 0:
        position = seq_len + position
    if position < 0 or position >= seq_len:
        raise ValueError(
            f"Position {position} is out of range for sequence length {seq_len}."
        )

    hidden = resid[position]  # [d_model]

    # Apply final layer norm (fold in bias if present)
    normed = model.ln_final(hidden.unsqueeze(0).unsqueeze(0))  # [1, 1, d_model]
    normed = normed.squeeze()  # [d_model]

    # Project through unembedding matrix
    # W_U: [d_model, vocab_size], b_U: [vocab_size] (may not exist on all models)
    logits = normed @ model.W_U  # [vocab_size]
    if hasattr(model, "b_U") and model.b_U is not None:
        logits = logits + model.b_U

    # Softmax → probabilities
    probs = F.softmax(logits, dim=-1)  # [vocab_size]

    # Top-K
    top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]))

    # Decode token IDs → human-readable strings
    # Use the underlying tokenizer.decode() which reliably accepts plain Python ints.
    top_ids_list = top_ids.cpu().tolist()
    top_probs_list = top_probs.cpu().tolist()
    top_tokens = [model.tokenizer.decode([tid]) for tid in top_ids_list]

    return {
        "layer": layer_idx,
        "top_tokens": top_tokens,
        "top_probs": top_probs_list,
        "top_token_ids": top_ids_list,
    }


def detect_convergence(layer_results: list[dict[str, Any]]) -> int | None:
    """Find the earliest layer where the top-1 prediction stabilises to the final answer.

    A layer is considered converged when its top-1 predicted token matches the
    final layer's top-1 prediction AND every subsequent layer also agrees.

    Args:
        layer_results: List of per-layer dicts as returned by
            ``compute_logit_lens_at_layer``, ordered 0 → N-1.

    Returns:
        The earliest converged layer index, or ``None`` if no convergence found.
    """
    if not layer_results:
        return None

    # Top-1 token at the final layer is the reference
    final_top1 = layer_results[-1]["top_tokens"][0]

    # Walk backwards to find the earliest run of matching predictions
    convergence_layer: int | None = None
    for result in reversed(layer_results):
        if result["top_tokens"][0] == final_top1:
            convergence_layer = result["layer"]
        else:
            break  # Chain broken — stop here

    return convergence_layer
