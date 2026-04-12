"""Core analysis engine — runs a single forward pass and extracts interpretability data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from models import AnalyzeResponse, LogitAttribution, ModelInfo, TensorData

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


def analyze(prompt: str, model: HookedTransformer) -> AnalyzeResponse:
    """Run a full interpretability analysis on a prompt.

    Performs a single ``model.run_with_cache()`` forward pass and extracts:
    - Per-token direct logit attribution scores
    - Attention patterns for every layer and head
    - MLP activation norms per layer
    - Residual stream snapshots at each layer boundary

    Args:
        prompt: The input text to analyze.
        model: A loaded HookedTransformer instance.

    Returns:
        An AnalyzeResponse containing all extracted data.
    """
    cfg = model.cfg
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    d_model = cfg.d_model

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    tokens_tensor = model.to_tokens(prompt)  # [1, seq_len]
    seq_len = tokens_tensor.shape[1]
    token_strings = model.to_str_tokens(prompt)

    logger.info(
        "Analyzing prompt (%d tokens) with %s (L=%d, H=%d, D=%d)",
        seq_len, cfg.model_name, n_layers, n_heads, d_model,
    )

    # ------------------------------------------------------------------
    # Single forward pass
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens_tensor)

    try:
        # --------------------------------------------------------------
        # 1. Attention patterns: [n_layers, n_heads, seq_len, seq_len]
        # --------------------------------------------------------------
        attention_patterns = torch.stack(
            [cache["pattern", layer][0] for layer in range(n_layers)]
        )  # [n_layers, n_heads, seq_len, seq_len]

        # --------------------------------------------------------------
        # 2. MLP activation norms per layer: [n_layers, seq_len]
        # --------------------------------------------------------------
        mlp_norms = torch.stack(
            [cache["mlp_out", layer][0].norm(dim=-1) for layer in range(n_layers)]
        )  # [n_layers, seq_len]

        # --------------------------------------------------------------
        # 3. Residual stream snapshots: [n_layers+1, seq_len, d_model]
        #    Layer 0 = after embedding, layers 1..N = after each block
        # --------------------------------------------------------------
        residual_snapshots = torch.stack(
            [cache["resid_post", layer][0] for layer in range(n_layers)]
        )  # [n_layers, seq_len, d_model]

        # Prepend the embedding (resid_pre of layer 0) for the full picture
        resid_pre_0 = cache["resid_pre", 0][0]  # [seq_len, d_model]
        residual_stream = torch.cat(
            [resid_pre_0.unsqueeze(0), residual_snapshots], dim=0
        )  # [n_layers+1, seq_len, d_model]

        # --------------------------------------------------------------
        # 4. Direct logit attribution
        # --------------------------------------------------------------
        logit_attribution = _compute_logit_attribution(model, cache, logits, n_layers)

    finally:
        # Free the cache to release GPU/CPU memory
        del cache

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    return AnalyzeResponse(
        tokens=token_strings,
        logit_attribution=logit_attribution,
        attention_patterns=TensorData(
            shape=list(attention_patterns.shape),
            data=attention_patterns.cpu().tolist(),
        ),
        mlp_norms=TensorData(
            shape=list(mlp_norms.shape),
            data=mlp_norms.cpu().tolist(),
        ),
        residual_stream=TensorData(
            shape=list(residual_stream.shape),
            data=residual_stream.cpu().tolist(),
        ),
        model_info=ModelInfo(
            name=cfg.model_name,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
        ),
    )


def _compute_logit_attribution(
    model: HookedTransformer,
    cache,
    logits: torch.Tensor,
    n_layers: int,
) -> LogitAttribution:
    """Compute direct logit attribution via residual stream decomposition.

    Decomposes the residual stream into per-component contributions (each
    attention head output + each MLP output), then projects each through the
    unembedding matrix to obtain attribution scores for the predicted token
    at each position.

    Args:
        model: The HookedTransformer model.
        cache: The activation cache from run_with_cache().
        logits: The model output logits [1, seq_len, vocab_size].
        n_layers: Number of transformer layers.

    Returns:
        A LogitAttribution with by_component and by_token breakdowns.
    """
    cfg = model.cfg
    W_U = model.W_U  # [d_model, vocab_size]

    # Get the predicted token at each position (for attribution direction)
    predicted_tokens = logits[0].argmax(dim=-1)  # [seq_len]

    # Per-token aggregate attribution scores
    seq_len = predicted_tokens.shape[0]
    by_token = torch.zeros(seq_len, device=logits.device)

    # Per-component attribution
    by_component: dict[str, list[float]] = {}

    for layer in range(n_layers):
        # Per-head outputs: cache["z", layer] has shape [batch, seq_len, n_heads, head_dim]
        # These are the per-head values after the OV circuit but before summing across heads.
        # Project through W_O to get each head's contribution to the residual stream.
        z = cache["z", layer][0]  # [seq_len, n_heads, head_dim]
        W_O = model.W_O[layer]   # [n_heads, head_dim, d_model]

        for head in range(cfg.n_heads):
            # z[:, head, :] is [seq_len, head_dim]
            # W_O[head] is [head_dim, d_model]
            head_contribution = z[:, head, :] @ W_O[head]  # [seq_len, d_model]
            # Project through unembedding for the predicted token at each position
            head_logits = head_contribution @ W_U  # [seq_len, vocab_size]
            # Attribution = logit contribution toward the predicted token
            scores = head_logits[torch.arange(seq_len), predicted_tokens]
            component_name = f"L{layer}H{head}"
            by_component[component_name] = scores.cpu().tolist()
            by_token += scores

        # MLP contribution
        mlp_out = cache["mlp_out", layer][0]  # [seq_len, d_model]
        mlp_logits = mlp_out @ W_U  # [seq_len, vocab_size]
        mlp_scores = mlp_logits[torch.arange(seq_len), predicted_tokens]
        by_component[f"MLP{layer}"] = mlp_scores.cpu().tolist()
        by_token += mlp_scores

    return LogitAttribution(
        by_component=by_component,
        by_token=by_token.cpu().tolist(),
    )
