"""FastAPI application — Mechanistic Interpretability Toolkit backend."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from analyzer import analyze
from model_registry import get_model, get_supported_models
from models import AnalyzeRequest, AnalyzeResponse, ErrorResponse, PatchRequest, PatchResponse
from patcher import run_activation_patching

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mechanistic Interpretability Toolkit",
    description="Real-time attention visualization, logit lens streaming, and activation patching.",
    version="0.1.0",
)

# Permissive CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_prompt(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run a full interpretability analysis on a prompt.

    Accepts a prompt and model name, runs a single forward pass through
    TransformerLens, and returns per-token logit attribution, attention
    patterns, MLP norms, and residual stream snapshots.
    """
    model_name = request.model.value
    prompt = request.prompt

    logger.info(
        "POST /analyze | model=%s | prompt_len=%d chars",
        model_name,
        len(prompt),
    )
    start = time.perf_counter()

    # Load model (lazy + cached)
    try:
        model = get_model(model_name)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    # Validate prompt length against model context window
    tokens = model.to_tokens(prompt)
    max_ctx = model.cfg.n_ctx
    if tokens.shape[1] > max_ctx:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt is {tokens.shape[1]} tokens, but {model_name} "
                f"supports a maximum of {max_ctx} tokens."
            ),
        )

    # Run analysis
    try:
        result = analyze(prompt, model)
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "oom" in error_msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"GPU out of memory during analysis. "
                    f"Try a shorter prompt or a smaller model (e.g. 'gpt2-small')."
                ),
            )
        raise HTTPException(status_code=503, detail=f"Analysis failed: {e}")

    elapsed = time.perf_counter() - start
    logger.info(
        "POST /analyze | model=%s | tokens=%d | %.2fs",
        model_name,
        len(result.tokens),
        elapsed,
    )

    return result


@app.post(
    "/patch",
    response_model=PatchResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def patch_activations(request: PatchRequest) -> PatchResponse:
    """Run head-level activation patching between source and target prompts.

    Caches all attention head outputs from the source prompt, then for each
    (layer, head) pair replaces that head's output in the target forward pass
    and measures the change in answer-token logit (ΔLogit).
    """
    model_name = request.model.value
    src_len = len(request.source_prompt)
    tgt_len = len(request.target_prompt)

    logger.info(
        "POST /patch | model=%s | source_len=%d chars | target_len=%d chars",
        model_name,
        src_len,
        tgt_len,
    )
    start = time.perf_counter()

    # Load model (lazy + cached)
    try:
        model = get_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Run patching
    try:
        result = run_activation_patching(
            model,
            request.source_prompt,
            request.target_prompt,
            request.answer_tokens,
        )
    except ValueError as e:
        # e.g. answer token not in vocabulary
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "oom" in error_msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    "GPU out of memory during patching. "
                    "Try a shorter prompt or a smaller model (e.g. 'gpt2-small')."
                ),
            )
        raise HTTPException(status_code=503, detail=f"Patching failed: {e}")

    elapsed = time.perf_counter() - start
    n_layers, n_heads = result.delta_logits.shape
    logger.info(
        "POST /patch | model=%s | patches=%d | %.2fs",
        model_name,
        n_layers * n_heads,
        elapsed,
    )

    return result


# ---------------------------------------------------------------------------
# Custom exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle validation errors that escape route handlers."""
    return HTTPException(status_code=400, detail=str(exc))
