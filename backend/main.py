"""FastAPI application — Mechanistic Interpretability Toolkit backend."""

from __future__ import annotations

import asyncio
import logging
import time

import torch

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from analyzer import analyze
from logit_lens import compute_logit_lens_at_layer, detect_convergence
from model_registry import get_model, get_supported_models
from models import AnalyzeRequest, AnalyzeResponse, ErrorResponse, PatchRequest, PatchResponse, SupportedModel
from patcher import run_activation_patching

# Timeout waiting for the initial client message (seconds)
_WS_INITIAL_TIMEOUT: float = 30.0

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
# WebSocket: /logit-lens
# ---------------------------------------------------------------------------

@app.websocket("/logit-lens")
async def logit_lens_stream(websocket: WebSocket) -> None:
    """Stream logit lens projections layer-by-layer over a WebSocket.

    **Protocol (client → server, initial message):**
    ```json
    {
        "prompt": "The capital of France is",
        "model": "gpt2-small",
        "top_k": 10,
        "position": -1
    }
    ```

    **Protocol (server → client, per-layer frame):**
    ```json
    {
        "type": "layer",
        "layer": 5,
        "top_tokens": [" Paris", ...],
        "top_probs": [0.34, ...],
        "top_token_ids": [6342, ...]
    }
    ```

    **Protocol (server → client, final frame):**
    ```json
    {
        "type": "done",
        "convergence_layer": 8,
        "final_prediction": " Paris",
        "total_layers": 12
    }
    ```
    """
    await websocket.accept()
    logger.info("WS /logit-lens | connection accepted from %s", websocket.client)

    try:
        # ------------------------------------------------------------------
        # 1. Wait for the initial configuration message (with timeout)
        # ------------------------------------------------------------------
        try:
            raw = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=_WS_INITIAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("WS /logit-lens | timed out waiting for initial message")
            await websocket.send_json(
                {"type": "error", "message": "Timed out waiting for initial message."}
            )
            await websocket.close(code=1008)
            return
        except WebSocketDisconnect:
            logger.info("WS /logit-lens | client disconnected before sending message")
            return

        # ------------------------------------------------------------------
        # 2. Validate the initial message
        # ------------------------------------------------------------------
        prompt: str = raw.get("prompt", "").strip()
        model_name: str = raw.get("model", SupportedModel.GPT2_SMALL.value)
        top_k: int = int(raw.get("top_k", 10))
        position: int = int(raw.get("position", -1))

        if not prompt:
            await websocket.send_json(
                {"type": "error", "message": "'prompt' must be a non-empty string."}
            )
            await websocket.close(code=1008)
            return

        if model_name not in {m.value for m in SupportedModel}:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        f"Unknown model '{model_name}'. "
                        f"Supported: {[m.value for m in SupportedModel]}"
                    ),
                }
            )
            await websocket.close(code=1008)
            return

        top_k = max(1, min(top_k, 100))  # clamp to [1, 100]

        logger.info(
            "WS /logit-lens | model=%s | prompt_len=%d | top_k=%d | position=%d",
            model_name, len(prompt), top_k, position,
        )

        # ------------------------------------------------------------------
        # 3. Load model
        # ------------------------------------------------------------------
        try:
            model = get_model(model_name)
        except ValueError as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1008)
            return
        except RuntimeError as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1011)
            return

        # ------------------------------------------------------------------
        # 4. Validate prompt length
        # ------------------------------------------------------------------
        with torch.no_grad():
            tokens_tensor = model.to_tokens(prompt)
        seq_len = tokens_tensor.shape[1]
        max_ctx = model.cfg.n_ctx
        if seq_len > max_ctx:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        f"Prompt is {seq_len} tokens but {model_name} supports "
                        f"a maximum of {max_ctx} tokens."
                    ),
                }
            )
            await websocket.close(code=1008)
            return

        # ------------------------------------------------------------------
        # 5. Validate position
        # ------------------------------------------------------------------
        resolved_position = position if position >= 0 else seq_len + position
        if resolved_position < 0 or resolved_position >= seq_len:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        f"Position {position} is out of range for "
                        f"sequence length {seq_len}."
                    ),
                }
            )
            await websocket.close(code=1008)
            return

        # ------------------------------------------------------------------
        # 6. Run forward pass with cache (Option B: cache-based)
        # ------------------------------------------------------------------
        start = time.perf_counter()
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_tensor)

        n_layers = model.cfg.n_layers
        logger.info(
            "WS /logit-lens | forward pass done in %.2fs | streaming %d layers",
            time.perf_counter() - start, n_layers,
        )

        # ------------------------------------------------------------------
        # 7. Stream one frame per layer
        # ------------------------------------------------------------------
        layer_results: list[dict] = []
        try:
            for layer_idx in range(n_layers):
                frame = compute_logit_lens_at_layer(
                    model, cache, layer_idx, position, top_k
                )
                layer_results.append(frame)
                await websocket.send_json({"type": "layer", **frame})
        except WebSocketDisconnect:
            logger.info(
                "WS /logit-lens | client disconnected mid-stream at layer %d",
                layer_idx,
            )
            return
        except ValueError as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1008)
            return
        finally:
            del cache  # release GPU/CPU memory

        # ------------------------------------------------------------------
        # 8. Compute convergence and send the done frame
        # ------------------------------------------------------------------
        convergence_layer = detect_convergence(layer_results)
        final_prediction = layer_results[-1]["top_tokens"][0] if layer_results else None

        await websocket.send_json(
            {
                "type": "done",
                "convergence_layer": convergence_layer,
                "final_prediction": final_prediction,
                "total_layers": n_layers,
            }
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "WS /logit-lens | completed | model=%s | layers=%d | convergence=%s | %.2fs",
            model_name, n_layers, convergence_layer, elapsed,
        )

        await websocket.close()

    except WebSocketDisconnect:
        logger.info("WS /logit-lens | client disconnected")
    except Exception as e:
        logger.exception("WS /logit-lens | unexpected error: %s", e)
        try:
            await websocket.send_json(
                {"type": "error", "message": f"Internal server error: {e}"}
            )
            await websocket.close(code=1011)
        except Exception:
            pass  # client already gone


# ---------------------------------------------------------------------------
# Custom exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle validation errors that escape route handlers."""
    return HTTPException(status_code=400, detail=str(exc))
