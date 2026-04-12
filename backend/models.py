"""Pydantic request/response schemas for the Mechanistic Interpretability Toolkit API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------------

class SupportedModel(str, Enum):
    """Enumeration of model names accepted by the API."""
    GPT2_SMALL = "gpt2-small"
    GPT2_MEDIUM = "gpt2-medium"
    QWEN3_05B = "qwen3-0.5b"
    QWEN3_15B = "qwen3-1.5b"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""
    prompt: str = Field(..., description="Input text to analyze")
    model: SupportedModel = Field(
        default=SupportedModel.GPT2_SMALL,
        description="Model to use for analysis",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_be_non_empty(cls, v: str) -> str:
        """Validate that the prompt is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Prompt must be a non-empty string")
        return v


# ---------------------------------------------------------------------------
# Response sub-schemas
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Metadata about the model used for analysis."""
    name: str
    n_layers: int
    n_heads: int
    d_model: int


class LogitAttribution(BaseModel):
    """Direct logit attribution scores."""
    by_component: dict[str, Any] = Field(
        ...,
        description="Attribution scores keyed by component name (e.g. 'L0H0', 'mlp0')",
    )
    by_token: list[float] = Field(
        ...,
        description="Aggregate attribution score per token position",
    )


class TensorData(BaseModel):
    """Generic container for a serialized tensor with its shape."""
    shape: list[int]
    data: list[Any]


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """Response body for POST /analyze."""
    tokens: list[str] = Field(..., description="Human-readable token strings")
    logit_attribution: LogitAttribution
    attention_patterns: TensorData
    mlp_norms: TensorData
    residual_stream: TensorData
    model_info: ModelInfo


class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str
    supported_models: list[str] | None = None
