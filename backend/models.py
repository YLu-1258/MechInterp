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
    QWEN3_06B = "qwen3-0.6b"
    QWEN3_17B = "qwen3-1.7b"


# ---------------------------------------------------------------------------
# Shared sub-schemas (referenced by both analyze and patch responses)
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Metadata about the model used for a request."""
    name: str
    n_layers: int
    n_heads: int
    d_model: int


class TensorData(BaseModel):
    """Generic container for a serialized tensor with its shape."""
    shape: list[int]
    data: list[Any]


# ---------------------------------------------------------------------------
# Analyze schemas
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


class AnalyzeResponse(BaseModel):
    """Response body for POST /analyze."""
    tokens: list[str] = Field(..., description="Human-readable token strings")
    logit_attribution: LogitAttribution
    attention_patterns: TensorData
    mlp_norms: TensorData
    residual_stream: TensorData
    model_info: ModelInfo


# ---------------------------------------------------------------------------
# Patch schemas
# ---------------------------------------------------------------------------

class AnswerTokens(BaseModel):
    """Pair of token strings used for logit-difference scoring."""
    correct: str = Field(..., description="The correct answer token string (e.g. ' Mary')")
    incorrect: str = Field(..., description="The incorrect answer token string (e.g. ' John')")


class PatchRequest(BaseModel):
    """Request body for POST /patch."""
    source_prompt: str = Field(..., description="Source prompt whose activations are cached")
    target_prompt: str = Field(..., description="Target prompt to run patching on")
    model: SupportedModel = Field(
        default=SupportedModel.GPT2_SMALL,
        description="Model to use for patching",
    )
    answer_tokens: AnswerTokens | None = Field(
        default=None,
        description="Optional correct/incorrect token pair for logit-difference scoring",
    )

    @field_validator("source_prompt", "target_prompt")
    @classmethod
    def prompt_must_be_non_empty(cls, v: str) -> str:
        """Validate that neither prompt is empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Prompt must be a non-empty string")
        return v


class PatchResponse(BaseModel):
    """Response body for POST /patch."""
    delta_logits: TensorData = Field(
        ...,
        description="ΔLogit matrix of shape [n_layers, n_heads]",
    )
    source_tokens: list[str] = Field(
        ..., description="Tokenized source prompt as human-readable strings"
    )
    target_tokens: list[str] = Field(
        ..., description="Tokenized target prompt as human-readable strings"
    )
    answer_token_info: dict[str, Any] | None = Field(
        default=None,
        description="Token IDs and clean logit info for the answer tokens",
    )
    model_info: ModelInfo


# ---------------------------------------------------------------------------
# Error schema
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str
    supported_models: list[str] | None = None
