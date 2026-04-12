"""Validation script for the activation patching implementation.

Runs all acceptance criteria from the spec:
  1. IOI task — name mover heads (9.9, 9.6, 10.0) should show high ΔLogit
  2. Identical prompts — all ΔLogit values should be ≈ 0
  3. Mismatched sequence lengths — should complete without shape errors
  4. Error cases — empty prompt, invalid model name (schema validation)

Run from the backend directory with:
    python ../tests/test_patcher.py
"""

from __future__ import annotations

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import torch

from model_registry import get_model
from models import AnswerTokens
from patcher import run_activation_patching

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(condition: bool, msg: str) -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon} {msg}")
    return condition


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def test_ioi_name_mover_heads(model) -> bool:
    """Test 1: IOI task — name mover heads should show high ΔLogit."""
    separator("Test 1: IOI Name Mover Heads")

    source = "When Mary and John went to the store, John gave a drink to"
    target = "When Mary and John went to the store, Mary gave a drink to"
    answer = AnswerTokens(correct=" Mary", incorrect=" John")

    result = run_activation_patching(model, source, target, answer)

    delta = np.array(result.delta_logits.data)
    print(f"  ΔLogit matrix shape: {result.delta_logits.shape}")
    print(f"  Clean logit diff: {result.answer_token_info['clean_logit_diff']:.4f}")
    print(f"  Matrix min/max: {delta.min():.4f} / {delta.max():.4f}")

    # Check shape
    ok = True
    ok &= check(
        result.delta_logits.shape == [12, 12],
        f"Shape is [12, 12] (got {result.delta_logits.shape})",
    )

    # Known name mover heads: 9.9, 9.6, 10.0 (0-indexed)
    name_movers = [(9, 9), (9, 6), (10, 0)]
    for (l, h) in name_movers:
        val = delta[l, h]
        ok &= check(val > 0, f"Head {l}.{h} ΔLogit={val:.4f} > 0 (name mover)")

    # Print top-5 heads
    flat_idx = np.argsort(delta.ravel())[::-1][:5]
    top_heads = [(i // 12, i % 12, delta[i // 12, i % 12]) for i in flat_idx]
    print(f"  Top-5 heads: {[(f'L{l}H{h}', f'{v:.4f}') for l, h, v in top_heads]}")

    return ok


def test_identical_prompts(model) -> bool:
    """Test 2: Identical source and target — all ΔLogit ≈ 0."""
    separator("Test 2: Identical Prompts → ΔLogit ≈ 0")

    prompt = "The quick brown fox jumps over the lazy dog"
    result = run_activation_patching(model, prompt, prompt, answer_tokens=None)

    delta = np.array(result.delta_logits.data)
    max_abs = np.abs(delta).max()
    print(f"  Max |ΔLogit|: {max_abs:.6f}")

    ok = check(max_abs < 1e-3, f"All ΔLogit ≈ 0 (max |ΔLogit|={max_abs:.6f})")
    return ok


def test_mismatched_lengths(model) -> bool:
    """Test 3: Different sequence lengths — no shape errors."""
    separator("Test 3: Mismatched Sequence Lengths")

    source = "Hello"
    target = "The quick brown fox jumps over the lazy dog"
    answer = None

    try:
        result = run_activation_patching(model, source, target, answer)
        delta = np.array(result.delta_logits.data)
        ok = check(True, f"Completed without error, shape={result.delta_logits.shape}")
        ok &= check(
            result.delta_logits.shape == [12, 12],
            f"Shape is [12, 12]",
        )
    except Exception as e:
        ok = check(False, f"Raised unexpected exception: {e}")

    # Also test reverse direction (target shorter than source)
    try:
        result2 = run_activation_patching(model, target, source, answer)
        ok &= check(True, "Reverse direction (target shorter) also completed without error")
    except Exception as e:
        ok &= check(False, f"Reverse direction raised unexpected exception: {e}")

    return ok


def test_schema_validation() -> bool:
    """Test 4: Schema validation — empty prompt and invalid model."""
    separator("Test 4: Schema Validation")
    from pydantic import ValidationError
    from models import PatchRequest

    ok = True

    # Empty source prompt
    try:
        PatchRequest(source_prompt="", target_prompt="hello", model="gpt2-small")
        ok &= check(False, "Empty source_prompt should raise ValidationError")
    except (ValidationError, ValueError):
        ok &= check(True, "Empty source_prompt raises ValidationError ✓")

    # Empty target prompt
    try:
        PatchRequest(source_prompt="hello", target_prompt="   ", model="gpt2-small")
        ok &= check(False, "Whitespace-only target_prompt should raise ValidationError")
    except (ValidationError, ValueError):
        ok &= check(True, "Whitespace-only target_prompt raises ValidationError ✓")

    # Invalid model name
    try:
        PatchRequest(source_prompt="hello", target_prompt="world", model="bad-model")
        ok &= check(False, "Invalid model should raise ValidationError")
    except (ValidationError, ValueError):
        ok &= check(True, "Invalid model name raises ValidationError ✓")

    return ok


def main() -> None:
    print("\n" + "=" * 60)
    print("  Activation Patching Validation Suite")
    print("=" * 60)

    # Load model once for all tests
    print("\nLoading gpt2-small...")
    model = get_model("gpt2-small")
    print("Model loaded.\n")

    results = []
    results.append(("Test 1: IOI Name Mover Heads", test_ioi_name_mover_heads(model)))
    results.append(("Test 2: Identical Prompts → ΔLogit ≈ 0", test_identical_prompts(model)))
    results.append(("Test 3: Mismatched Sequence Lengths", test_mismatched_lengths(model)))
    results.append(("Test 4: Schema Validation", test_schema_validation()))

    separator("Summary")
    all_passed = True
    for name, passed in results:
        icon = PASS if passed else FAIL
        print(f"  {icon} {name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("  All tests passed! 🎉")
    else:
        print("  Some tests failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
