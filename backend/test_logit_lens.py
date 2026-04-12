"""Phase 4 test suite for the /logit-lens WebSocket endpoint.

Covers all acceptance criteria from tasks.md:
  1. Frames arrive for each layer followed by a done frame.
  2. Convergence layer is reported in the done frame.
  3. Top tokens are human-readable strings (not raw IDs / bytes).
  4. Client disconnection mid-stream does not crash the server.
  5. Invalid initial message → error frame received.
  6. `position` parameter works for non-final token positions.

Run with:
    python test_logit_lens.py

The server must already be running at ws://localhost:8000/logit-lens.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import websockets

WS_URL = "ws://localhost:8000/logit-lens"
PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    results.append((name, ok, detail))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

async def collect_frames(ws, max_frames: int = 200) -> list[dict]:
    """Collect all frames until a done/error frame or max_frames is reached."""
    frames = []
    async for raw in ws:
        frame = json.loads(raw)
        frames.append(frame)
        if frame.get("type") in ("done", "error"):
            break
        if len(frames) >= max_frames:
            break
    return frames


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

async def test_happy_path() -> None:
    """TC-1: Valid prompt streams N layer frames + 1 done frame."""
    print("\n[TC-1] Happy path — 'The capital of France is'")
    prompt = "The capital of France is"

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"prompt": prompt, "model": "gpt2-small", "top_k": 10, "position": -1}))
        frames = await collect_frames(ws)

    layer_frames = [f for f in frames if f.get("type") == "layer"]
    done_frames  = [f for f in frames if f.get("type") == "done"]
    err_frames   = [f for f in frames if f.get("type") == "error"]

    record("No error frames", len(err_frames) == 0, str(err_frames))

    n_layers = done_frames[0].get("total_layers") if done_frames else None
    record("Received exactly one done frame", len(done_frames) == 1)
    record(
        f"Received one frame per layer ({n_layers} layers)",
        n_layers is not None and len(layer_frames) == n_layers,
        f"got {len(layer_frames)}, expected {n_layers}",
    )

    # TC-2: Convergence layer reported
    conv = done_frames[0].get("convergence_layer") if done_frames else None
    final_pred = done_frames[0].get("final_prediction") if done_frames else None
    record("Convergence layer is reported (not None)", conv is not None, f"convergence_layer={conv}")
    record(f"Final prediction non-empty", bool(final_pred), f"final_prediction={final_pred!r}")
    if final_pred:
        print(f"         → Final prediction: {final_pred!r}, converges at layer {conv}")

    # TC-3: Top tokens are human-readable strings
    if layer_frames:
        sample = layer_frames[0]
        top_tokens = sample.get("top_tokens", [])
        top_probs  = sample.get("top_probs", [])
        top_ids    = sample.get("top_token_ids", [])
        record("top_tokens is a list of strings", all(isinstance(t, str) for t in top_tokens), str(top_tokens[:3]))
        record("top_probs is a list of floats",   all(isinstance(p, float) for p in top_probs))
        record("top_token_ids is a list of ints", all(isinstance(i, int) for i in top_ids))
        record("top_tokens has 10 entries",        len(top_tokens) == 10, f"got {len(top_tokens)}")
        total_prob = sum(top_probs)
        record(
            "Top-K probs are reasonable (≥ 0.01 total)",
            total_prob >= 0.01,
            f"sum={total_prob:.4f}",
        )

    # Verify layer indices are sequential 0..N-1
    indices = [f["layer"] for f in layer_frames]
    record(
        "Layer indices are sequential 0..N-1",
        indices == list(range(len(layer_frames))),
        str(indices[:5]),
    )


async def test_position_parameter() -> None:
    """TC-6: position parameter works for non-final token positions."""
    print("\n[TC-6] Position parameter — position=0 (first token)")
    payload = {"prompt": "Hello world", "model": "gpt2-small", "top_k": 5, "position": 0}

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps(payload))
        frames = await collect_frames(ws)

    layer_frames = [f for f in frames if f.get("type") == "layer"]
    done_frames  = [f for f in frames if f.get("type") == "done"]
    err_frames   = [f for f in frames if f.get("type") == "error"]

    record("No error frames for position=0", len(err_frames) == 0, str(err_frames))
    record("Received done frame for position=0", len(done_frames) == 1)
    record("Layer frames received for position=0", len(layer_frames) > 0, f"got {len(layer_frames)}")
    if layer_frames:
        sample = layer_frames[0]
        record(
            "top_k=5 honoured",
            len(sample.get("top_tokens", [])) == 5,
            str(sample.get("top_tokens")),
        )


async def test_mid_stream_disconnect() -> None:
    """TC-4: Client disconnects mid-stream — server must not crash."""
    print("\n[TC-4] Mid-stream client disconnection")
    payload = {"prompt": "The quick brown fox", "model": "gpt2-small", "top_k": 5, "position": -1}

    layer_count = 0
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps(payload))
            try:
                async for raw in ws:
                    frame = json.loads(raw)
                    if frame.get("type") == "layer":
                        layer_count += 1
                    if layer_count >= 3:
                        # Force-close the connection abruptly
                        await ws.close()
                        break
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                # Server closed after detecting our disconnect — expected
                pass
    except (
        websockets.exceptions.ConnectionClosedOK,
        websockets.exceptions.ConnectionClosedError,
    ):
        pass  # Clean close triggered by the server — expected
    except Exception as e:
        record("Mid-stream disconnect (no client exception)", False, str(e))
        return

    record(
        "Client closed after ≥2 frames without exception",
        layer_count >= 2,
        f"received {layer_count} layer frames before disconnect",
    )

    # Give the server a moment to handle the disconnect
    await asyncio.sleep(1.0)

    # Verify server is still alive by hitting /health
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        body = json.loads(resp.read())
        record("Server still alive after mid-stream disconnect", body.get("status") == "ok")
    except Exception as e:
        record("Server still alive after mid-stream disconnect", False, str(e))


async def test_empty_prompt() -> None:
    """TC-5a: Empty prompt → error frame."""
    print("\n[TC-5a] Invalid input — empty prompt")
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"prompt": "", "model": "gpt2-small"}))
        frames = await collect_frames(ws)

    err_frames = [f for f in frames if f.get("type") == "error"]
    record("Error frame received for empty prompt", len(err_frames) >= 1, str(err_frames))
    if err_frames:
        record("Error message is non-empty", bool(err_frames[0].get("message")), err_frames[0].get("message"))


async def test_bad_model() -> None:
    """TC-5b: Unknown model name → error frame."""
    print("\n[TC-5b] Invalid input — unknown model")
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"prompt": "Hello", "model": "gpt-99-turbo"}))
        frames = await collect_frames(ws)

    err_frames = [f for f in frames if f.get("type") == "error"]
    record("Error frame received for bad model", len(err_frames) >= 1, str(err_frames))


async def test_out_of_bounds_position() -> None:
    """TC-5c: Position out of range → error frame."""
    print("\n[TC-5c] Invalid input — out-of-range position")
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"prompt": "Hi", "model": "gpt2-small", "position": 999}))
        frames = await collect_frames(ws)

    err_frames = [f for f in frames if f.get("type") == "error"]
    record("Error frame received for out-of-bounds position", len(err_frames) >= 1, str(err_frames))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> int:
    print("=" * 60)
    print("  Logit Lens WebSocket — Phase 4 Test Suite")
    print("  Target:", WS_URL)
    print("=" * 60)

    # Quick connectivity check
    try:
        async with websockets.connect(WS_URL, open_timeout=5):
            pass
    except Exception as e:
        print(f"\n\033[31mERROR: Cannot connect to {WS_URL}\033[0m")
        print(f"       Make sure the server is running: uvicorn main:app --reload")
        print(f"       Details: {e}")
        return 1

    t0 = time.perf_counter()

    await test_happy_path()
    await test_position_parameter()
    await test_mid_stream_disconnect()
    await test_empty_prompt()
    await test_bad_model()
    await test_out_of_bounds_position()

    elapsed = time.perf_counter() - t0

    # Summary
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed  ({failed} failed)  [{elapsed:.1f}s]")
    print("=" * 60)

    if failed:
        print("\nFailed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  ✗ {name}" + (f": {detail}" if detail else ""))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
