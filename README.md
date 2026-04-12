# Mechanistic Interpretability Toolkit

## Overview

The Mechanistic Interpretability Toolkit is a high-performance, web-based platform designed to facilitate the deep inspection of transformer language models. This project provides researchers and engineers with real-time visualizations of attention mechanisms, multi-layer perceptron (MLP) neurons, and residual stream components. By dissecting models such as GPT-2 and Qwen3, the toolkit allows users to isolate causal circuits and visualize internal model behaviors with granular precision.

## Key Features

### Interactive Interpretability Interface
- **Token Heatmaps**: Inline token visualizations utilizing direct logit attribution via residual stream decomposition. Modifying the query position dynamically updates the surrounding visual context.
- **Attention Head Analysis**: An interactive `n_layers x n_heads` grid visualization that maps mean attention entropy. Users can drill down into specific layer-head combinations to view full `seq_len x seq_len` attention pattern matrices.
- **Logit Lens Streaming**: Real-time layer-by-layer decoding of next-token predictions, streamed via WebSockets to illustrate the exact computational layer where a model's prediction stabilizes.
- **Activation Patching**: Granular activation patching across source and target prompts. The interface renders the resulting change in logit difference (`ΔLogit`) per head to rapidly identify minimal causal circuits.

### CUDA-Accelerated Inference (Planned)
Complex analytical operations like activation patching traditionally suffer from memory allocation and transfer bottlenecks. To achieve interactive latency, this toolkit will feature a custom, fused CUDA kernel that replaces standard Python-managed caches.

This high-performance kernel seamlessly:
- Writes raw activations directly into a pre-allocated pinned memory buffer to enable fast overhead asynchronous host transfers.
- Computes per-head L2 norms in-place using warp-level reduction natively on the GPU.
- Applies numerically stable streaming softmax operations over logit projections within a single hardware pass.

This kernel integrates directly via PyTorch C++ extensions into the standard TransformerLens hooks, cutting forward-pass latency by up to 50%.

## Technology Stack

- **Backend**: Python, FastAPI, WebSockets
- **Hardware Acceleration**: CUDA/C++, PyTorch C++ Extensions
- **Modeling Tooling**: TransformerLens
- **Frontend**: React, TypeScript, D3.js

## Project Architecture

The repository is logically separated into the API backend, specialized GPU kernels, and the interactive frontend:

- `backend/`: Contains the FastAPI application, TransformerLens analysis pipelines, and the PyTorch/CUDA extensions.
- `frontend/`: The React web interface consisting of D3.js-powered visualizers for the various interpretability techniques.
- `scripts/`: Optimization and profiling scripts to validate the latency and occupancy of the compiled CUDA implementations.

## Deployment

The application is containerized for streamlined deployment. Running the default compose configuration starts the backend API and serves the frontend interface locally. Model weights are automatically cached inside a Docker volume to eliminate redundant network requests.

```bash
docker-compose up --build
```
