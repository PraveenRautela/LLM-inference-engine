# High Performance LLM Inference Engine

This project implements an optimized inference pipeline for GPT-style transformer models.  
The goal is to explore system-level optimizations used in modern Large Language Model (LLM) serving systems.

Instead of using the default HuggingFace `generate()` API, this project builds a custom decoding loop and integrates several inference optimizations such as KV-cache reuse, batched decoding, and quantized inference.

---

## Features

* KV Cache reuse for efficient autoregressive decoding
* Batched decoding to improve GPU utilization
* Temperature and Top-p token sampling
* 8-bit quantization using BitsAndBytes
* Benchmark framework measuring:
  - latency
  - throughput
  - GPU memory utilization

---

Key optimizations:

### KV Cache
Stores previously computed key/value tensors so the model does not recompute attention for previous tokens.

### Batched Decoding
Processes multiple prompts simultaneously to improve GPU utilization.

### Quantization
Loads the model in 8-bit precision to reduce memory usage.

### Token Sampling
Uses temperature and nucleus sampling to generate more natural text.
