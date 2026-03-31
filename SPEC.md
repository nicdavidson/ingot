# Ingot — MoE Inference Engine for Apple Silicon

**Version:** 0.1.0 (Spec Draft)
**Author:** Captain Nic Davidson + XO Milo
**Date:** 2026-03-31
**License:** MIT

---

## What Is This

Ingot is a purpose-built inference engine for running massive Mixture-of-Experts (MoE) language models on Apple Silicon laptops. It streams hundreds of gigabytes of expert weights from SSD, using macOS page cache as a transparent memory extension, and exposes an OpenAI-compatible API with proper streaming, tool calling, and chat template support.

**The pitch:** Run a 397 billion parameter model on a MacBook. No cloud. No API key. No data leaving your machine. 13+ tok/s.

---

## Why This Exists

Existing inference engines (llama.cpp, Ollama, MLX) top out at ~70B parameters on 48GB RAM. They load the entire model into memory. MoE models like Qwen 3.5-397B have 397B total parameters but only activate 17B per token — the rest are dormant expert weights.

Ingot exploits this sparsity. Expert weights are mmap'd from SSD and managed by the OS page cache. Only the active experts (~4GB per token) need to be in RAM at any given moment. The M5 Max SSD delivers ~7GB/s, making expert streaming viable for interactive generation.

This approach builds on [Apple's research into efficient LLM inference on Apple Silicon](https://machinelearning.apple.com/research/efficient-large-language), which demonstrated that SSD-based weight streaming is viable due to unified memory architecture and fast NVMe storage.

---

## Target Models

All Qwen 3.5 MoE models share the same hybrid attention architecture (Gated DeltaNet + Gated Sliding Window Attention in 3:1 ratio):

| Model | Total Params | Active Params | Experts | Active Experts | Layers | Hidden Dim |
|-------|-------------|--------------|---------|---------------|--------|-----------|
| Qwen3.5-35B-A3B | 35B | 3B | 256 | 9 (8 routed + 1 shared) | 40 | 2,048 |
| Qwen3.5-122B-A10B | 122B | 10B | 256 | 9 (8 routed + 1 shared) | 48 | 3,072 |
| Qwen3.5-397B-A17B | 397B | 17B | 512 | 11 (10 routed + 1 shared) | 60 | 4,096 |

All use 262K native context, Multi-Token Prediction training, and 2 KV heads (aggressive GQA).

**Stretch:** Mixtral, DeepSeek V3, and other MoE architectures via pluggable model configs.

---

## Target Hardware

- **Primary:** MacBook Pro M5 Max (48GB unified memory, 40-core GPU, ~7GB/s SSD)
- **Compatible:** Any Apple Silicon Mac (M1 Pro+ with 32GB+ recommended)
- **GPU API:** Metal (Metal 3 and Metal 4)
- **OS:** macOS only (Apple Silicon required)

---

## Core Architecture

### The "Trust the OS" Principle

The central design decision. We do NOT build a custom expert cache. Instead:

1. Expert weight files are `mmap()`'d with `MAP_SHARED`
2. macOS page cache manages which experts are in RAM
3. `madvise(MADV_WILLNEED)` hints prefetch selected experts before they're needed
4. `F_NOCACHE` on cold expert file descriptors prevents evicting hot experts
5. The OS handles LRU eviction better than any userspace cache we'd write

Research shows that removing custom caching layers in favor of OS page cache management yields significant speedups (~38%) because the userspace cache duplicates work the OS is already doing, wastes CPU cycles on cache management, and fights the OS's own eviction decisions.

### Memory Layout

```
┌─────────────────────────────────────────────┐
│ 48GB Unified Memory (M5 Max)                │
│                                             │
│ ┌─────────────────────┐  Always resident    │
│ │ Shared weights ~5GB │  (mmap + mlock)     │
│ │ KV cache ~2GB       │                     │
│ │ Metal buffers ~350MB│                     │
│ │ App overhead ~200MB │                     │
│ └─────────────────────┘                     │
│                                             │
│ ┌─────────────────────┐  OS page cache      │
│ │ Hot experts ~35GB   │  (mmap, on-demand)  │
│ │ (rotating window)   │                     │
│ └─────────────────────┘                     │
│                                             │
│ ══════════════════════════  SSD boundary     │
│                                             │
│ ┌─────────────────────┐  On NVMe SSD        │
│ │ All experts ~200GB  │  (paged in/out by   │
│ │ (cold storage)      │   macOS VM system)  │
│ └─────────────────────┘                     │
└─────────────────────────────────────────────┘
```

### File Structure

```
ingot/
├── README.md
├── LICENSE
├── Makefile
├── SPEC.md                        # This file
├── RESEARCH.md                    # Pitfalls, gotchas, lessons learned
│
├── docs/
│   ├── architecture.md            # Deep dive on design
│   ├── model-format.md            # Weight file layout
│   └── api.md                     # OpenAI API reference
│
├── src/
│   ├── main.m                     # CLI entry point (~150 lines)
│   │
│   ├── config/
│   │   ├── config.h               # Model config structs
│   │   └── config.c               # JSON config parser
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.h            # Public API: encode(), decode()
│   │   ├── tokenizer.c            # BPE merge logic
│   │   ├── bpet.c                 # BPET format loader (magic byte validation!)
│   │   └── byte_decode.c          # BPE byte tokens → UTF-8
│   │
│   ├── model/
│   │   ├── model.h                # Model struct, load/unload lifecycle
│   │   ├── model.c                # Weight loading, mmap orchestration
│   │   ├── mmap_pool.h            # Expert file mmap management
│   │   ├── mmap_pool.c            # fd tracking, madvise hints
│   │   ├── tensor.h               # Tensor struct (shape, data, quant type)
│   │   └── tensor.c               # Tensor utilities
│   │
│   ├── compute/
│   │   ├── metal_context.h        # Metal device/queue/library
│   │   ├── metal_context.m        # Device init, shader compilation
│   │   ├── kernels.h              # Kernel dispatch API
│   │   ├── kernels.m              # Dispatch implementations
│   │   └── shaders/
│   │       ├── matmul.metal       # Matrix multiply (FP16, Q4_K)
│   │       ├── rmsnorm.metal      # RMS normalization
│   │       ├── rope.metal         # Rotary position embeddings
│   │       ├── softmax.metal      # Numerically stable softmax
│   │       ├── elementwise.metal  # SiLU, GELU, add, multiply
│   │       ├── attention.metal    # Fused attention
│   │       ├── moe_gate.metal     # Top-K expert routing
│   │       └── deltanet.metal     # Gated DeltaNet linear attention
│   │
│   ├── inference/
│   │   ├── inference.h            # Public API: generate(), prefill()
│   │   ├── inference.c            # Main inference loop
│   │   ├── attention.c            # Attention dispatch (DeltaNet vs SWA)
│   │   ├── moe.c                  # MoE forward: gate → route → expert → combine
│   │   ├── kv_cache.h             # KV cache management
│   │   ├── kv_cache.c             # Allocate, grow, reuse
│   │   └── sampler.c              # Temperature, top-p, top-k, rep penalty
│   │
│   ├── chat/
│   │   ├── template.h             # Chat template API
│   │   ├── template.c             # Qwen ChatML formatting
│   │   └── tool_parser.c          # Parse <tool_call> blocks from output
│   │
│   ├── server/
│   │   ├── server.h               # HTTP server lifecycle
│   │   ├── server.c               # kqueue-based HTTP/1.1 server
│   │   ├── routes.c               # /v1/chat/completions, /v1/models, /health
│   │   ├── sse.c                  # SSE streaming with UTF-8 decode
│   │   ├── json_write.c           # Streaming JSON builder
│   │   └── request_parse.c        # HTTP + JSON body parsing
│   │
│   └── util/
│       ├── log.h                  # LOG_INFO, LOG_WARN, LOG_ERROR
│       ├── log.c                  # Timestamped structured logging
│       ├── arena.h                # Bump allocator
│       ├── arena.c                # Per-request scratch memory
│       ├── timer.h                # mach_absolute_time wrapper
│       └── json_parse.c           # Minimal JSON parser
│
├── tools/
│   ├── convert_weights.py         # HF safetensors → ingot format
│   └── validate_model.py          # Weight integrity check
│
└── tests/
    ├── test_tokenizer.c           # Encode/decode round-trip
    ├── test_config.c              # Config parsing
    ├── test_template.c            # Chat template formatting
    ├── test_moe_gate.c            # Expert routing correctness
    └── test_server.c              # HTTP/SSE parsing
```

---

## Implementation Phases

### Phase 1: Foundation (Config, Tokenizer, Chat Template)

**Goal:** Parse a model config, tokenize a prompt, decode tokens back to text.

- 1.1: Project skeleton + Makefile (clang, Metal framework, debug/release)
- 1.2: Utilities (logger, arena allocator, JSON parser, timer)
- 1.3: Config parser (read HF config.json → ModelConfig struct)
- 1.4: Tokenizer with BPET format detection and byte-level decoding
- 1.5: Chat template (Qwen ChatML with tool definition injection)

**Exit criteria:** `./ingot --tokenize "Hello world"` prints token IDs and decodes back to clean UTF-8.

### Phase 2: Metal Compute Layer

**Goal:** Metal device up, shaders compiled, basic tensor ops working on GPU.

- 2.1: Metal context (device init, command queue, runtime shader compile)
- 2.2: Core shaders (matmul, rmsnorm, rope, softmax, elementwise, moe_gate)
- 2.3: Kernel dispatch wrappers (C-callable, handle buffer management)
- 2.4: DeltaNet attention kernel (75% of Qwen 3.5 layers use this)
- 2.5: Tensor infrastructure (shapes, strides, quant types)

**Exit criteria:** matmul kernel matches CPU reference within FP16 tolerance.

### Phase 3: Model Loading + MoE Inference

**Goal:** Load weights, run full forward pass, get coherent output.

- 3.1: Weight converter (Python: safetensors → ingot format)
- 3.2: Model loader (mmap shared weights + expert files, build weight index)
- 3.3: Dense forward pass (embedding → attention → FFN → logits)
- 3.4: MoE forward pass (gate → route → expert dispatch → combine)
- 3.5: KV cache (sliding window for SWA layers, recurrent state for DeltaNet)
- 3.6: Sampler (temperature, top-p, top-k)

**Exit criteria:** `./ingot --model /path/to/35B --prompt "Hello"` generates coherent text.

### Phase 4: HTTP Server + API

**Goal:** Drop-in replacement for any OpenAI-compatible client.

- 4.1: kqueue HTTP server (no external deps, ~500 lines)
- 4.2: Streaming JSON writer
- 4.3: API routes (/v1/chat/completions, /v1/models, /health)
- 4.4: SSE streaming with proper BPE byte decoding
- 4.5: Think tag stripping (filter `<think>` blocks from content)
- 4.6: Tool call parsing (detect `<tool_call>` XML, return as OpenAI tool_calls)

**Exit criteria:** Works with OpenCode, Open WebUI, and raw curl. Clean UTF-8, no BPE artifacts.

### Phase 5: Optimization

- 5.1: Tiered I/O (F_NOCACHE on cold experts, normal on hot)
- 5.2: Prefill optimization (batch prompt processing)
- 5.3: Expert prefetch (madvise WILLNEED after gate computation)
- 5.4: /metrics endpoint
- 5.5: Prompt caching (prefix KV reuse)

---

## API Specification

### POST /v1/chat/completions

Standard OpenAI chat completions. Supports:

- `messages[]` — full conversation history (system, user, assistant, tool roles)
- `tools[]` — function definitions (injected into prompt via Qwen template)
- `stream` — true/false for SSE streaming
- `max_tokens` — generation limit
- `temperature`, `top_p`, `top_k` — sampling parameters

**Tool calling format (Qwen 3.5 native):**

Model output:
```xml
<tool_call>
<function=read_file>
<parameter=path>
/etc/hosts
</parameter>
</function>
</tool_call>
```

API response:
```json
{
  "choices": [{
    "message": {
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "read_file",
          "arguments": "{\"path\": \"/etc/hosts\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Tool results sent back as:
```json
{"role": "tool", "tool_call_id": "call_1", "content": "127.0.0.1 localhost\n..."}
```

### GET /v1/models

Returns loaded model info.

### GET /health

Returns server status, tok/s, memory usage, expert cache stats.

---

## CLI Interface

```bash
# Serve mode (primary use case)
ingot serve --model /path/to/model --port 8090

# Single generation
ingot generate --model /path/to/model --prompt "Hello" --tokens 200

# Interactive chat
ingot chat --model /path/to/model

# Convert weights from HuggingFace format
ingot convert --input /path/to/hf/model --output /path/to/ingot/model

# Tokenize (debug)
ingot tokenize --model /path/to/model --text "Hello world"
```

---

## Performance Targets

| Model | Hardware | Target tok/s | Notes |
|-------|----------|-------------|-------|
| Qwen3.5-35B-A3B | M5 Max 48GB | 40+ | Small experts, should fly |
| Qwen3.5-122B-A10B | M5 Max 48GB | 20-30 | Sweet spot model |
| Qwen3.5-397B-A17B | M5 Max 48GB | 10-15 | Expert streaming limited by SSD bandwidth |

---

## Non-Goals (v1)

- Linux/Windows support (Metal is macOS only)
- Training or fine-tuning
- Multi-GPU (Apple Silicon is single-GPU)
- Dense model support (use llama.cpp for that)
- Image/video/audio modality (text-only for v1)

---

## Success Criteria

- [ ] All three Qwen 3.5 MoE models load and generate coherent text
- [ ] 397B runs on 48GB M5 Max without OOM via expert streaming
- [ ] OpenAI API works with OpenCode, Open WebUI, and standard clients
- [ ] Tool calling works end-to-end (model outputs tools, API formats them)
- [ ] SSE streaming produces clean UTF-8 (no raw BPE bytes)
- [ ] Think tags stripped from output
- [ ] No file over 1,000 lines
- [ ] Competitive performance on M5 Max (10-15 tok/s on 397B)
- [ ] Clean build with zero warnings on `clang -Wall -Wextra`
