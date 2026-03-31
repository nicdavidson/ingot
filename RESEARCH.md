# Ingot — Research Notes, Pitfalls & Gotchas

**Date:** 2026-03-31


This document captures design principles, known pitfalls, and technical findings so we build this right the first time.

---

## 1. Tokenizer Format Detection (Critical)

### The Problem
BPET (byte-level BPE tokenizer) files start with a 4-byte magic header: `BPET` (0x42504554). If a tokenizer loader blindly reads the first 4 bytes as `uint32_t num_entries`, it interprets `BPET` as 1,413,828,674 entries. A subsequent `calloc(1.4B, sizeof(char*))` requests 11.3GB — macOS Jetsam kills the process with SIGKILL (exit code 137).

### Why It's Dangerous
- SIGKILL has no stack trace, no error message — just "zsh: killed"
- Memory reporting (`resident_size`) shows low numbers because the allocation is virtual, not yet touched
- The failure looks like a GPU or mmap issue, not a tokenizer issue

### Design Rule for Ingot
**ALWAYS validate file magic bytes before reading format-specific fields.** The tokenizer module must:
1. Read first 4 bytes
2. Check for known magic (`BPET`, sentencepiece magic, etc.)
3. Only then parse format-specific headers
4. Assert that vocab size is sane (< 500K for any current model)

```c
// GOOD: validate before trusting
uint32_t magic;
fread(&magic, 4, 1, f);
if (magic == 0x42504554) {  // "BPET"
    return load_bpet_tokenizer(f);
} else if (magic < 500000) {
    // Probably raw format where first field is num_entries
    return load_raw_vocab(f, magic);
} else {
    LOG_ERROR("Unknown tokenizer format: magic=0x%08x", magic);
    return NULL;
}
```

---

## 2. macOS Memory Metrics (Know What You're Measuring)

### Two Different Numbers
- **`resident_size`** (from `MACH_TASK_BASIC_INFO`): How much physical RAM your process is using. Does NOT include mmap'd pages that haven't been touched.
- **`phys_footprint`** (from `TASK_VM_INFO`): What Jetsam actually uses to decide whether to kill you. Includes dirty pages, compressed pages, and purgeable-but-not-purged memory.

### MAP_PRIVATE vs MAP_SHARED
- `MAP_PRIVATE` pages count toward Jetsam footprint (they're copy-on-write, kernel charges them to you)
- `MAP_SHARED` pages come from the page cache and are NOT charged to your process
- **For expert weights, always use MAP_SHARED** — they're read-only and the OS manages them

### Design Rules for Ingot
- Use `MAP_SHARED` for all mmap'd weight files (shared weights AND experts)
- Only use `mlock()` for the shared weights (~5GB) that must stay resident
- Monitor `phys_footprint`, not `resident_size`, for OOM risk
- Log both metrics at key points during startup for debugging

---

## 3. The "Trust the OS" Principle

### Why No Custom Expert Cache
The intuitive approach is to build a userspace LRU cache for expert weights — track which experts are hot, evict cold ones, manage memory yourself. **Don't do this.**

macOS page cache already tracks access frequency and handles eviction. Building a custom cache:
- Duplicates work the OS is already doing
- Wastes CPU cycles on cache management bookkeeping
- Fights with the OS's own eviction decisions (two competing LRU policies)
- Adds complexity and bugs for zero benefit

Published experiments show ~38% speedup from removing custom caching and letting the OS handle everything.

### What We Do Instead
1. `mmap()` expert files with `MAP_SHARED`
2. `madvise(MADV_RANDOM)` on all expert files initially (tell OS not to prefetch sequentially)
3. After gate computation: `madvise(MADV_WILLNEED)` on selected experts (prefetch hint)
4. Optionally: `F_NOCACHE` on cold expert file descriptors to prevent them from polluting page cache
5. **That's it.** No LRU. No custom eviction. No userspace cache.

### The Numbers
- M5 Max SSD: ~7 GB/s sequential read
- Expert weight per layer per expert: ~7MB (4-bit quantized, 397B model)
- K=11 active experts per token: ~77MB of expert data needed
- Page fault latency: <1ms for cached, ~10ms for cold
- The OS keeps ~35GB of hot experts in page cache naturally

---

## 4. Metal / GPU Findings

### Metal 3 vs Metal 4 Compatibility
Metal 4 (M5 Max) is backward compatible with Metal 3 shaders. The key APIs are stable across generations:
- `MTLDevice`, `MTLCommandQueue`, `MTLComputePipelineState`
- `newLibraryWithSource:` (runtime shader compilation)
- `newBufferWithBytesNoCopy:` (wrap mmap'd memory as Metal buffer)

No shader modifications needed between M3 Max and M5 Max.

### GPU Memory Allocation
Metal init on M5 Max for a 397B model allocates ~335MB:
- 15 KV caches × 16.8MB = 252MB
- Delta-net state: 195MB
- Scores buffer: 134MB
- Scratch buffers: ~5MB

This is all in unified memory — GPU and CPU share the same physical RAM. No PCIe copies.

### Shader Compilation
- Runtime compilation from source via `newLibraryWithSource:` takes ~1ms on M5 Max
- No need for offline metallib compilation
- Embed shader source as C strings in the binary (build step converts .metal files)

### newBufferWithBytesNoCopy
- Wraps an existing memory region (e.g., mmap'd file) as a Metal buffer
- Zero-copy — GPU reads directly from the mmap'd region
- Used for the shared weights file (~5.5GB)
- Confirmed working on M5 Max

---

## 5. Qwen 3.5 Model Architecture Details

### Hybrid Attention (3:1 Ratio)
Pattern: `N × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`
- 35B: N=10 (30 DeltaNet + 10 SWA layers)
- 122B: N=12 (36 DeltaNet + 12 SWA layers)
- 397B: N=15 (45 DeltaNet + 15 SWA layers)

### Gated DeltaNet (Linear Attention)
- 75% of layers use this instead of traditional attention
- Maintains recurrent state matrix instead of KV cache
- Memory advantage: no KV entries needed for these layers
- **HIGH RISK to implement** — novel architecture, limited reference code
- Reference: HuggingFace transformers PyTorch implementation

### Expert Architecture
- Shared expert: always active, weights always in RAM
- Routed experts: selected by gate, weights streamed from SSD
- Gate: linear projection → softmax → top-K selection
- Each expert: gate_proj + up_proj → SiLU → down_proj

### Tokenizer
- BPET format (byte-level BPE)
- ~248K vocab, ~247K merges, 33 added tokens
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`
- Byte-level encoding: Ġ = space (0x20), Ċ = newline (0x0A), etc.

---

## 6. Tool Calling Format (Qwen 3.5 Specific)

### What Ingot Must Handle

The server MUST properly parse the full `messages[]` array from API requests, including system messages where tool definitions live. It must also apply the correct Qwen chat template with tool injection.

**Tested and confirmed:** Qwen 3.5 397B at 4-bit quantization produces perfect structured tool calls when properly prompted.

### Qwen 3.5 Tool Format

Tools defined in system message:
```
<tools>
{"type": "function", "function": {"name": "read_file", ...}}
</tools>
```

Model outputs:
```xml
<tool_call>
<function=read_file>
<parameter=path>
/etc/hosts
</parameter>
</function>
</tool_call>
```

Tool results sent as user message:
```
<tool_response>
{"content": "127.0.0.1 localhost\n..."}
</tool_response>
```

### Important Details
- NOT special tokens — all plain text XML tags
- Model can include reasoning BEFORE `<tool_call>` but NOT after
- Multiple tool calls = multiple `<tool_call>` blocks
- vLLM uses `--tool-call-parser qwen3_coder` (not `hermes`)
- Known bug in official template's `tool_call.arguments | items` Jinja loop — breaks in some Jinja implementations. We have a patched template at `/data/models/chat-templates/qwen3.5-patched.jinja`

---

## 7. Server Design Requirements

Common pitfalls in MoE inference servers that Ingot must avoid:

| Pitfall | Impact | Ingot Requirement |
|---------|--------|-------------------|
| Ignoring system messages | Model never sees tool definitions | Parse ALL messages in the array |
| Raw BPE tokens in SSE output | Ġ, Ċ garbage in responses | Byte-level decode before sending |
| Think tags in content | Raw reasoning leaks to user | Filter `<think>` blocks from content |
| Hardcoded system prompt | Can't customize model behavior | Use system message from request |
| No multi-message support | Conversations broken | Full conversation history in prompt |
| Ignoring stream:false | Always returns SSE | Respect the stream parameter |
| Missing CORS headers | Browser clients fail | Include Access-Control-Allow-Origin |

---

## 8. Development Environment

### Build Machine (Linux)
- **Role:** Code editing, git, CI
- **Note:** No Metal support — pure C subset only

### Test Machine: MacBook Pro with Apple Silicon (macOS)
- **Role:** Compilation (requires Metal framework) and runtime testing
- **Build:** `make` (clang with `-framework Metal -framework Foundation`)

### Workflow
1. Edit on Linux build machine
2. Push to GitHub
3. Pull on Mac — `git pull`
4. Compile on Mac — `make` (must be on macOS for Metal headers)
5. Test on Mac — `./ingot serve --model /path --port 8090`

### Cross-Compilation Note
The code CANNOT be compiled on Linux. Metal framework, Foundation framework, and Objective-C ARC are macOS-only. Code is written on Linux but always compiled on the Mac. Syntax checking on Linux is limited to `clang -fsyntax-only` for the pure C files.

### Model Files
- **Pre-converted (397B):** `model_weights.bin` (5.5GB shared) + `packed_experts/` (60 layer files, 203GB total) + `vocab.bin` (BPET tokenizer)
- **HuggingFace safetensors:** Raw shards, need conversion via `tools/convert_weights.py`

---

## 9. Performance Reference Points

### M5 Max Benchmarks (397B model, SSD expert streaming)
- TTFT: ~228ms (1 token prompt)
- Generation: ~13 tok/s (short prompts)
- Sustained: 7-10 tok/s (long generation, 800+ tokens)
- Metal init: ~46ms
- Shader compile: ~1ms
- Memory at steady state: ~383MB footprint (rest is page cache managed by OS)

### Token Speed Variance
Speed varies by token position due to expert access patterns:
- Tokens accessing cached experts: 10-13 tok/s
- Tokens causing page faults: 6-7 tok/s
- Average across long generation: ~8 tok/s

### Memory at Key Points (397B)
| Stage | RSS | Phys Footprint |
|-------|-----|---------------|
| Before Metal | 6 MB | 2 MB |
| After Metal | 344 MB | 335 MB |
| After weights mmap | 346 MB | 337 MB |
| After Metal buffer wrap | 346 MB | 337 MB |
| After vocab load | 392 MB | 383 MB |
| After layer file open | 392 MB | 383 MB |

Note: RSS barely changes because mmap'd files are demand-paged. The real memory growth happens during inference as experts are faulted in, but that shows up in page cache stats, not process RSS.

---

## 10. Key References

### Apple Research
- **Efficient LLM Inference on Apple Silicon:** https://machinelearning.apple.com/research/efficient-large-language
- Key finding: SSD-based weight streaming viable on Apple Silicon due to unified memory and fast NVMe

### Qwen 3.5
- **Model cards:** https://huggingface.co/Qwen/Qwen3.5-397B-A17B, Qwen3.5-122B-A10B, Qwen3.5-35B-A3B
- **Architecture paper:** (linked from model cards)
- **Tool calling docs:** https://qwen.readthedocs.io/en/latest/framework/function_call.html
- **Chat template deep dive:** https://huggingface.co/blog/qwen-3-chat-template-deep-dive

### Related Work
- **llama.cpp:** Dense model inference, GGUF format. Good reference for tokenizer and sampler implementations.
- **MLX:** Apple's ML framework. Good reference for Metal shader patterns on Apple Silicon.

---

## 11. Open Questions

1. **DeltaNet implementation:** How closely can we port from HuggingFace transformers? The Metal kernel needs to match the PyTorch reference exactly or output diverges.

2. **Weight format:** Design our own optimized format for mmap, or adopt an existing one? Our format should optimize for per-expert mmap granularity.

3. **Context window:** KV cache for 262K tokens at the 397B model size would be enormous. What's a practical max context for v1?

4. **Multi-model hot-swap:** Can we support loading a different model without restarting the server? Or is that a v2 feature?

5. **2-bit expert quantization:** Further compression to reduce SSD I/O. Worth investigating for the 397B model.
