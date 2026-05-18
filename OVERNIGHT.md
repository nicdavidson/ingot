# Ingot Overnight Optimization Sprint

**Goal:** Get 35B to 13+ tok/s, fix 397B regression, approach TheFlash MoE baselines.
**Method:** Iterative — fix one thing, build, test, measure, move to next.
**Build:** `make clean && make` on this Mac.
**Test 35B:** `./ingot generate --model /Users/nic/models/Qwen3.5-35B-A3B-4bit/ --prompt "Hello" --max-tokens 20`
**Test 397B:** `./ingot generate --model /Users/nic/models/Qwen3.5-397B-A17B-4bit/ --prompt "Hello" --max-tokens 10`

---

## Current Baseline (measure first!)

Before changing ANYTHING, run both model tests and record tok/s. This is your starting point.
Write results to `PERF_LOG.md` with timestamps.

Expected starting point:
- 35B: ~4.1 tok/s, prefill ~1.5s/5tok, per-layer 6.3ms
- 397B: ~0.04 tok/s (BROKEN — regression from buffer wrapping)

---

## Phase 1: Fix the 397B Regression (CRITICAL — do this first)

**Problem:** 397B is slower than before (0.04 vs 0.1 tok/s). Metal buffer wrap is failing for large attention buffers, causing double-processing (GPU attempt fails silently, falls through to CPU).

**Root cause location:** `src/inference/attention.c`, function `q4_proj()` (lines 19-63)

**The bug:** `metal_wrap_buffer()` with `newBufferWithBytesNoCopy` has alignment requirements. For 397B model dimensions (hidden=4096, q_proj output=4096*128*2=1M floats), the buffer sizes may exceed Metal's no-copy limit or hit page alignment issues. When wrap fails, it returns NULL, the GPU path is skipped, BUT the CPU fallback still runs — so every matmul is attempted twice (GPU fail + CPU succeed).

**Fix strategy:**
1. Add logging to `q4_proj()` to confirm: is `metal_wrap_buffer` returning NULL for 397B?
2. Check `metal_wrap_buffer` in `src/compute/metal_context.m` — what's the size limit? Is alignment enforced?
3. **Solution A (preferred):** Pre-allocate reusable Metal buffers for attention Q/K/V/O projections in `InferenceContext`, similar to how expert buffers are already pre-allocated (`gpu_gate_out`, `gpu_up_out`, etc.). The attention code should use these cached buffers instead of wrapping/freeing per call.
4. **Solution B (fallback):** If wrap is failing due to alignment, ensure `calloc` in attention gives page-aligned memory (use `posix_memalign` or `aligned_alloc` with 16384-byte alignment).

**Validation:** 397B should produce coherent text at >= 0.1 tok/s (restoring baseline). If it's producing garbage, there's a correctness bug too — check output text.

---

## Phase 2: Cache Metal Buffers in q4_proj (Biggest perf win for both models)

**Problem:** Every call to `q4_proj()` in `attention.c` does:
```
metal_wrap_buffer() → kernel dispatch → metal_free_buffer()
metal_wrap_buffer() → kernel dispatch → metal_free_buffer()
```
This is called 4-6 times per layer (q_proj, k_proj, v_proj, o_proj + optional). The wrap/free overhead per call dominates at small matrix sizes.

**Fix:** Add persistent GPU buffers to attention, similar to how `inference.c` already does it for the MoE path.

In `attention.h` or a new struct, add:
```c
typedef struct {
    void *gpu_q;      // Metal buffer for Q projection output
    void *gpu_k;      // Metal buffer for K projection output
    void *gpu_v;      // Metal buffer for V projection output
    void *gpu_attn;   // Metal buffer for attention output
    void *gpu_input;  // Metal buffer for layer input (hidden state)
    float *cpu_q, *cpu_k, *cpu_v, *cpu_attn, *cpu_input;  // CPU pointers
    bool valid;
} AttentionGPUBuffers;
```

Allocate these once in `inference_create()`, sized for the max dimensions:
- Q buffer: `num_heads * head_dim * 2` (×2 for output gate)
- K/V buffer: `num_kv_heads * head_dim`
- Input: `hidden_size`

Then modify `q4_proj()` to accept optional pre-allocated Metal buffers instead of wrapping.

**Better approach:** Make `q4_proj` take `(void *gpu_in, void *gpu_out)` optional params. When non-NULL, skip the wrap/free. When NULL, do the current wrap path (for code outside the hot loop).

**Validation:** 35B per-layer time should drop significantly. Measure before/after.

---

## Phase 3: Wire Up fused_gate_up_swiglu Kernel

**Problem:** The `fused_gate_up_swiglu` Metal shader is compiled and the dispatch function exists in `kernels.m` (line 253), but the expert hot path in `inference.c` still does separate gate_proj + up_proj + cpu_silu_mul.

**Current code (inference.c ~line 459-481):**
```
batch gate_proj → GPU
batch up_proj → GPU
end_batch (commit + wait)
cpu_silu_mul → CPU
down_proj → GPU
```

**Optimized code:**
```
fused_gate_up_swiglu → GPU (one dispatch, reads input once)
down_proj → GPU
```

**Implementation:**
1. In the GPU expert path (inference.c, inside `#ifdef PLATFORM_MACOS` block around line 454):
2. Replace the batched gate+up with a single `kernel_fused_gate_up_swiglu` call
3. But there's a catch: `kernel_fused_gate_up_swiglu()` takes separate buffer pointers, not offsets into a shared buffer
4. **You need an offset version** — create `kernel_fused_gate_up_swiglu_offsets()` that takes the expert_buf + offsets, similar to `kernel_matmul_q4_fma_offsets()`
5. The shader itself (`fused_gate_up_swiglu.metal`) is ready — just need the dispatch wrapper

**New function signature:**
```objc
void kernel_fused_gate_up_swiglu_offsets(
    MetalContext *ctx,
    void *expert_buf,
    size_t gate_w_off, size_t gate_s_off, size_t gate_b_off,
    size_t up_w_off, size_t up_s_off, size_t up_b_off,
    void *x, void *out,
    uint32_t moe_dim, uint32_t K, uint32_t group_size);
```

**Validation:** Expert processing per token should get faster. This saves one full kernel dispatch + eliminates the CPU silu_mul hop.

---

## Phase 4: Deferred Command Buffers (Layer Pipelining)

**Problem:** Every Metal kernel call does `commit + waitUntilCompleted`. This serializes GPU and CPU work. While the GPU computes layer N's matmuls, the CPU sits idle waiting.

**Fix:** For the main inference loop, encode ALL GPU work for a layer into a single command buffer, commit it, and start preparing the NEXT layer's data while GPU finishes.

This is the most complex optimization. Only attempt if Phases 1-3 are done and tested.

**Approach:**
1. Create a "layer command buffer" that batches: rmsnorm + attention projections + MoE matmuls
2. While that executes, CPU can do: arena reset, name formatting, weight offset lookups for next layer
3. Use `addCompletedHandler:` instead of `waitUntilCompleted` where possible
4. Be careful with data dependencies — can't start next layer until current layer's hidden state is written

**Simpler version:** Just batch all the q4_matmul calls within MoE (shared expert + all routed experts) into one command buffer. Currently gate+up are batched but down_proj is separate, and shared expert is fully separate.

---

## Measurement Protocol

After EACH phase, record in PERF_LOG.md:
```
## Phase N Complete — [timestamp]
35B: X.X tok/s (was Y.Y, delta +Z.Z)
397B: X.X tok/s (was Y.Y, delta +Z.Z)
Per-layer 35B: X.Xms
Changes: [brief description]
```

---

## Rules

1. **Build must compile clean** — zero warnings with `-Wall -Wextra -Wpedantic`
2. **Test after every change** — run 35B generate before moving to next phase
3. **Don't break what works** — if a change makes things worse, revert it
4. **397B fix is Phase 1** — don't skip to optimizations while the big model is broken
5. **Commit after each working phase** — `git add -A && git commit -m "phase N: description"`
6. **Read the existing code carefully** — the patterns are consistent, follow them
7. **Metal buffer lifecycle matters** — anything allocated with `metal_alloc_buffer` or `metal_wrap_buffer` must be freed with `metal_free_buffer`. Unified memory pointers from `metal_buffer_contents` are valid only while the buffer lives.
8. **Check `src/compute/metal_context.m` and `src/compute/metal_context.h`** for the Metal buffer API before writing new dispatch code.

---

## Success Criteria

- [ ] 397B produces coherent text (not garbage)
- [ ] 397B >= 0.5 tok/s (recovered + some improvement)
- [ ] 35B >= 8 tok/s
- [ ] All changes compile clean
- [ ] Git history shows incremental progress
- [ ] PERF_LOG.md has measurements for each phase
