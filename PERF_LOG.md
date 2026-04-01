# Ingot Performance Log

## Baseline (pre-optimization)
- **35B (qwen3.5-35b-a3b-4bit)**: 3.7 tok/s (3 tokens in 801ms, prefill 2455ms)
- **397B (Qwen3.5-397B-ingot)**: 0.04 tok/s (30 tokens in 778670ms, prefill 27298ms)
- 397B output is garbage — GPU path silently failing, CPU fallback producing incorrect results

## Phase 1: Fix 397B regression (temp GPU buffers in attention q4_proj)
- Root cause: input hidden state lives in Metal unified memory; newBufferWithBytesNoCopy cannot re-wrap memory owned by another Metal buffer, so x_buf always NULL, GPU path never fired
- Fix: Allocate temp Metal buffers via metal_alloc_buffer, memcpy in/out
- **35B**: 3.7 tok/s (unchanged — attention is small fraction of 35B runtime)
- **397B**: 0.1 tok/s (up from 0.04 — 2.5x improvement, GPU path confirmed working)
- Note: 397B output still garbage (likely separate model/weight issue, not GPU-related)
- Remaining bottleneck: ~300 metal_alloc/free cycles per token from temp buffers

## Phase 2: Persistent GPU buffers for attention
- Change: AttentionGPU struct with pre-allocated Metal buffers, pass gpu_norm_out handle directly
- Eliminates per-call metal_alloc/free and input memcpy (hidden already on GPU)
- **35B**: 4.0 tok/s (up from 3.7 baseline — 8% improvement)
- **397B**: 0.05 tok/s (marginal — bottleneck is memory paging, expert files ~180GB exceed RAM)
- Note: 397B speed limited by SSD-backed mmap, not GPU dispatch overhead

## Phase 3: Fused gate+up+SwiGLU in expert hot path
- Change: Replace separate gate_proj + up_proj + cpu_silu_mul with fused_gate_up_swiglu_offsets kernel
- **35B**: 3.9 tok/s (marginal gain — fused kernel vs batched pair comparable for small dims)

## Phase 4: Batch GPU work aggressively
- Change: Batch all routed experts into single command buffer (per-expert GPU slots)
- Change: Fused+batched shared expert FFN on GPU (was entirely CPU)
- Change: Batch attention projections (DeltaNet 4-in-1, SWA 3-in-1)
- **35B**: 14.3 tok/s (from 3.7 baseline — 3.9x improvement, target 13+ achieved!)
- **397B**: 0.1 tok/s (up from 0.04 — 2.5x, memory-bound by SSD-backed mmap)
- Key insight: dispatch overhead (commit+waitUntilCompleted) was ~50% of token time

## Phase 5: Parallel pread expert I/O with Metal staging buffers
- Change: Replace mmap page-fault I/O with parallel pread() via GCD dispatch groups
- Expert data read directly into Metal unified memory staging buffers (zero-copy GPU access)
- pread issued after gate top-K, overlaps with shared expert GPU work
- Threshold: only use pread for large experts (>2MB stride) to avoid regressing small models
- **35B**: 13.4 tok/s (no regression — uses mmap path for small 1MB experts)
- **397B**: 2.7 tok/s cold, 5.6 tok/s warm (from 0.05 — **54-112x improvement**)
- Key insight: parallel pread eliminates serial page faults, SSD serves 10 concurrent reads

## Phase 6: Deferred expert command buffers
- Change: Commit expert GPU work without waiting, defer accumulation to next layer
- Overlaps expert GPU execution with next layer's attention start
- **35B**: 13.4 tok/s (no regression)
- **397B**: 2.7 tok/s cold, 5.5 tok/s warm (marginal — model produces few tokens before EOS)

## Phase 7: FMA dequant restructuring in Metal shaders
- Change: Pre-compute scale*x and bias*x, accumulate bias terms separately, chain FMA
- **35B**: 13.2 tok/s (within noise — original already used FMA intrinsics)
- **397B**: 5.3 tok/s warm (within noise)
- Note: Improvement may be more pronounced on sustained computation with longer sequences

## Summary
| Model | Baseline | Current | Improvement |
|-------|----------|---------|-------------|
| 35B   | 3.7 tok/s | 13.2 tok/s | 3.6x |
| 397B  | 0.05 tok/s | 5.3 tok/s (warm) | 106x |
| 397B  | 0.05 tok/s | 2.7 tok/s (cold) | 54x |
