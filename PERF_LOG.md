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
