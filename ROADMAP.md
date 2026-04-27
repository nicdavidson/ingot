# Ingot Roadmap — DeepSeek V4-Flash to Production

**Last updated:** 2026-04-27 (afternoon — V4 forward path live)
**Author:** Milo (XO)
**Goal:** Get V4-Flash generating coherent text with tok/s metrics, then wire it into Hermes.

## Status snapshot

V4-Flash now runs end-to-end. Text generates, hidden states are healthy through all 43 layers,
zero NaN/Inf, logits look distributionally reasonable. **Output is still gibberish multilingual**
(e.g. "Gikuha� gihulagway"), so something in the dequant constants or HC scaling is still off.
Speed is ~0.03 tok/s because routed experts run on CPU instead of GPU — see "Remaining work".

## Fixes that landed this session (Phase A → mid-Phase D)

1. **Tokenizer special tokens** — fall back from Qwen names (`<|endoftext|>` etc.) to V4's
   full-width unicode names (`<｜end▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`).
   `src/tokenizer/tokenizer.c`.
2. **JSON unicode escape decoding** — `json_string` now decodes `\uXXXX` (with surrogate pairs)
   to UTF-8. Without this, V4's vocab.json keys never matched. Took merges from 28k→127k.
   `src/util/json_parse.c`.
3. **V4 chat template** — new branch in `template_apply` keyed off `ModelArch`. Uses
   `<｜begin▁of▁sentence｜>` BOS, `<｜User｜>`/`<｜Assistant｜>` turn markers, `<think>`/`</think>`
   for thinking mode. Tool calling for V4 not yet wired. `src/chat/template.{c,h}`,
   plus arch threading through `src/server/routes.c`.
4. **V4 MLA segfault fix** — `kv_idx` was not resetting between tokens because `v4_forward.c`
   on the Mac was older than the local copy. Synced.
5. **F32 vs BF16 hc_fn detection** — `attn_hc.fn`, `ffn_hc.fn`, `hc_head.fn` are F32 (shape
   [24, 16384]) but the code read them as BF16, producing NaN. Now dtype-detected by buffer size.
   `src/inference/v4_forward.c::v4_hc_pre`, `v4_compute_logits`.
6. **`sz` reuse bug** — both `hc_*.scale` and `hc_*.base` lookups wrote to the same `sz`,
   so the scale dtype check used the base's size. Fixed both call sites.
7. **MXFP4 routed experts** — V4 packed experts use FP4 (E2M1) nibbles + E8M0 group exponents
   (group_size=32, no biases), not Qwen's int4 + BF16 affine. Added `dequant_matmul_mxfp4`
   in `src/inference/dequant.{c,h}`; routed expert loops in v4_forward call it instead of
   `dequant_matmul_q4`.

## Remaining work (in priority order)

### P0 (NEW after second session): Diff against reference impl

We've done every "obvious" piece — Compressor, FP8 sim, attn_sink, output
RoPE inverse, K=V cache, per-layer rope theta, MXFP4, HC math fixes — and
output is still incoherent on `"What is the capital of France?"`. With and
without the Compressor produces different gibberish, with and without FP8
sim produces different gibberish — meaning each piece is structurally
contributing but something subtle is still wrong.

The cheapest path forward is **diff against the reference impl tensor-by-tensor**:
- Reference: `/Users/nic/models/DeepSeek-V4-Flash/inference/generate.py` +
  `model.py` (PyTorch + tilelang).
- `tilelang` is now installed on the Mac. `fast_hadamard_transform` is the
  only remaining missing dep — it's used by the Indexer's `rotate_activation`
  and won't compile without CUDA. Workaround: edit `model.py` to replace
  `rotate_activation` with a Python Walsh-Hadamard transform (~10 lines),
  OR run with `--input-file` of a 4-token prompt where the indexer doesn't
  fire (still depends on it being defined at import time? — check).
- Add per-layer hidden-state dumps to a file from both my `v4_forward.c` and
  the reference. Numpy-diff. First diverging layer localizes the bug.

### P1: Fix output quality — still gibberish

Pipeline is numerically clean (no NaN/Inf, healthy norms, sensible logit distributions)
but output is gibberish multilingual ("veteinteure材teue材"). Reading the reference at
`/Users/nic/models/DeepSeek-V4-Flash/inference/model.py` shows the V4 attention is
substantially more elaborate than my MLA implementation. The actual gap:

**Already fixed against reference (this session):**
- HC `pre = sigmoid(...) + eps` — was missing `+eps` (`v4_forward.c::v4_hc_pre`).
- HC `post = 2 * sigmoid(...)` — was just `sigmoid(...)`, missing the 2× (same file).
- HC `comb` Sinkhorn: replaced log-space alternating-LSE with reference's algorithm
  (row-softmax + eps, col-divide, then iterations of row/col division with `+eps`
  in denominators). Earlier impl drifted off the reference.
- HC `comb` index transpose in `v4_hc_post`: reference sums over the FIRST comb axis
  (`y[m] = post[m]*x + sum_a comb[a, m] * residual[a]`), I had `comb[m, a]`.
- Final `hc_head` `pre = sigmoid(...) + hc_eps` — was missing `+eps` in
  `v4_compute_logits`.

**Still missing — the real culprit for incoherent output:**

- [ ] **`attn_sink`** — V4 has a learnable per-head bias added to attention scores.
  Already mmapped from the model (`layers.X.attn.attn_sink`, F32 [n_heads]) but not
  yet read or applied anywhere. `sparse_attn` in the reference uses it.
- [ ] **Output-side RoPE un-rotation** — reference does
  `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` AFTER attention; the `True` flag
  inverses the rotation so the output's last `rope_head_dim` slice ends up back in
  the un-rotated frame for the absorbed V projection. My code never un-rotates.
- [ ] **Compressor (CSA layers, compress_ratio=4, layers 2–22)** — V4 maintains a
  *compressed* KV cache: every `compress_ratio` tokens get folded into one compressed
  KV via a learned `wkv` + `wgate` + `ape` (absolute positional embedding) + RMSNorm.
  21 of the 43 layers use this; my code currently treats them as plain sliding-window.
- [ ] **Indexer (HCA layers, compress_ratio=128, layers 23–42)** — separate
  `Indexer` module (its own small attention) selects top-K=512 compressed tokens to
  attend to. Without this, the 20 HCA layers attend to the wrong context entirely.
- [ ] **`sparse_attn` with top-K** — even sliding-window layers use sparse attention
  with `topk_idxs` (window indices + compressed indices), not dense softmax. My code
  does dense softmax over the entire window.
- [ ] **FP8 simulation of KV non-rope dims** — `act_quant(kv[..., :-rd], 64, ...)`
  is a QAT-style fake-quant pass that the model was trained to expect. Skipping it
  causes a small but real distribution shift.

The CSA Compressor + HCA Indexer are the biggest pieces. Until they're implemented
(or all layers are forced to sliding-window CPU as a debugging mode that probably
just produces different gibberish), V4 output will not be coherent.

A reasonable shortcut: implement Compressor + Indexer + sparse_attn + attn_sink +
output-RoPE-inverse on CPU first (slow but correct), validate output quality against
mlx-lm or the reference repo on a 5-token completion, then port the inner kernels
to Metal.

### P2: GPU MXFP4 shader

Routed-expert MoE on CPU is ~1000x slower than the GPU path. Until we add a shader,
V4 generation will be measured in seconds-per-token, not the other way around.

- [ ] Write a Metal shader `matmul_mxfp4` that reads U32 weight + U8 E8M0 scale, decodes
  via a constant FP4 LUT (or `frexp`-style bit shuffle), multiplies, group_size=32.
- [ ] Add a fused `gate_up_swiglu_mxfp4` to mirror the existing `fused_gate_up_swiglu`.
- [ ] Re-enable the `if (false &&` GPU branches in `v4_forward.c::v4_forward_layer`
  (search for "Fall through to the CPU MXFP4 path") once the shaders exist.
- [ ] Mind the bias-less layout: scale arg is U8 [moe_dim, K/32], no bias offset.

### P3: Phase E benchmarks (still relevant)

- [ ] Once output is coherent, measure TTFT, sustained tok/s, cold-start tok/s, phys footprint.
- [ ] V4 timing infra (`v4_timing_report`) is hooked up — just needs to be enabled per-request.

### P4: Hermes integration (Part 2 of original plan)

Unchanged. Tool-calling format for V4 (`<｜tool▁calls▁begin｜>` etc.) is not yet wired
into `src/chat/template.c` — the V4 branch falls back to plain text for tool messages.

---

## Current State

**What works:**
- Binary compiled (Apr 25) at `./ingot`
- V4-Flash converted: 141GB at `/Users/nic/models/DeepSeek-V4-Flash-ingot/`
- 43 expert layer files, 2266 weight entries, config parses correctly
- Server starts clean: Metal init (M5 Max), all 43 expert files mmap'd as Metal buffers
- Load time: 153ms
- V4 forward path fully coded: `v4_forward.c` (1176 lines), `attention_v4_mla_forward` (~200 lines)
- Hyper-connections, MoE gating, hash routing, CSA/HCA — all implemented
- Qwen 3.5 path (35B/122B/397B) fully operational and benchmarked

**What's broken (the "all E tokens, 0ms timings" symptom):**

Three red flags from the server load log:

1. **`tokenizer: eos=-1, im_start=-1, im_end=-1`**
   V4 uses a different tokenizer (129K vocab). Special token detection was written for Qwen.
   Without EOS → can't stop generating. Without im_start/im_end → chat template is mangled.
   **This alone explains garbage output.**

2. **`metal: compiled 18/20 pipelines`**
   2 Metal shader pipelines failed to compile. If they're used in the V4 path,
   GPU matmuls silently produce zeros.

3. **Weight name mapping — unverified**
   Forward pass constructs names like `layers.0.attn_hc.fn.weight`.
   If the converter wrote them differently, lookups return NULL, zeros propagate silently.

---

## Part 1: Present → Readable V4 Output with tok/s

### Phase A: Fix the Tokenizer (1-2 hours)

The V4 tokenizer uses different special token names than Qwen.
Qwen: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`
V4: likely `<｜end▁of▁sentence｜>` or similar full-width unicode markers.

**Steps:**
- [ ] Inspect `tokenizer_config.json` in DeepSeek-V4-Flash-ingot for actual special token strings
- [ ] Inspect `tokenizer.json` added_tokens for EOS/BOS/pad token IDs
- [ ] Update `src/tokenizer/tokenizer.c` special token detection to handle V4's names
- [ ] Verify EOS, BOS resolve to valid IDs (not -1)
- [ ] Check chat template (`chat_template.jinja`) — V4 uses different template than Qwen
- [ ] Update `src/chat/chat_template.c` if needed for V4's format

**Validation:** After fix, server load log should show actual token IDs, not -1.

**Files:**
- `src/tokenizer/tokenizer.c` — special token detection
- `src/chat/chat_template.c` — template application
- Model dir: `tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`

---

### Phase B: Validate Weight Name Mapping (1-2 hours)

Every `snprintf(name, ...)` in v4_forward.c and attention.c constructs a weight name
and looks it up in weight_index.json. If ANY name is wrong, that weight silently reads as zeros.

**Steps:**
- [ ] Dump all keys from `weight_index.json`:
  ```bash
  python3 -c "import json; d=json.load(open('weight_index.json')); [print(k) for k in sorted(d.keys())]" > /tmp/v4_weights.txt
  ```
- [ ] Grep all `snprintf(.*name` patterns in `v4_forward.c` and `attention.c`
- [ ] Cross-reference: for each constructed name, verify it exists in weight_index.json
- [ ] Common mismatches to check:
  - `attn_hc` vs `attn_hc` (might have `model.` prefix or not)
  - `attn_norm` vs `input_layernorm` (Qwen vs V4 naming)
  - `ffn_norm` vs `post_attention_layernorm`
  - `ffn.gate` vs `mlp.gate`
  - `hc_head.fn` vs `model.final_hc.hc_fn`
- [ ] Fix mismatches in either forward pass OR converter (prefer fixing forward pass)
- [ ] Add startup validation: for layer 0, check all expected V4 weights are found, WARN on misses

**Validation:** Zero "missing weight" warnings at startup.

**Files:**
- `src/inference/v4_forward.c` — weight name construction (lines 261-264, 295-296, 581-582, 617-618, 629-632, 700-708, etc.)
- `src/inference/attention.c` — MLA weight names (~line 1098+)
- `src/config/config.c` — V4 config parsing
- Model dir: `weight_index.json`

---

### Phase C: Fix Metal Pipeline Failures (30 min)

Server reports `compiled 18/20 pipelines`. Need to know which 2 failed.

**Steps:**
- [ ] In `src/compute/metal_context.m`, add logging for which pipeline names fail
- [ ] Rebuild, check which 2 fail
- [ ] If they're used in V4 path → fix the shader or add CPU fallback
- [ ] If they're Qwen-only (e.g., deltanet) → ignore, not relevant to V4

**Validation:** Either 20/20 compile, or the 2 failures are confirmed Qwen-only.

**Files:**
- `src/compute/metal_context.m` — pipeline compilation (~line 146)
- `src/compute/shaders/` — 9 .metal files

---

### Phase D: Layer-by-Layer Norm Validation (2-3 hours)

Same debugging approach that worked for Qwen. Run 1 token, trace hidden state norms.

**Steps:**
- [ ] Add `LOG_INFO` for L2 norm of hidden state after each sublayer:
  - After HC pre (should reduce [4*H] → [H], norm ~1-10)
  - After attention (norm ~1-10)
  - After HC post (4 copies updated, norms ~1-10)
  - After FFN HC pre
  - After MoE (shared + routed)
  - After FFN HC post
- [ ] Run 1 token through all 43 layers
- [ ] Check: norms should be stable ~1-10 range, not 0, NaN, or exploding
- [ ] If norms are 0 at a specific layer → weight lookup failing (Phase B)
- [ ] If norms explode → likely culprits:
  - Sinkhorn normalization producing degenerate matrices (check iteration count = 20)
  - MLA Q norm applied wrong (should be inline rsqrt(mean(q^2) + eps), not RMSNorm)
  - Missing RoPE on last 64 dims of head_dim=512
  - Compressed attention indexer returning garbage positions
- [ ] Validate layer 0 independently — simplest V4 layer (sliding window + hash routing, no CSA/HCA)

**Validation:** Hidden state norms stable through all 43 layers, no NaN/Inf/zero.

**Files:**
- `src/inference/v4_forward.c` — `v4_forward_layer()` (line 553)
- `src/inference/attention.c` — `attention_v4_mla_forward()` (line 1098)

---

### Phase E: First Coherent Output + Benchmarks (1-2 hours)

Once norms are healthy:

- [ ] Generate a few tokens with a simple prompt ("Hello, I am")
- [ ] Verify output is actual words, not repeated characters
- [ ] Test multi-turn conversation
- [ ] Measure and report:
  - TTFT (time to first token)
  - Sustained tok/s (after page cache warm)
  - Cold start tok/s (first prompt)
  - Memory footprint (phys_footprint, not resident_size)
- [ ] V4 timing infrastructure already exists: `v4_timing_report()` prints hc/attn/shared/routed breakdown
- [ ] Add tok/s to the SSE stream metadata (same pattern as Qwen path)

**Expected V4-Flash performance (estimate):**
- 284B params, 13B active — smaller active set than 397B Qwen (17B)
- Expert data ~130GB, will thrash page cache on 48GB machine
- Rough estimate: **5-8 tok/s** on M5 Max 48GB
- Could be faster than 397B Qwen since active params are smaller (13B vs 17B)

**Files:**
- `src/inference/v4_forward.c` — `v4_timing_report()` (line 527)
- `src/server/sse.c` — streaming output

---

## Part 2: Working Ingot V4 → Hermes Integration

This is the easy part. Ingot already speaks OpenAI API. Hermes already consumes it.

### Step 1: Point Hermes at Ingot (5 min)

Current Hermes config (`~/.hermes/config.yaml`):
```yaml
model: qwen3.6-27b-4bit
base_url: http://macbook-pro.local:8000/v1
```

Change to:
```yaml
model: deepseek-v4-flash
base_url: http://macbook-pro.local:<ingot-port>/v1
```

Also update `~/.hermes/.env` and `~/.hermes/hindsight/config.json` to match.

### Step 2: Validate Tool Calling Format (30 min)

V4-Flash may use a different tool call format than Qwen.

- [ ] Check what format V4 uses for function calls (XML like Qwen? JSON? Different tags?)
- [ ] Verify `src/chat/tool_parse.c` handles V4's format
- [ ] Verify chat template injects `<tools>` block correctly for V4
- [ ] Test a Hermes skill that uses tool calling → verify it works end-to-end

### Step 3: Validate Streaming (15 min)

- [ ] Hermes uses `stream: true` — Ingot already does SSE
- [ ] Test multi-turn conversation: Hermes → Ingot → response
- [ ] Verify Hindsight memory extraction works (it calls the same LLM for entity extraction)

### Step 4: Dual Model Setup (30 min)

Decide the architecture for running both models:

**Option A: Replace Qwen entirely**
- Ingot V4 on one port, everything points at it
- Pro: simple. Con: V4 is slower for simple tasks.

**Option B: Side by side**
- oMLX/llama-server serves Qwen on port 8000
- Ingot serves V4 on port 9900
- Hermes uses V4 for agent tasks (smarter), Qwen for Hindsight memory (faster)
- Add model aliases in Hermes config:
  ```yaml
  models:
    v4:  { provider: custom, base_url: "http://macbook-pro.local:9900/v1", model: "deepseek-v4-flash" }
    mac: { provider: custom, base_url: "http://macbook-pro.local:8000/v1", model: "qwen3.6-27b-4bit" }
  ```

**Recommendation:** Option B. V4 for the brain, Qwen for the memory system.

### Step 5: systemd Service (15 min)

Once stable, create a systemd service on the Mac (or launchd plist):
- [ ] Service starts Ingot with V4 model on designated port
- [ ] Restart policy: on-failure with delay
- [ ] Log to file for monitoring

---

## Priority Order

If time is limited, this is the kill chain:

1. **Phase A (tokenizer)** — highest probability root cause, fastest to verify
2. **Phase B (weight names)** — second most likely, mechanical to fix
3. **Phase D (norm tracing)** — reveals any remaining issues
4. **Phase C (Metal)** — probably not blocking, but check
5. **Phase E (benchmarks)** — victory lap
6. **Hermes integration** — config change once inference works

**Estimated total for Part 1:** 6-10 hours focused work.
If tokenizer + weight names are the only issues (likely given the symptoms): 3-4 hours.

**Estimated total for Part 2:** 1-2 hours after Part 1 is done.

---

## Quick Reference

```bash
# Build
make clean && make

# Test V4 load
./ingot serve --model /Users/nic/models/DeepSeek-V4-Flash-ingot/ --port 9900

# Test inference
curl -s http://localhost:9900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# Dump weight names
python3 -c "import json; d=json.load(open('/Users/nic/models/DeepSeek-V4-Flash-ingot/weight_index.json')); [print(k) for k in sorted(d.keys())]"

# Check tokenizer special tokens
python3 -c "import json; d=json.load(open('/Users/nic/models/DeepSeek-V4-Flash-ingot/tokenizer_config.json')); print(json.dumps(d, indent=2))"
```
