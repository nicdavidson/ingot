#define _POSIX_C_SOURCE 200809L

#include "inference/v4_forward.h"
#include "inference/v4_compressor.h"
#include "inference/dequant.h"
#include "model/mmap_pool.h"
#include "util/log.h"
#include "util/timer.h"
#include "util/arena.h"

#ifdef PLATFORM_MACOS
#include "compute/kernels.h"
#include "compute/metal_context.h"
#include "model/expert_io.h"
#include <Accelerate/Accelerate.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// --- V4 inference context ---

struct V4InferenceContext {
    Model            *model;
    const ModelConfig *cfg;
    InferenceCache   *cache;
    AttentionGPU     *attn_gpu;
    Arena             arena;
    V4Compressor     *compressor;

    // Hyper-connection state: 4 copies of hidden state, persistent across layers
    float *hc_state;      // [hc_mult * hidden_size]
    float *hc_copies;     // [hc_mult * hidden_size] — saved for post-HC
    float *sublayer_in;   // [hidden_size] — reduced input for sublayers
    float *sublayer_out;  // [hidden_size] — sublayer output
    float *norm_out;      // [hidden_size] — post-norm for attention/MoE input

    // MoE scratch
    float *gate_logits;   // [num_experts]
    float *expert_out;    // [hidden_size]
    float *expert_buf;    // [moe_intermediate * 3]
    float *logits_buf;    // [vocab_size]

    // Layer tracking
    int    kv_idx;
    int    position;

#ifdef PLATFORM_MACOS
    MetalContext *metal;
    void         *shared_buf;
    bool          use_gpu;
    void         *gpu_norm_out;
    void         *gpu_hidden;       // sublayer_in on GPU
    void         *gpu_out;
    void         *gpu_logits;
    void         *gpu_gate_logits;
    void         *gpu_ffn_mid;
    void         *gpu_expert_result;
    void         *gpu_gate_out;
    void         *gpu_up_out;
    float        *cpu_ffn_mid;
    float        *cpu_expert_result;

    // Per-expert GPU slots (batched dispatch)
    int    num_expert_slots;
    void  *gpu_ffn_mid_slots[16];
    void  *gpu_result_slots[16];
    float *cpu_ffn_mid_slots[16];
    float *cpu_result_slots[16];

    // Expert staging buffers for pread
    void  *gpu_expert_staging[16];
    void  *cpu_expert_staging[16];
    size_t expert_staging_size;
    ExpertIO *expert_io;

    // Deferred expert completion
    void  *prev_expert_signal;
    float  prev_expert_weights[16];
    int    prev_expert_count;
#endif
};


// --- CPU compute primitives (shared with inference.c, duplicated here to avoid coupling) ---

static void cpu_rmsnorm(float *out, const float *x, const float *weight,
                        int n, float eps) {
#ifdef PLATFORM_MACOS
    float ss;
    vDSP_svesq(x, 1, &ss, (vDSP_Length)n);
    float inv_rms = 1.0f / sqrtf(ss / (float)n + eps);
    vDSP_vsmul(x, 1, &inv_rms, out, 1, (vDSP_Length)n);
    vDSP_vmul(out, 1, weight, 1, out, 1, (vDSP_Length)n);
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] / rms) * weight[i];
#endif
}

static void cpu_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void cpu_silu_mul(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float silu = gate[i] / (1.0f + expf(-gate[i]));
        out[i] = silu * up[i];
    }
}

static void bf16_to_float_vec(float *out, const void *bf16_data, int n) {
    const uint16_t *src = bf16_data;
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 7 < n; i += 8) {
        uint16x8_t v = vld1q_u16(src + i);
        uint32x4_t lo = vshll_n_u16(vget_low_u16(v), 16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(v), 16);
        vst1q_f32(out + i, vreinterpretq_f32_u32(lo));
        vst1q_f32(out + i + 4, vreinterpretq_f32_u32(hi));
    }
    for (; i < n; i++) out[i] = bf16_to_f32(src[i]);
#else
    for (int i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
#endif
}


// --- Quantized matmul (same pattern as inference.c) ---

static void q4_matmul(V4InferenceContext *ctx, float *out, const float *x,
                      const char *w_name, const char *s_name, const char *b_name,
                      int M, int K, int group_size) {
#ifdef PLATFORM_MACOS
    if (ctx->use_gpu) {
        long w_off = model_get_weight_offset(ctx->model, w_name);
        long s_off = model_get_weight_offset(ctx->model, s_name);
        long b_off = model_get_weight_offset(ctx->model, b_name);

        if (w_off >= 0 && s_off >= 0 && b_off >= 0) {
            void *x_buf = NULL;
            if (x == ctx->sublayer_in) x_buf = ctx->gpu_hidden;
            if (x == ctx->norm_out)    x_buf = ctx->gpu_norm_out;

            void *out_buf = NULL;
            if (out == ctx->sublayer_in) out_buf = ctx->gpu_hidden;
            if (out == ctx->norm_out)    out_buf = ctx->gpu_norm_out;
            if (out == ctx->logits_buf)  out_buf = ctx->gpu_logits;
            if (out == ctx->gate_logits) out_buf = ctx->gpu_gate_logits;

            if (x_buf && out_buf) {
                kernel_matmul_q4_fma_offsets(ctx->metal, ctx->shared_buf,
                    (size_t)w_off, (size_t)s_off, (size_t)b_off,
                    x_buf, out_buf,
                    (uint32_t)M, (uint32_t)K, (uint32_t)group_size);
                return;
            }
        }
    }
#endif

    size_t ws, ss, bs;
    const void *w = model_get_weight(ctx->model, w_name, &ws);
    const void *s = model_get_weight(ctx->model, s_name, &ss);
    const void *b = model_get_weight(ctx->model, b_name, &bs);
    if (!w || !s || !b) {
        memset(out, 0, (size_t)M * sizeof(float));
        return;
    }
    dequant_matmul_q4(out, w, s, b, x, M, K, group_size);
}


// --- Hyper-Connection Math ---

// Sinkhorn normalization for V4 hyper-connections, matching the reference impl
// (DeepSeek-V4-Flash/inference/kernel.py::hc_split_sinkhorn_kernel):
//   1. Initial row-softmax + eps, then divide by (col_sum + eps).
//   2. (n_iters - 1) iterations of: divide by (row_sum + eps), divide by (col_sum + eps).
// The +eps additions matter — without them this drifts off the reference and HC
// mixing produces wildly different state magnitudes downstream.
static void sinkhorn_normalize(float *m, int n, int n_iters) {
    const float eps = 1e-6f;

    // 1a. Row softmax: m[i,j] = exp(m[i,j] - max_i) / sum_j(exp(...))
    for (int i = 0; i < n; i++) {
        float row_max = m[i * n];
        for (int j = 1; j < n; j++)
            if (m[i * n + j] > row_max) row_max = m[i * n + j];
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            m[i * n + j] = expf(m[i * n + j] - row_max);
            row_sum += m[i * n + j];
        }
        for (int j = 0; j < n; j++)
            m[i * n + j] = m[i * n + j] / row_sum + eps;
    }
    // 1b. Column normalize: m[i,j] /= (col_sum + eps)
    for (int j = 0; j < n; j++) {
        float col_sum = 0.0f;
        for (int i = 0; i < n; i++) col_sum += m[i * n + j];
        float inv = 1.0f / (col_sum + eps);
        for (int i = 0; i < n; i++) m[i * n + j] *= inv;
    }

    // 2. Remaining iterations of row-then-col normalization with +eps denominators.
    for (int iter = 0; iter < n_iters - 1; iter++) {
        for (int i = 0; i < n; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < n; j++) row_sum += m[i * n + j];
            float inv = 1.0f / (row_sum + eps);
            for (int j = 0; j < n; j++) m[i * n + j] *= inv;
        }
        for (int j = 0; j < n; j++) {
            float col_sum = 0.0f;
            for (int i = 0; i < n; i++) col_sum += m[i * n + j];
            float inv = 1.0f / (col_sum + eps);
            for (int i = 0; i < n; i++) m[i * n + j] *= inv;
        }
    }
}

// Hyper-Connection pre-sublayer:
// Takes hc_state [hc_mult * H], produces sublayer input [H].
// Saves copies for post-HC.
// hc_fn: quantized weight [4*H, hc_output_dim], hc_scale: [3], hc_base: [hc_output_dim]
static void v4_hc_pre(V4InferenceContext *ctx, const char *hc_prefix,
                      int layer_idx) {
    const ModelConfig *cfg = ctx->cfg;
    int H = cfg->hidden_size;
    int M = cfg->v4.hc_mult;  // 4
    int hc_out_dim = M + M + M * M;  // pre(4) + post(4) + comb(4x4) = 24
    float eps = cfg->v4.hc_eps;

    // Save copies for post-HC
    memcpy(ctx->hc_copies, ctx->hc_state, (size_t)(M * H) * sizeof(float));

    // RMSNorm over the flattened [M*H] state
    // HC uses its own norm — not a per-layer norm weight.
    // The norm is applied without learned weight (just geometric normalization).
    float ss = 0.0f;
    int MH = M * H;
#ifdef PLATFORM_MACOS
    vDSP_svesq(ctx->hc_state, 1, &ss, (vDSP_Length)MH);
#else
    for (int i = 0; i < MH; i++) ss += ctx->hc_state[i] * ctx->hc_state[i];
#endif
    float inv_rms = 1.0f / sqrtf(ss / (float)MH + eps);

    float *normed = arena_alloc(&ctx->arena, (size_t)MH * sizeof(float));
#ifdef PLATFORM_MACOS
    vDSP_vsmul(ctx->hc_state, 1, &inv_rms, normed, 1, (vDSP_Length)MH);
#else
    for (int i = 0; i < MH; i++) normed[i] = ctx->hc_state[i] * inv_rms;
#endif

    // Project: normed[M*H] @ hc_fn[M*H, hc_out_dim] → proj[hc_out_dim]
    float *proj = arena_alloc(&ctx->arena, (size_t)hc_out_dim * sizeof(float));

    char wn[128], sn[128], bn[128];
    snprintf(wn, sizeof(wn), "layers.%d.%s.fn.weight", layer_idx, hc_prefix);
    snprintf(sn, sizeof(sn), "layers.%d.%s.fn.scales", layer_idx, hc_prefix);
    snprintf(bn, sizeof(bn), "layers.%d.%s.fn.biases", layer_idx, hc_prefix);

    // hc_fn might be quantized (.fn.weight/.scales/.biases) or unquantized
    // (.fn as a single tensor). When unquantized, dtype is detected from size:
    // F32 → 4 bytes/elem, BF16 → 2 bytes/elem.
    size_t ws;
    const void *w = model_get_weight(ctx->model, wn, &ws);
    if (w) {
        q4_matmul(ctx, proj, normed, wn, sn, bn, hc_out_dim, MH, 64);
    } else {
        snprintf(wn, sizeof(wn), "layers.%d.%s.fn", layer_idx, hc_prefix);
        const void *hc_fn = model_get_weight(ctx->model, wn, &ws);
        size_t expected_f32 = (size_t)hc_out_dim * (size_t)MH * sizeof(float);
        if (hc_fn && ws == expected_f32) {
            // F32 matmul
            const float *W = hc_fn;
            for (int i = 0; i < hc_out_dim; i++) {
                float dot = 0.0f;
                for (int j = 0; j < MH; j++)
                    dot += W[i * MH + j] * normed[j];
                proj[i] = dot;
            }
        } else if (hc_fn) {
            // BF16 matmul: proj[i] = sum_j(bf16_to_f32(W[i*MH+j]) * normed[j])
            const uint16_t *W = hc_fn;
            for (int i = 0; i < hc_out_dim; i++) {
                float dot = 0.0f;
                for (int j = 0; j < MH; j++)
                    dot += bf16_to_f32(W[i * MH + j]) * normed[j];
                proj[i] = dot;
            }
        } else {
            memset(proj, 0, (size_t)hc_out_dim * sizeof(float));
            LOG_WARN("v4: missing hc_fn weight for layer %d %s", layer_idx, hc_prefix);
        }
    }

    // Apply scale and base: proj = proj * scale + base
    // hc_scale[3]: one per section (pre, post, comb)
    // hc_base[hc_out_dim]: one per output element
    char scale_name[128], base_name[128];
    snprintf(scale_name, sizeof(scale_name), "layers.%d.%s.scale", layer_idx, hc_prefix);
    snprintf(base_name, sizeof(base_name), "layers.%d.%s.base", layer_idx, hc_prefix);

    size_t scale_sz, base_sz;
    const void *scale_data = model_get_weight(ctx->model, scale_name, &scale_sz);
    const void *base_data = model_get_weight(ctx->model, base_name, &base_sz);

    float scale[3] = {1.0f, 1.0f, 1.0f};
    float *base = arena_alloc(&ctx->arena, (size_t)hc_out_dim * sizeof(float));
    memset(base, 0, (size_t)hc_out_dim * sizeof(float));

    if (scale_data) {
        // Could be BF16 or F32
        if (scale_sz == 3 * sizeof(float)) {
            memcpy(scale, scale_data, sizeof(scale));
        } else {
            const uint16_t *s16 = scale_data;
            for (int i = 0; i < 3; i++) scale[i] = bf16_to_f32(s16[i]);
        }
    }
    if (base_data) {
        if (base_sz == (size_t)hc_out_dim * sizeof(float)) {
            memcpy(base, base_data, base_sz);
        } else {
            const uint16_t *b16 = base_data;
            for (int i = 0; i < hc_out_dim; i++) base[i] = bf16_to_f32(b16[i]);
        }
    }

    // Apply: pre section (0..M), post section (M..2M), comb section (2M..2M+M*M)
    for (int i = 0; i < M; i++)
        proj[i] = proj[i] * scale[0] + base[i];
    for (int i = M; i < 2 * M; i++)
        proj[i] = proj[i] * scale[1] + base[i];
    for (int i = 2 * M; i < hc_out_dim; i++)
        proj[i] = proj[i] * scale[2] + base[i];

    // Reference: pre[j] = sigmoid(...) + eps   (DeepSeek-V4 kernel.py:392)
    const float hc_eps = cfg->v4.hc_eps;
    float pre[4];
    for (int i = 0; i < M; i++)
        pre[i] = 1.0f / (1.0f + expf(-proj[i])) + hc_eps;
    // Reference: post[j] = 2 * sigmoid(...)   (DeepSeek-V4 kernel.py:394)
    for (int i = 0; i < M; i++)
        proj[M + i] = 2.0f / (1.0f + expf(-proj[M + i]));


    // Store post weights for post-HC (used after sublayer)
    // We stash them in the arena — post_weights and comb will be needed in v4_hc_post
    // For now, store proj pointer and parse in post.
    // Actually, let's compute sublayer input here and store post/comb for later.

    // Compute sublayer input: y[H] = sum_m(pre[m] * copies[m, :])
    memset(ctx->sublayer_in, 0, (size_t)H * sizeof(float));
    for (int m = 0; m < M; m++) {
        const float *copy = ctx->hc_copies + m * H;
#ifdef PLATFORM_MACOS
        vDSP_vsma(copy, 1, &pre[m], ctx->sublayer_in, 1, ctx->sublayer_in, 1, (vDSP_Length)H);
#else
        for (int i = 0; i < H; i++)
            ctx->sublayer_in[i] += pre[m] * copy[i];
#endif
    }

    // Sinkhorn-normalize the comb matrix
    float *comb = proj + 2 * M;  // [M*M] = [4*4]
    sinkhorn_normalize(comb, M, cfg->v4.hc_sinkhorn_iters);

    // Store post weights [M] and comb matrix [M*M] for v4_hc_post
    // We use a known layout at the start of the arena (allocated but not freed until layer end)
    float *stored_post = arena_alloc(&ctx->arena, (size_t)M * sizeof(float));
    float *stored_comb = arena_alloc(&ctx->arena, (size_t)(M * M) * sizeof(float));
    memcpy(stored_post, proj + M, (size_t)M * sizeof(float));
    memcpy(stored_comb, comb, (size_t)(M * M) * sizeof(float));
}

// Hyper-Connection post-sublayer:
// Takes sublayer output [H] and updates hc_state [M*H] using stored post/comb.
// post_weights and comb_matrix are the last arena allocations from v4_hc_pre.
static void v4_hc_post(V4InferenceContext *ctx,
                       const float *sublayer_output,
                       const float *post_weights,    // [M]
                       const float *comb_matrix) {    // [M*M]
    int H = ctx->cfg->hidden_size;
    int M = ctx->cfg->v4.hc_mult;

    // Reference (DeepSeek-V4-Flash inference/model.py::hc_post):
    //   y = post.unsqueeze(-1) * x.unsqueeze(-2) + sum_a comb[a, b] * residual[a, k]
    //                                                   (a is summed-over axis,
    //                                                    b is output copy index)
    // i.e. new_state[b, k] = post[b] * x[k] + sum_a comb[a, b] * old_copies[a, k]
    // The summed-over index is the FIRST dim of comb, so when iterating output
    // copy m we must read comb[a*M + m] (transposed vs the obvious order).
    for (int m = 0; m < M; m++) {
        float *dst = ctx->hc_state + m * H;
        float post_w = post_weights[m];

        for (int i = 0; i < H; i++)
            dst[i] = post_w * sublayer_output[i];

        for (int j = 0; j < M; j++) {
            float c = comb_matrix[j * M + m];   // comb[j, m] — j is summed
            const float *copy = ctx->hc_copies + j * H;
#ifdef PLATFORM_MACOS
            vDSP_vsma(copy, 1, &c, dst, 1, dst, 1, (vDSP_Length)H);
#else
            for (int i = 0; i < H; i++)
                dst[i] += c * copy[i];
#endif
        }
    }
}


// --- V4 sqrtsoftplus gating ---

static void v4_sqrtsoftplus_gate(float *probs, const float *logits, int n) {
    // sqrtsoftplus(x) = sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
    for (int i = 0; i < n; i++) {
        float x = logits[i];
        float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
        probs[i] = sqrtf(sp > 0.0f ? sp : 1e-8f);
    }
}

// --- Creation / Destruction ---

V4InferenceContext *v4_inference_create(Model *model, const ModelConfig *cfg,
                                        InferenceCache *cache, AttentionGPU *attn_gpu) {
    V4InferenceContext *ctx = calloc(1, sizeof(V4InferenceContext));
    ctx->model = model;
    ctx->cfg = cfg;
    ctx->cache = cache;
    ctx->attn_gpu = attn_gpu;
    ctx->arena = arena_create(64 * 1024 * 1024);

    // Compressor stores rolling state + compressed-KV cache for layers with
    // compress_ratio > 0. Cap to 4096 max_seq for testing — real serving will
    // need this sized off cache->kv_layers[0].max_seq, but the smaller cap
    // keeps startup memory sane and avoids allocating 300+ MB of unused cache
    // for short prompts.
    ctx->compressor = v4_compressor_create(cfg, 4096);

    int H = cfg->hidden_size;
    int M = cfg->v4.hc_mult;

    ctx->hc_state    = calloc((size_t)(M * H), sizeof(float));
    ctx->hc_copies   = calloc((size_t)(M * H), sizeof(float));
    ctx->sublayer_in = calloc((size_t)H, sizeof(float));
    ctx->sublayer_out = calloc((size_t)H, sizeof(float));
    ctx->norm_out    = calloc((size_t)H, sizeof(float));
    ctx->gate_logits = calloc((size_t)cfg->num_experts, sizeof(float));
    ctx->expert_out  = calloc((size_t)H, sizeof(float));
    ctx->expert_buf  = calloc((size_t)cfg->moe_intermediate_size * 3, sizeof(float));
    ctx->logits_buf  = calloc((size_t)cfg->vocab_size, sizeof(float));

#ifdef PLATFORM_MACOS
    ctx->metal = model_get_metal(model);
    ctx->shared_buf = model_get_metal_shared_buf(model);
    if (ctx->metal && ctx->shared_buf) {
        int max_out = cfg->vocab_size > H ? cfg->vocab_size : H;
        ctx->gpu_hidden   = metal_alloc_buffer(ctx->metal, (size_t)H * sizeof(float));
        ctx->gpu_norm_out = metal_alloc_buffer(ctx->metal, (size_t)H * sizeof(float));
        ctx->gpu_out      = metal_alloc_buffer(ctx->metal, (size_t)max_out * sizeof(float));
        ctx->gpu_logits   = metal_alloc_buffer(ctx->metal, (size_t)cfg->vocab_size * sizeof(float));
        ctx->gpu_gate_logits = metal_alloc_buffer(ctx->metal, (size_t)cfg->num_experts * sizeof(float));

        int expert_dim = cfg->moe_intermediate_size;
        if (cfg->shared_expert_intermediate_size > expert_dim)
            expert_dim = cfg->shared_expert_intermediate_size;
        ctx->gpu_gate_out      = metal_alloc_buffer(ctx->metal, (size_t)expert_dim * sizeof(float));
        ctx->gpu_up_out        = metal_alloc_buffer(ctx->metal, (size_t)expert_dim * sizeof(float));
        ctx->gpu_ffn_mid       = metal_alloc_buffer(ctx->metal, (size_t)expert_dim * sizeof(float));
        ctx->gpu_expert_result = metal_alloc_buffer(ctx->metal, (size_t)H * sizeof(float));

        if (ctx->gpu_hidden && ctx->gpu_norm_out && ctx->gpu_out && ctx->gpu_logits &&
            ctx->gpu_gate_out && ctx->gpu_up_out && ctx->gpu_ffn_mid && ctx->gpu_expert_result) {

            free(ctx->sublayer_in);
            ctx->sublayer_in = metal_buffer_contents(ctx->gpu_hidden);
            free(ctx->norm_out);
            ctx->norm_out = metal_buffer_contents(ctx->gpu_norm_out);
            free(ctx->logits_buf);
            ctx->logits_buf = metal_buffer_contents(ctx->gpu_logits);
            free(ctx->gate_logits);
            ctx->gate_logits = metal_buffer_contents(ctx->gpu_gate_logits);
            ctx->cpu_ffn_mid = metal_buffer_contents(ctx->gpu_ffn_mid);
            ctx->cpu_expert_result = metal_buffer_contents(ctx->gpu_expert_result);
            ctx->use_gpu = true;

            int K_max = cfg->num_experts_per_tok;
            if (K_max > 16) K_max = 16;
            bool slots_ok = true;
            for (int i = 0; i < K_max; i++) {
                ctx->gpu_ffn_mid_slots[i] = metal_alloc_buffer(ctx->metal, (size_t)expert_dim * sizeof(float));
                ctx->gpu_result_slots[i] = metal_alloc_buffer(ctx->metal, (size_t)H * sizeof(float));
                if (ctx->gpu_ffn_mid_slots[i] && ctx->gpu_result_slots[i]) {
                    ctx->cpu_ffn_mid_slots[i] = metal_buffer_contents(ctx->gpu_ffn_mid_slots[i]);
                    ctx->cpu_result_slots[i] = metal_buffer_contents(ctx->gpu_result_slots[i]);
                } else {
                    slots_ok = false;
                    break;
                }
            }
            if (slots_ok) ctx->num_expert_slots = K_max;

            size_t max_stride = model_get_expert_stride(model, 0);
            if (max_stride > 0) {
                bool staging_ok = true;
                for (int i = 0; i < K_max; i++) {
                    ctx->gpu_expert_staging[i] = metal_alloc_buffer_aligned(ctx->metal, max_stride, 2 * 1024 * 1024);
                    if (ctx->gpu_expert_staging[i])
                        ctx->cpu_expert_staging[i] = metal_buffer_contents(ctx->gpu_expert_staging[i]);
                    else { staging_ok = false; break; }
                }
                if (staging_ok) {
                    ctx->expert_staging_size = max_stride;
                    ctx->expert_io = expert_io_create(K_max);
                }
            }

            LOG_INFO("v4: GPU acceleration enabled (%d expert slots)", ctx->num_expert_slots);
        }
    }
#endif

    LOG_INFO("v4: context created (hc_mult=%d, hidden=%d)", M, H);
    return ctx;
}

void v4_init_hc_state(V4InferenceContext *v4, const float *embedding, int hidden_size) {
    int M = v4->cfg->v4.hc_mult;
    // Initialize all M copies to the same embedding
    for (int m = 0; m < M; m++)
        memcpy(v4->hc_state + m * hidden_size, embedding, (size_t)hidden_size * sizeof(float));
    // Reset per-token layer counters (kv_idx tracks KV cache layer slot)
    v4->kv_idx = 0;
}


// --- V4 Timing ---

static double _v4_acc_hc = 0, _v4_acc_attn = 0, _v4_acc_shared = 0, _v4_acc_routed = 0;

void v4_timing_reset(void) {
    _v4_acc_hc = _v4_acc_attn = _v4_acc_shared = _v4_acc_routed = 0;
}

void v4_timing_report(int token_num) {
    double total = _v4_acc_hc + _v4_acc_attn + _v4_acc_shared + _v4_acc_routed;
    fprintf(stderr, "[V4-TIMING tok%d] hc=%.1f attn=%.1f shared=%.1f routed=%.1f TOTAL=%.1f ms\n",
            token_num, _v4_acc_hc, _v4_acc_attn, _v4_acc_shared, _v4_acc_routed, total);
}

// --- V4 Forward Layer ---

#ifdef PLATFORM_MACOS
static void v4_complete_deferred_experts(V4InferenceContext *ctx) {
    if (!ctx->prev_expert_signal) return;

    kernel_wait_deferred(ctx->prev_expert_signal);
    ctx->prev_expert_signal = NULL;

    int H = ctx->cfg->hidden_size;

    for (int k = 0; k < ctx->prev_expert_count; k++) {
        float weight = ctx->prev_expert_weights[k];
        float *result = ctx->cpu_result_slots[k];
        vDSP_vsma(result, 1, &weight, ctx->sublayer_out, 1, ctx->sublayer_out, 1, (vDSP_Length)H);
    }
    ctx->prev_expert_count = 0;
}
#endif

void v4_forward_layer(V4InferenceContext *ctx, int layer_idx, int position, int token_id) {
    const ModelConfig *cfg = ctx->cfg;
    int H = cfg->hidden_size;
    int M = cfg->v4.hc_mult;
    uint64_t _t0 = timer_now_ns(), _t1;

    ctx->position = position;
    (void)M;

#ifdef PLATFORM_MACOS
    v4_complete_deferred_experts(ctx);
#endif

    // ====== ATTENTION BLOCK ======

    // HC pre for attention: reduces [4*H] → [H] sublayer_in
    // Arena allocations for post_weights and comb_matrix happen inside
    size_t arena_mark = ctx->arena.used;
    v4_hc_pre(ctx, "attn_hc", layer_idx);

    // We need to find the post_weights and comb_matrix that were arena-allocated.
    // They're the last two allocations: post[M] then comb[M*M].
    // Reconstruct pointers: comb is at current arena end minus M*M*4,
    // post is before that at minus (M*M + M)*4
    float *attn_comb = (float *)((char *)ctx->arena.buf + ctx->arena.used) - M * M;
    float *attn_post = attn_comb - M;

    // Pre-attention RMSNorm
    char name[128];
    snprintf(name, sizeof(name), "layers.%d.attn_norm.weight", layer_idx);
    size_t nsz;
    const void *norm_bf16 = model_get_weight(ctx->model, name, &nsz);
    if (!norm_bf16) return;

    float *norm_weight = arena_alloc(&ctx->arena, (size_t)H * sizeof(float));
    bf16_to_float_vec(norm_weight, norm_bf16, H);
    cpu_rmsnorm(ctx->norm_out, ctx->sublayer_in, norm_weight, H, cfg->rms_norm_eps);

    // Compressor (CSA: ratio=4 / HCA: ratio=128). Updates rolling window
    // buffers each token and emits a new compressed-KV entry every `ratio`
    // tokens. Operates on the SAME post-attn_norm hidden as the attention
    // wkv projection (matches reference Attention.forward).
    int compressed_count = 0;
    const float *compressed_kv = NULL;
    if (cfg->v4.compress_ratios && cfg->v4.compress_ratios[layer_idx] > 0) {
        v4_compressor_step(ctx->compressor, ctx->model, cfg, layer_idx,
                           ctx->norm_out, position);
        compressed_kv = v4_compressor_cache(ctx->compressor, layer_idx,
                                            &compressed_count);
    }

    // Attention forward pass (MLA)
#ifdef PLATFORM_MACOS
    if (ctx->use_gpu)
        attention_gpu_set_input(ctx->attn_gpu, ctx->gpu_norm_out, ctx->norm_out);
#endif

    attention_v4_mla_forward(ctx->sublayer_out, ctx->norm_out, ctx->model, cfg,
                             ctx->cache, ctx->attn_gpu,
                             layer_idx, ctx->kv_idx, position,
                             compressed_kv, compressed_count);
    ctx->kv_idx++;

    // HC post for attention: updates hc_state using sublayer_out
    v4_hc_post(ctx, ctx->sublayer_out, attn_post, attn_comb);
    _t1 = timer_now_ns(); _v4_acc_attn += timer_elapsed_ms(_t0, _t1); _t0 = _t1;

    // Reset arena for MoE block (post/comb no longer needed)
    ctx->arena.used = arena_mark;

    // ====== MoE BLOCK ======

    // HC pre for FFN
    v4_hc_pre(ctx, "ffn_hc", layer_idx);

    float *ffn_comb = (float *)((char *)ctx->arena.buf + ctx->arena.used) - M * M;
    float *ffn_post = ffn_comb - M;

    // Post-attention layernorm
    snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", layer_idx);
    const void *post_norm_bf16 = model_get_weight(ctx->model, name, &nsz);
    if (post_norm_bf16) {
        bf16_to_float_vec(norm_weight, post_norm_bf16, H);
        cpu_rmsnorm(ctx->norm_out, ctx->sublayer_in, norm_weight, H, cfg->rms_norm_eps);
    }

    // MoE Gate
    int num_experts = cfg->num_experts;
    int K_active = cfg->num_experts_per_tok;

    // V4 uses "ffn.gate" not "mlp.gate"
    snprintf(name, sizeof(name), "layers.%d.ffn.gate.weight", layer_idx);
    char sname[128], bname[128];
    snprintf(sname, sizeof(sname), "layers.%d.ffn.gate.scales", layer_idx);
    snprintf(bname, sizeof(bname), "layers.%d.ffn.gate.biases", layer_idx);

    // Check if this is a hash-routing layer (first num_hash_layers layers)
    bool use_hash = (layer_idx < cfg->v4.num_hash_layers);

    if (!use_hash) {
        // Non-hash V4 gate: BF16 weight [num_experts, H], no scales/biases.
        // Reference Gate.forward (model.py:564): scores = sqrtsoftplus(W @ x).
        // For top-K selection, ADD e_score_correction_bias to a copy. Weights
        // are gathered from the UNBIASED scores (the "noaux_tc" trick).
        size_t gw_sz;
        const void *gate_w_bf16 = model_get_weight(ctx->model, name, &gw_sz);
        if (gate_w_bf16 && gw_sz == (size_t)num_experts * (size_t)H * sizeof(uint16_t)) {
            const uint16_t *W = gate_w_bf16;
            for (int e = 0; e < num_experts; e++) {
                float dot = 0.0f;
                const uint16_t *row = W + (size_t)e * (size_t)H;
                for (int k = 0; k < H; k++)
                    dot += bf16_to_f32(row[k]) * ctx->norm_out[k];
                ctx->gate_logits[e] = dot;
            }
        } else {
            q4_matmul(ctx, ctx->gate_logits, ctx->norm_out, name, sname, bname,
                      num_experts, H, 64);
        }

        // sqrtsoftplus → original_scores (unbiased, used for weights).
        float *gate_probs = arena_alloc(&ctx->arena, (size_t)num_experts * sizeof(float));
        v4_sqrtsoftplus_gate(gate_probs, ctx->gate_logits, num_experts);

        // Biased copy for top-K selection only.
        char bias_name[160];
        snprintf(bias_name, sizeof(bias_name),
                 "layers.%d.ffn.gate.e_score_correction_bias", layer_idx);
        size_t bias_sz;
        const void *bias_raw = model_get_weight(ctx->model, bias_name, &bias_sz);
        float *biased_probs = arena_alloc(&ctx->arena,
                                          (size_t)num_experts * sizeof(float));
        memcpy(biased_probs, gate_probs, (size_t)num_experts * sizeof(float));
        if (bias_raw && bias_sz == (size_t)num_experts * sizeof(float)) {
            const float *bias_v = bias_raw;
            for (int e = 0; e < num_experts; e++)
                biased_probs[e] += bias_v[e];
        }

        // Top-K selection on biased scores; weights come from unbiased scores.
        int top_indices[16];
        float top_weights[16];
        for (int k = 0; k < K_active && k < 16; k++) {
            float best = -1e30f;
            int best_idx = 0;
            for (int e = 0; e < num_experts; e++) {
                bool skip = false;
                for (int j = 0; j < k; j++)
                    if (top_indices[j] == e) { skip = true; break; }
                if (skip) continue;
                if (biased_probs[e] > best) {
                    best = biased_probs[e];
                    best_idx = e;
                }
            }
            top_indices[k] = best_idx;
            top_weights[k] = gate_probs[best_idx];   // unbiased weight
        }

        // Renormalize weights, then scale by routed_scaling_factor (V4: 1.5).
        float wsum = 0.0f;
        for (int k = 0; k < K_active; k++) wsum += top_weights[k];
        float route_scale_nh = (float)cfg->v4.route_scale;
        if (route_scale_nh == 0.0f) route_scale_nh = 1.0f;
        if (wsum > 0.0f)
            for (int k = 0; k < K_active; k++)
                top_weights[k] = (top_weights[k] / wsum) * route_scale_nh;

        // Kick off parallel pread for routed experts (overlaps with shared expert)
#ifdef PLATFORM_MACOS
        bool pread_experts = false;
        size_t expert_stride = model_get_expert_stride(ctx->model, layer_idx);
        int expert_fd = model_get_expert_fd(ctx->model, layer_idx);
        if (ctx->use_gpu && ctx->expert_io && expert_fd >= 0 &&
            expert_stride > (2 * 1024 * 1024) &&
            K_active <= ctx->num_expert_slots &&
            expert_stride <= ctx->expert_staging_size) {
            size_t offsets[16], sizes[16];
            void *dests[16];
            for (int k = 0; k < K_active; k++) {
                offsets[k] = (size_t)top_indices[k] * expert_stride;
                sizes[k]   = expert_stride;
                dests[k]   = ctx->cpu_expert_staging[k];
            }
            expert_io_fetch(ctx->expert_io, expert_fd, offsets, sizes, dests, K_active);
            pread_experts = true;
        }
#endif

        // Shared expert FFN (V4: no sigmoid gate, direct add — runs while pread is in flight)
        int moe_dim = cfg->shared_expert_intermediate_size;
        bool shared_on_gpu = false;

#ifdef PLATFORM_MACOS
        if (ctx->use_gpu) {
            char gw[128], gs_n[128], gb_n[128], uw[128], us_n[128], ub_n[128];
            char dw_n[128], ds_n[128], db_n[128];
            snprintf(gw, sizeof(gw), "layers.%d.ffn.shared_experts.gate_proj.weight", layer_idx);
            snprintf(gs_n, sizeof(gs_n), "layers.%d.ffn.shared_experts.gate_proj.scales", layer_idx);
            snprintf(gb_n, sizeof(gb_n), "layers.%d.ffn.shared_experts.gate_proj.biases", layer_idx);
            snprintf(uw, sizeof(uw), "layers.%d.ffn.shared_experts.up_proj.weight", layer_idx);
            snprintf(us_n, sizeof(us_n), "layers.%d.ffn.shared_experts.up_proj.scales", layer_idx);
            snprintf(ub_n, sizeof(ub_n), "layers.%d.ffn.shared_experts.up_proj.biases", layer_idx);
            snprintf(dw_n, sizeof(dw_n), "layers.%d.ffn.shared_experts.down_proj.weight", layer_idx);
            snprintf(ds_n, sizeof(ds_n), "layers.%d.ffn.shared_experts.down_proj.scales", layer_idx);
            snprintf(db_n, sizeof(db_n), "layers.%d.ffn.shared_experts.down_proj.biases", layer_idx);

            long gw_off = model_get_weight_offset(ctx->model, gw);
            long gs_off = model_get_weight_offset(ctx->model, gs_n);
            long gb_off = model_get_weight_offset(ctx->model, gb_n);
            long uw_off = model_get_weight_offset(ctx->model, uw);
            long us_off = model_get_weight_offset(ctx->model, us_n);
            long ub_off = model_get_weight_offset(ctx->model, ub_n);
            long dw_off = model_get_weight_offset(ctx->model, dw_n);
            long ds_off = model_get_weight_offset(ctx->model, ds_n);
            long db_off = model_get_weight_offset(ctx->model, db_n);

            if (gw_off >= 0 && uw_off >= 0 && dw_off >= 0) {
                void *batch = kernel_begin_batch(ctx->metal);

                kernel_batch_fused_gate_up_swiglu_offsets(batch, ctx->shared_buf,
                    (size_t)gw_off, (size_t)gs_off, (size_t)gb_off,
                    (size_t)uw_off, (size_t)us_off, (size_t)ub_off,
                    ctx->gpu_norm_out, ctx->gpu_ffn_mid,
                    (uint32_t)moe_dim, (uint32_t)H, 64);

                kernel_batch_q4_fma_offsets(batch, ctx->shared_buf,
                    (size_t)dw_off, (size_t)ds_off, (size_t)db_off,
                    ctx->gpu_ffn_mid, ctx->gpu_expert_result,
                    (uint32_t)H, (uint32_t)moe_dim, 64);

                kernel_end_batch(batch);
                memcpy(ctx->sublayer_out, ctx->cpu_expert_result, (size_t)H * sizeof(float));
                shared_on_gpu = true;
            }
        }
#endif

        if (!shared_on_gpu) {
            float *gate_out = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
            float *up_out   = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
            float *ffn_mid  = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));

            snprintf(name, sizeof(name), "layers.%d.ffn.shared_experts.gate_proj.weight", layer_idx);
            snprintf(sname, sizeof(sname), "layers.%d.ffn.shared_experts.gate_proj.scales", layer_idx);
            snprintf(bname, sizeof(bname), "layers.%d.ffn.shared_experts.gate_proj.biases", layer_idx);
            q4_matmul(ctx, gate_out, ctx->norm_out, name, sname, bname, moe_dim, H, 64);

            snprintf(name, sizeof(name), "layers.%d.ffn.shared_experts.up_proj.weight", layer_idx);
            snprintf(sname, sizeof(sname), "layers.%d.ffn.shared_experts.up_proj.scales", layer_idx);
            snprintf(bname, sizeof(bname), "layers.%d.ffn.shared_experts.up_proj.biases", layer_idx);
            q4_matmul(ctx, up_out, ctx->norm_out, name, sname, bname, moe_dim, H, 64);

            // Reference shared expert constructs Expert() with default
            // swiglu_limit=0, which skips the clamp entirely (model.py:628).
            cpu_silu_mul(ffn_mid, gate_out, up_out, moe_dim);

            snprintf(name, sizeof(name), "layers.%d.ffn.shared_experts.down_proj.weight", layer_idx);
            snprintf(sname, sizeof(sname), "layers.%d.ffn.shared_experts.down_proj.scales", layer_idx);
            snprintf(bname, sizeof(bname), "layers.%d.ffn.shared_experts.down_proj.biases", layer_idx);
            q4_matmul(ctx, ctx->sublayer_out, ffn_mid, name, sname, bname, H, moe_dim, 64);
        }

        // Routed experts
        moe_dim = cfg->moe_intermediate_size;
        // V4 routed experts use MXFP4 (FP4 + E8M0 group exponent, group_size=32, no biases).
        // The GPU shaders below assume Qwen-style int4 + BF16 affine quant, so they cannot
        // process V4 experts correctly. Fall through to the CPU MXFP4 path until shaders
        // are taught the new format.
        int group_size = 32;
        size_t w_size = (size_t)moe_dim * (size_t)(H / 8) * 4;
        size_t s_size = (size_t)moe_dim * (size_t)(H / group_size) * 1;
        size_t dw_size = (size_t)H * (size_t)(moe_dim / 8) * 4;
        size_t ds_size = (size_t)H * (size_t)(moe_dim / group_size) * 1;
        size_t proj_size = w_size + s_size;  // unused — kept for the disabled GPU paths
        (void)proj_size;

#ifdef PLATFORM_MACOS
        if (false && ctx->use_gpu && K_active <= ctx->num_expert_slots && pread_experts) {
            expert_io_wait(ctx->expert_io);

            void *batch = kernel_begin_batch(ctx->metal);
            for (int k = 0; k < K_active; k++) {
                size_t down_base = proj_size * 2;
                kernel_batch_fused_gate_up_swiglu_offsets(batch,
                    ctx->gpu_expert_staging[k],
                    0, w_size, w_size + s_size,
                    proj_size, proj_size + w_size, proj_size + w_size + s_size,
                    ctx->gpu_norm_out, ctx->gpu_ffn_mid_slots[k],
                    (uint32_t)moe_dim, (uint32_t)H, (uint32_t)group_size);
                kernel_batch_q4_fma_offsets(batch,
                    ctx->gpu_expert_staging[k],
                    down_base, down_base + dw_size, down_base + dw_size + ds_size,
                    ctx->gpu_ffn_mid_slots[k], ctx->gpu_result_slots[k],
                    (uint32_t)H, (uint32_t)moe_dim, (uint32_t)group_size);
            }
            kernel_end_batch(batch);

            for (int k = 0; k < K_active; k++) {
                float weight = top_weights[k];
                float *result = ctx->cpu_result_slots[k];
                vDSP_vsma(result, 1, &weight, ctx->sublayer_out, 1,
                          ctx->sublayer_out, 1, (vDSP_Length)H);
            }
        } else if (false && ctx->use_gpu && K_active <= ctx->num_expert_slots) {
            void *expert_buf = model_get_expert_metal_buf(ctx->model, layer_idx);
            if (expert_buf) {
                void *batch = kernel_begin_batch(ctx->metal);
                int batch_count = 0;
                float batch_weights[16];
                for (int k = 0; k < K_active; k++) {
                    int expert_idx = top_indices[k];
                    size_t stride;
                    const void *expert_data = model_get_expert(ctx->model, layer_idx,
                                                                expert_idx, &stride);
                    if (!expert_data) continue;
                    size_t base = (size_t)expert_idx * stride;
                    size_t down_base = base + proj_size * 2;
                    kernel_batch_fused_gate_up_swiglu_offsets(batch, expert_buf,
                        base, base + w_size, base + w_size + s_size,
                        base + proj_size, base + proj_size + w_size,
                        base + proj_size + w_size + s_size,
                        ctx->gpu_norm_out, ctx->gpu_ffn_mid_slots[batch_count],
                        (uint32_t)moe_dim, (uint32_t)H, (uint32_t)group_size);
                    kernel_batch_q4_fma_offsets(batch, expert_buf,
                        down_base, down_base + dw_size, down_base + dw_size + ds_size,
                        ctx->gpu_ffn_mid_slots[batch_count], ctx->gpu_result_slots[batch_count],
                        (uint32_t)H, (uint32_t)moe_dim, (uint32_t)group_size);
                    batch_weights[batch_count] = top_weights[k];
                    batch_count++;
                }
                kernel_end_batch(batch);

                for (int k = 0; k < batch_count; k++) {
                    float weight = batch_weights[k];
                    float *result = ctx->cpu_result_slots[k];
                    vDSP_vsma(result, 1, &weight, ctx->sublayer_out, 1,
                              ctx->sublayer_out, 1, (vDSP_Length)H);
                }
            }
        } else
#endif
        {
            for (int k = 0; k < K_active; k++) {
                int expert_idx = top_indices[k];
                float weight = top_weights[k];
                size_t stride;
                const void *expert_data = model_get_expert(ctx->model, layer_idx,
                                                            expert_idx, &stride);
                if (!expert_data) continue;

                const char *ptr = expert_data;
                const void *gw = ptr; ptr += w_size;
                const void *gs = ptr; ptr += s_size;

                float *r_gate = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
                dequant_matmul_mxfp4(r_gate, gw, gs, ctx->norm_out, moe_dim, H, group_size);

                const void *uw = ptr; ptr += w_size;
                const void *us = ptr; ptr += s_size;

                float *r_up = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
                dequant_matmul_mxfp4(r_up, uw, us, ctx->norm_out, moe_dim, H, group_size);

                // Reference Expert.forward (model.py:601):
                //   up = clamp(up, -limit, limit)  (symmetric)
                //   gate = clamp(gate, max=limit)  (UPPER ONLY — not symmetric)
                for (int i = 0; i < moe_dim; i++) {
                    if (r_gate[i] > 10.0f) r_gate[i] = 10.0f;
                    if (r_up[i]   >  10.0f) r_up[i]   =  10.0f;
                    if (r_up[i]   < -10.0f) r_up[i]   = -10.0f;
                }

                float *r_mid = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
                cpu_silu_mul(r_mid, r_gate, r_up, moe_dim);

                const void *dw = ptr; ptr += dw_size;
                const void *ds = ptr; ptr += ds_size;

                float *result = arena_alloc(&ctx->arena, (size_t)H * sizeof(float));
                dequant_matmul_mxfp4(result, dw, ds, r_mid, H, moe_dim, group_size);

#ifdef PLATFORM_MACOS
                vDSP_vsma(result, 1, &weight, ctx->sublayer_out, 1, ctx->sublayer_out, 1, (vDSP_Length)H);
#else
                for (int i = 0; i < H; i++)
                    ctx->sublayer_out[i] += weight * result[i];
#endif
            }
        }
    } else {
        // Hash routing: tid2eid lookup chooses the K_active experts per token,
        // but the WEIGHTS are still derived from the gate matmul + sqrtsoftplus
        // (reference Gate.forward, model.py:564). For hash layers, the gate
        // weight is stored as raw BF16 (not quantized) — q4_matmul would zero
        // out the output because scales/biases don't exist.
        int top_indices[16];
        float top_weights[16];

        // 1. Compute gate scores via plain BF16 matmul.
        size_t gw_sz;
        const void *gate_w_bf16 = model_get_weight(ctx->model, name, &gw_sz);
        if (gate_w_bf16 && gw_sz == (size_t)num_experts * (size_t)H * sizeof(uint16_t)) {
            const uint16_t *W = gate_w_bf16;
            for (int e = 0; e < num_experts; e++) {
                float dot = 0.0f;
                const uint16_t *row = W + (size_t)e * (size_t)H;
                for (int k = 0; k < H; k++)
                    dot += bf16_to_f32(row[k]) * ctx->norm_out[k];
                ctx->gate_logits[e] = dot;
            }
        } else {
            // Fall back to the quantized path in case the converter ever
            // packs hash-layer gates as q4.
            q4_matmul(ctx, ctx->gate_logits, ctx->norm_out, name, sname, bname,
                      num_experts, H, 64);
        }
        float *hash_probs = arena_alloc(&ctx->arena,
                                        (size_t)num_experts * sizeof(float));
        v4_sqrtsoftplus_gate(hash_probs, ctx->gate_logits, num_experts);

        // 2. Look up the K_active expert indices for this token.
        char tid_name[128];
        snprintf(tid_name, sizeof(tid_name), "layers.%d.ffn.gate.tid2eid", layer_idx);
        size_t tid_size;
        const void *tid2eid = model_get_weight(ctx->model, tid_name, &tid_size);
        if (tid2eid && token_id >= 0 && token_id < cfg->vocab_size) {
            const int64_t *table = (const int64_t *)tid2eid;
            for (int k = 0; k < K_active && k < 16; k++) {
                int idx = (int)table[token_id * K_active + k];
                top_indices[k] = idx;
                top_weights[k] = (idx >= 0 && idx < num_experts)
                    ? hash_probs[idx] : 0.0f;
            }
        } else {
            LOG_WARN("v4: tid2eid missing for layer %d, token %d", layer_idx, token_id);
            for (int k = 0; k < K_active && k < 16; k++) {
                top_indices[k] = k;
                top_weights[k] = hash_probs[k];
            }
        }

        // 3. Renormalize so weights sum to 1, then scale by route_scale.
        float wsum = 0.0f;
        for (int k = 0; k < K_active; k++) wsum += top_weights[k];
        float route_scale = (float)cfg->v4.route_scale;
        if (route_scale == 0.0f) route_scale = 1.0f;
        if (wsum > 0.0f)
            for (int k = 0; k < K_active; k++)
                top_weights[k] = (top_weights[k] / wsum) * route_scale;

        // Shared expert FFN (same as standard path, CPU)
        int hr_dim = cfg->shared_expert_intermediate_size;
        float *hr_gate = arena_alloc(&ctx->arena, (size_t)hr_dim * sizeof(float));
        float *hr_up   = arena_alloc(&ctx->arena, (size_t)hr_dim * sizeof(float));
        float *hr_mid  = arena_alloc(&ctx->arena, (size_t)hr_dim * sizeof(float));

        snprintf(name, sizeof(name), "layers.%d.ffn.shared_experts.gate_proj.weight", layer_idx);
        snprintf(sname, sizeof(sname), "layers.%d.ffn.shared_experts.gate_proj.scales", layer_idx);
        snprintf(bname, sizeof(bname), "layers.%d.ffn.shared_experts.gate_proj.biases", layer_idx);
        q4_matmul(ctx, hr_gate, ctx->norm_out, name, sname, bname, hr_dim, H, 64);

        snprintf(name, sizeof(name), "layers.%d.ffn.shared_experts.up_proj.weight", layer_idx);
        snprintf(sname, sizeof(sname), "layers.%d.ffn.shared_experts.up_proj.scales", layer_idx);
        snprintf(bname, sizeof(bname), "layers.%d.ffn.shared_experts.up_proj.biases", layer_idx);
        q4_matmul(ctx, hr_up, ctx->norm_out, name, sname, bname, hr_dim, H, 64);

        // Shared expert has swiglu_limit=0 (no clamp) per reference model.py:628.
        cpu_silu_mul(hr_mid, hr_gate, hr_up, hr_dim);

        snprintf(name, sizeof(name), "layers.%d.ffn.shared_experts.down_proj.weight", layer_idx);
        snprintf(sname, sizeof(sname), "layers.%d.ffn.shared_experts.down_proj.scales", layer_idx);
        snprintf(bname, sizeof(bname), "layers.%d.ffn.shared_experts.down_proj.biases", layer_idx);
        q4_matmul(ctx, ctx->sublayer_out, hr_mid, name, sname, bname, H, hr_dim, 64);

        // Routed experts (CPU path — only 3 hash layers, not perf-critical).
        // V4 packed expert layout per projection: weight[hr_dim,K/8] U32 + scales[hr_dim,K/32] U8.
        // Format is MXFP4 (FP4 E2M1 nibbles + E8M0 group exponent), group_size=32, no biases.
        hr_dim = cfg->moe_intermediate_size;
        int hr_gs = 32;
        size_t hr_w_size  = (size_t)hr_dim * (size_t)(H / 8) * 4;
        size_t hr_s_size  = (size_t)hr_dim * (size_t)(H / hr_gs) * 1;

        for (int k = 0; k < K_active; k++) {
            int expert_idx = top_indices[k];
            float weight = top_weights[k];
            size_t stride;
            const void *expert_data = model_get_expert(ctx->model, layer_idx,
                                                        expert_idx, &stride);
            if (!expert_data) continue;

            const char *ptr = expert_data;
            const void *gw = ptr; ptr += hr_w_size;
            const void *gs = ptr; ptr += hr_s_size;

            float *r_gate = arena_alloc(&ctx->arena, (size_t)hr_dim * sizeof(float));
            dequant_matmul_mxfp4(r_gate, gw, gs, ctx->norm_out, hr_dim, H, hr_gs);

            const void *uw = ptr; ptr += hr_w_size;
            const void *us = ptr; ptr += hr_s_size;

            float *r_up = arena_alloc(&ctx->arena, (size_t)hr_dim * sizeof(float));
            dequant_matmul_mxfp4(r_up, uw, us, ctx->norm_out, hr_dim, H, hr_gs);

            // Routed expert: gate clamped MAX-only, up symmetric (model.py:601).
            for (int i = 0; i < hr_dim; i++) {
                if (r_gate[i] > 10.0f) r_gate[i] = 10.0f;
                if (r_up[i]   >  10.0f) r_up[i]   =  10.0f;
                if (r_up[i]   < -10.0f) r_up[i]   = -10.0f;
            }

            float *r_mid = arena_alloc(&ctx->arena, (size_t)hr_dim * sizeof(float));
            cpu_silu_mul(r_mid, r_gate, r_up, hr_dim);

            // down_proj: [H, hr_dim/8] U32 + [H, hr_dim/32] U8 scales.
            size_t hr_dw_size = (size_t)H * (size_t)(hr_dim / 8) * 4;
            size_t hr_ds_size = (size_t)H * (size_t)(hr_dim / hr_gs) * 1;

            const void *dw = ptr; ptr += hr_dw_size;
            const void *ds = ptr; ptr += hr_ds_size;

            float *result = arena_alloc(&ctx->arena, (size_t)H * sizeof(float));
            dequant_matmul_mxfp4(result, dw, ds, r_mid, H, hr_dim, hr_gs);

#ifdef PLATFORM_MACOS
            vDSP_vsma(result, 1, &weight, ctx->sublayer_out, 1, ctx->sublayer_out, 1, (vDSP_Length)H);
#else
            for (int i = 0; i < H; i++)
                ctx->sublayer_out[i] += weight * result[i];
#endif
        }
    }

    // HC post for FFN
    v4_hc_post(ctx, ctx->sublayer_out, ffn_post, ffn_comb);

    _t1 = timer_now_ns(); _v4_acc_routed += timer_elapsed_ms(_t0, _t1);

    arena_reset(&ctx->arena);
}

void v4_compute_logits(V4InferenceContext *v4, float *out_logits) {
    const ModelConfig *cfg = v4->cfg;
    int H = cfg->hidden_size;
    int M = cfg->v4.hc_mult;

    // Final HC: reduce [M*H] → [H]
    // Uses model.final_hc.hc_fn, hc_scale, hc_base
    // But final_hc only has pre (no post/comb — it's terminal)
    char wn[128], sn[128], bn[128];
    snprintf(wn, sizeof(wn), "hc_head.fn.weight");
    snprintf(sn, sizeof(sn), "hc_head.fn.scales");
    snprintf(bn, sizeof(bn), "hc_head.fn.biases");

    int MH = M * H;

    // RMSNorm the flattened HC state
    float ss = 0.0f;
    float eps = cfg->v4.hc_eps;
#ifdef PLATFORM_MACOS
    vDSP_svesq(v4->hc_state, 1, &ss, (vDSP_Length)MH);
#else
    for (int i = 0; i < MH; i++) ss += v4->hc_state[i] * v4->hc_state[i];
#endif
    float inv_rms = 1.0f / sqrtf(ss / (float)MH + eps);

    float *normed = arena_alloc(&v4->arena, (size_t)MH * sizeof(float));
#ifdef PLATFORM_MACOS
    vDSP_vsmul(v4->hc_state, 1, &inv_rms, normed, 1, (vDSP_Length)MH);
#else
    for (int i = 0; i < MH; i++) normed[i] = v4->hc_state[i] * inv_rms;
#endif

    // Project to get pre weights [M]
    float pre_proj[4];

    size_t ws;
    const void *w = model_get_weight(v4->model, wn, &ws);
    if (w) {
        q4_matmul(v4, pre_proj, normed, wn, sn, bn, M, MH, 64);
    } else {
        const void *hc_fn = model_get_weight(v4->model, "hc_head.fn", &ws);
        size_t expected_f32 = (size_t)M * (size_t)MH * sizeof(float);
        if (hc_fn && ws == expected_f32) {
            const float *W = hc_fn;
            for (int i = 0; i < M; i++) {
                float dot = 0.0f;
                for (int j = 0; j < MH; j++)
                    dot += W[i * MH + j] * normed[j];
                pre_proj[i] = dot;
            }
        } else if (hc_fn) {
            const uint16_t *W = hc_fn;
            for (int i = 0; i < M; i++) {
                float dot = 0.0f;
                for (int j = 0; j < MH; j++)
                    dot += bf16_to_f32(W[i * MH + j]) * normed[j];
                pre_proj[i] = dot;
            }
        } else {
            for (int i = 0; i < M; i++) pre_proj[i] = 0.0f;
        }
    }

    // Apply scale and base for final HC
    size_t scale_sz, base_sz;
    const void *scale_data = model_get_weight(v4->model, "hc_head.scale", &scale_sz);
    const void *base_data = model_get_weight(v4->model, "hc_head.base", &base_sz);

    if (scale_data && base_data) {
        float scale;
        if (scale_sz >= sizeof(float))
            memcpy(&scale, scale_data, sizeof(float));
        else
            scale = bf16_to_f32(*(const uint16_t *)scale_data);

        for (int i = 0; i < M; i++) {
            float b;
            if (base_sz >= (size_t)(M * sizeof(float))) {
                memcpy(&b, (const char *)base_data + i * sizeof(float), sizeof(float));
            } else {
                b = bf16_to_f32(((const uint16_t *)base_data)[i]);
            }
            pre_proj[i] = pre_proj[i] * scale + b;
        }
    }

    // Reference (DeepSeek-V4-Flash inference/model.py::hc_head):
    //   pre = sigmoid(mixes * scale + base) + hc_eps
    const float hc_eps = cfg->v4.hc_eps;
    for (int i = 0; i < M; i++)
        pre_proj[i] = 1.0f / (1.0f + expf(-pre_proj[i])) + hc_eps;

    // Weighted sum of copies → hidden [H]
    float *hidden = v4->sublayer_in;  // reuse buffer
    memset(hidden, 0, (size_t)H * sizeof(float));
    for (int m = 0; m < M; m++) {
        const float *copy = v4->hc_state + m * H;
#ifdef PLATFORM_MACOS
        vDSP_vsma(copy, 1, &pre_proj[m], hidden, 1, hidden, 1, (vDSP_Length)H);
#else
        for (int i = 0; i < H; i++)
            hidden[i] += pre_proj[m] * copy[i];
#endif
    }

    // Final RMSNorm
    size_t norm_sz;
    const void *norm_bf16 = model_get_weight(v4->model, "norm.weight", &norm_sz);
    if (norm_bf16) {
        float *norm_w = arena_alloc(&v4->arena, (size_t)H * sizeof(float));
        bf16_to_float_vec(norm_w, norm_bf16, H);
        float *tmp = arena_alloc(&v4->arena, (size_t)H * sizeof(float));
        cpu_rmsnorm(tmp, hidden, norm_w, H, cfg->rms_norm_eps);
        memcpy(hidden, tmp, (size_t)H * sizeof(float));
    }

    // lm_head projection (route through GPU logits buffer when available)
    q4_matmul(v4, v4->logits_buf, hidden,
              "lm_head.weight", "lm_head.scales", "lm_head.biases",
              cfg->vocab_size, H, 64);
    memcpy(out_logits, v4->logits_buf, (size_t)cfg->vocab_size * sizeof(float));

    arena_reset(&v4->arena);
}

void v4_inference_free(V4InferenceContext *ctx) {
    if (!ctx) return;

#ifdef PLATFORM_MACOS
    if (ctx->use_gpu) {
        ctx->sublayer_in = NULL;
        ctx->norm_out = NULL;
        ctx->logits_buf = NULL;
        ctx->gate_logits = NULL;
        metal_free_buffer(ctx->gpu_hidden);
        metal_free_buffer(ctx->gpu_norm_out);
        metal_free_buffer(ctx->gpu_out);
        metal_free_buffer(ctx->gpu_logits);
        metal_free_buffer(ctx->gpu_gate_out);
        metal_free_buffer(ctx->gpu_up_out);
        metal_free_buffer(ctx->gpu_ffn_mid);
        metal_free_buffer(ctx->gpu_expert_result);
        metal_free_buffer(ctx->gpu_gate_logits);
        for (int i = 0; i < ctx->num_expert_slots; i++) {
            metal_free_buffer(ctx->gpu_ffn_mid_slots[i]);
            metal_free_buffer(ctx->gpu_result_slots[i]);
        }
        if (ctx->expert_io) {
            for (int i = 0; i < ctx->num_expert_slots; i++) {
                if (ctx->gpu_expert_staging[i])
                    metal_free_buffer(ctx->gpu_expert_staging[i]);
            }
            expert_io_free(ctx->expert_io);
        }
    }
#endif

    free(ctx->hc_state);
    free(ctx->hc_copies);
    free(ctx->sublayer_in);
    free(ctx->sublayer_out);
    free(ctx->norm_out);
    free(ctx->gate_logits);
    free(ctx->expert_out);
    free(ctx->expert_buf);
    free(ctx->logits_buf);
    v4_compressor_free(ctx->compressor);
    arena_destroy(&ctx->arena);
    free(ctx);
}

void v4_reset_state(V4InferenceContext *v4) {
    if (!v4) return;
    v4_compressor_reset(v4->compressor);
    // hc_state is reinitialized at the start of every token via v4_init_hc_state,
    // so it doesn't need an explicit reset here.
}
