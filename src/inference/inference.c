#define _POSIX_C_SOURCE 200809L

#include "inference/inference.h"
#include "inference/attention.h"
#include "inference/kv_cache.h"
#include "inference/sampler.h"
#include "inference/dequant.h"
#include "config/config.h"
#include "model/mmap_pool.h"
#include "util/log.h"
#include "util/timer.h"
#include "util/arena.h"

#ifdef PLATFORM_MACOS
#include "compute/kernels.h"
#include "compute/metal_context.h"
#include "model/expert_io.h"
#endif

#include <alloca.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Scratch buffers for a single forward pass
typedef struct {
    float *hidden;       // [hidden_size]
    float *residual;     // [hidden_size]
    float *norm_out;     // [hidden_size]
    float *q;            // [num_heads * head_dim]
    float *k;            // [num_kv_heads * head_dim]
    float *v;            // [num_kv_heads * head_dim]
    float *attn_out;     // [num_heads * head_dim]
    float *attn_scores;  // [num_heads * max_seq]
    float *gate_logits;  // [num_experts]
    float *gate_probs;   // [num_experts]
    float *expert_out;   // [hidden_size]
    float *expert_buf;   // [moe_intermediate * 3]
    float *logits;       // [vocab_size]
} ScratchBuffers;

struct InferenceContext {
    Model          *model;
    InferenceCache *cache;
    ScratchBuffers  scratch;
    Arena           arena;
    int             position;  // current token position
    int             kv_idx;    // current KV layer index (for SWA layers)
    int             dn_idx;    // current DeltaNet layer index
    AttentionGPU   *attn_gpu;  // reusable GPU buffers for attention projections

#ifdef PLATFORM_MACOS
    // GPU scratch buffers (Metal handles — CPU-accessible via unified memory)
    MetalContext   *metal;
    void           *shared_buf;    // Metal buffer wrapping all shared weights
    void           *gpu_hidden;    // Metal buffer for hidden state
    void           *gpu_norm_out;  // Metal buffer for norm output
    void           *gpu_out;       // Metal buffer for matmul output (reusable)
    void           *gpu_logits;    // Metal buffer for logits
    // Expert GPU scratch (shared expert uses slot 0)
    void           *gpu_gate_out;  // Metal buffer for expert gate_proj output
    void           *gpu_up_out;    // Metal buffer for expert up_proj output
    void           *gpu_ffn_mid;   // Metal buffer for SiLU(gate)*up intermediate
    void           *gpu_expert_result; // Metal buffer for expert down_proj output
    float          *cpu_gate_out;  // CPU pointer into gpu_gate_out
    float          *cpu_up_out;
    float          *cpu_ffn_mid;
    float          *cpu_expert_result;
    // Per-expert GPU buffers for batched dispatch (all experts in one commit)
    int             num_expert_slots;
    void           *gpu_ffn_mid_slots[16];
    void           *gpu_result_slots[16];
    float          *cpu_ffn_mid_slots[16];
    float          *cpu_result_slots[16];
    // Expert staging buffers — pread directly into Metal unified memory
    void           *gpu_expert_staging[16];  // Metal buffer handles
    void           *cpu_expert_staging[16];  // CPU pointers into staging
    size_t          expert_staging_size;      // bytes per staging buffer
    ExpertIO       *expert_io;               // GCD parallel pread subsystem
    // Deferred expert completion from previous layer
    void           *prev_expert_signal;      // dispatch_semaphore from deferred batch
    float           prev_expert_weights[16]; // weights for pending accumulation
    int             prev_expert_count;       // number of pending experts
    bool            use_gpu;
#endif
};

static void alloc_scratch(ScratchBuffers *s, const ModelConfig *cfg, int max_seq) {
    s->hidden      = calloc((size_t)cfg->hidden_size, sizeof(float));
    s->residual    = calloc((size_t)cfg->hidden_size, sizeof(float));
    s->norm_out    = calloc((size_t)cfg->hidden_size, sizeof(float));
    s->q           = calloc((size_t)cfg->num_attention_heads * (size_t)cfg->head_dim, sizeof(float));
    s->k           = calloc((size_t)cfg->num_key_value_heads * (size_t)cfg->head_dim, sizeof(float));
    s->v           = calloc((size_t)cfg->num_key_value_heads * (size_t)cfg->head_dim, sizeof(float));
    s->attn_out    = calloc((size_t)cfg->num_attention_heads * (size_t)cfg->head_dim, sizeof(float));
    s->attn_scores = calloc((size_t)cfg->num_attention_heads * (size_t)max_seq, sizeof(float));
    s->gate_logits = calloc((size_t)cfg->num_experts, sizeof(float));
    s->gate_probs  = calloc((size_t)cfg->num_experts, sizeof(float));
    s->expert_out  = calloc((size_t)cfg->hidden_size, sizeof(float));
    s->expert_buf  = calloc((size_t)cfg->moe_intermediate_size * 3, sizeof(float));
    s->logits      = calloc((size_t)cfg->vocab_size, sizeof(float));
}

static void free_scratch(ScratchBuffers *s) {
    free(s->hidden);
    free(s->residual);
    free(s->norm_out);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->attn_out);
    free(s->attn_scores);
    free(s->gate_logits);
    free(s->gate_probs);
    free(s->expert_out);
    free(s->expert_buf);
    free(s->logits);
}

InferenceContext *inference_create(Model *model) {
    InferenceContext *ctx = calloc(1, sizeof(InferenceContext));
    ctx->model = model;

    const ModelConfig *cfg = model_config(model);

    // For v1: limit context to 8K to keep memory reasonable
    int max_seq = 8192;
    if (max_seq > cfg->max_position_embeddings)
        max_seq = cfg->max_position_embeddings;

    ctx->cache = cache_create(cfg, max_seq);
    alloc_scratch(&ctx->scratch, cfg, max_seq);
    ctx->arena = arena_create(64 * 1024 * 1024); // 64 MB scratch arena
    ctx->attn_gpu = attention_gpu_create(model, cfg);

#ifdef PLATFORM_MACOS
    ctx->metal = model_get_metal(model);
    ctx->shared_buf = model_get_metal_shared_buf(model);
    if (ctx->metal && ctx->shared_buf) {
        int H = cfg->hidden_size;
        int max_out = cfg->vocab_size > H ? cfg->vocab_size : H;
        ctx->gpu_hidden   = metal_alloc_buffer(ctx->metal, (size_t)H * sizeof(float));
        ctx->gpu_norm_out  = metal_alloc_buffer(ctx->metal, (size_t)H * sizeof(float));
        ctx->gpu_out       = metal_alloc_buffer(ctx->metal, (size_t)max_out * sizeof(float));
        ctx->gpu_logits    = metal_alloc_buffer(ctx->metal, (size_t)cfg->vocab_size * sizeof(float));

        // Expert scratch buffers — size for the larger of moe_intermediate and shared_expert
        int expert_dim = cfg->moe_intermediate_size;
        if (cfg->shared_expert_intermediate_size > expert_dim)
            expert_dim = cfg->shared_expert_intermediate_size;
        ctx->gpu_gate_out      = metal_alloc_buffer(ctx->metal, (size_t)expert_dim * sizeof(float));
        ctx->gpu_up_out        = metal_alloc_buffer(ctx->metal, (size_t)expert_dim * sizeof(float));
        ctx->gpu_ffn_mid       = metal_alloc_buffer(ctx->metal, (size_t)expert_dim * sizeof(float));
        ctx->gpu_expert_result = metal_alloc_buffer(ctx->metal, (size_t)H * sizeof(float));

        if (ctx->gpu_hidden && ctx->gpu_norm_out && ctx->gpu_out && ctx->gpu_logits &&
            ctx->gpu_gate_out && ctx->gpu_up_out && ctx->gpu_ffn_mid && ctx->gpu_expert_result) {
            // Point scratch arrays at the GPU buffer contents (unified memory)
            free(ctx->scratch.hidden);
            free(ctx->scratch.norm_out);
            free(ctx->scratch.logits);
            ctx->scratch.hidden   = metal_buffer_contents(ctx->gpu_hidden);
            ctx->scratch.norm_out = metal_buffer_contents(ctx->gpu_norm_out);
            ctx->scratch.logits   = metal_buffer_contents(ctx->gpu_logits);
            ctx->cpu_gate_out      = metal_buffer_contents(ctx->gpu_gate_out);
            ctx->cpu_up_out        = metal_buffer_contents(ctx->gpu_up_out);
            ctx->cpu_ffn_mid       = metal_buffer_contents(ctx->gpu_ffn_mid);
            ctx->cpu_expert_result = metal_buffer_contents(ctx->gpu_expert_result);
            ctx->use_gpu = true;

            // Per-expert GPU buffers for batched dispatch
            int K_max = cfg->num_experts_per_tok;
            if (K_max > 16) K_max = 16;
            bool slots_ok = true;
            for (int i = 0; i < K_max; i++) {
                ctx->gpu_ffn_mid_slots[i] = metal_alloc_buffer(ctx->metal,
                    (size_t)expert_dim * sizeof(float));
                ctx->gpu_result_slots[i] = metal_alloc_buffer(ctx->metal,
                    (size_t)H * sizeof(float));
                if (ctx->gpu_ffn_mid_slots[i] && ctx->gpu_result_slots[i]) {
                    ctx->cpu_ffn_mid_slots[i] = metal_buffer_contents(ctx->gpu_ffn_mid_slots[i]);
                    ctx->cpu_result_slots[i] = metal_buffer_contents(ctx->gpu_result_slots[i]);
                } else {
                    slots_ok = false;
                    break;
                }
            }
            if (slots_ok) {
                ctx->num_expert_slots = K_max;
            }

            // Expert staging buffers — pread directly into Metal unified memory
            // Each staging buffer holds one full expert's weight data
            size_t max_stride = model_get_expert_stride(model, 0);
            if (max_stride > 0) {
                bool staging_ok = true;
                for (int i = 0; i < K_max; i++) {
                    ctx->gpu_expert_staging[i] = metal_alloc_buffer(ctx->metal, max_stride);
                    if (ctx->gpu_expert_staging[i]) {
                        ctx->cpu_expert_staging[i] = metal_buffer_contents(
                            ctx->gpu_expert_staging[i]);
                    } else {
                        staging_ok = false;
                        break;
                    }
                }
                if (staging_ok) {
                    ctx->expert_staging_size = max_stride;
                    ctx->expert_io = expert_io_create(K_max);
                    LOG_INFO("inference: expert staging allocated (%d x %zu MB)",
                             K_max, max_stride / (1024 * 1024));
                }
            }

            LOG_INFO("inference: GPU acceleration enabled (%d expert slots)", ctx->num_expert_slots);
        }
    }
#endif

    LOG_INFO("inference: context created (max_seq=%d)", max_seq);
    return ctx;
}

// Complete deferred expert accumulation from previous layer
#ifdef PLATFORM_MACOS
static void complete_deferred_experts(InferenceContext *ctx) {
    if (!ctx->prev_expert_signal) return;

    kernel_wait_deferred(ctx->prev_expert_signal);
    ctx->prev_expert_signal = NULL;

    const ModelConfig *cfg = model_config(ctx->model);
    int H = cfg->hidden_size;
    ScratchBuffers *s = &ctx->scratch;

    for (int k = 0; k < ctx->prev_expert_count; k++) {
        float weight = ctx->prev_expert_weights[k];
        float *result = ctx->cpu_result_slots[k];
        for (int i = 0; i < H; i++) {
            s->hidden[i] += weight * result[i];
        }
    }
    ctx->prev_expert_count = 0;
}
#endif

// --- CPU compute primitives ---

static void cpu_rmsnorm(float *out, const float *x, const float *weight,
                        int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] / rms) * weight[i];
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

__attribute__((unused))
static void cpu_rope(float *x, int head_dim, int rotary_dim,
                     int position, float theta_base, int num_heads) {
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)rotary_dim);
            float angle = (float)position * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            int idx = h * head_dim + i * 2;
            float x0 = x[idx];
            float x1 = x[idx + 1];
            x[idx]     = x0 * cos_a - x1 * sin_a;
            x[idx + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

__attribute__((unused))
static void cpu_attention(float *out, const float *q, const float *k_cache,
                          const float *v_cache, int num_heads, int num_kv_heads,
                          int head_dim, int seq_len) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_group = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / kv_group;
        const float *qi = q + h * head_dim;

        // Compute scores
        float *scores = alloca((size_t)seq_len * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            const float *ki = k_cache + t * num_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qi[d] * ki[d];
            scores[t] = dot * scale;
        }

        // Softmax
        cpu_softmax(scores, seq_len);

        // Weighted sum of values
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                sum += scores[t] * v_cache[t * num_kv_heads * head_dim + kv_h * head_dim + d];
            }
            out[h * head_dim + d] = sum;
        }
    }
}

// Helper: quantized matmul using weight name lookup
// Uses GPU when available, falls back to CPU.
static void q4_matmul(InferenceContext *ctx, float *out, const float *x,
                      const char *w_name, const char *s_name, const char *b_name,
                      int M, int K, int group_size) {
#ifdef PLATFORM_MACOS
    if (ctx->use_gpu) {
        long w_off = model_get_weight_offset(ctx->model, w_name);
        long s_off = model_get_weight_offset(ctx->model, s_name);
        long b_off = model_get_weight_offset(ctx->model, b_name);

        if (w_off >= 0 && s_off >= 0 && b_off >= 0) {
            // Determine which GPU buffer the input x lives in
            void *x_buf = NULL;
            if (x == ctx->scratch.hidden)   x_buf = ctx->gpu_hidden;
            if (x == ctx->scratch.norm_out) x_buf = ctx->gpu_norm_out;

            // Determine which GPU buffer the output goes to
            void *out_buf = NULL;
            if (out == ctx->scratch.hidden)   out_buf = ctx->gpu_hidden;
            if (out == ctx->scratch.norm_out) out_buf = ctx->gpu_norm_out;
            if (out == ctx->scratch.logits)   out_buf = ctx->gpu_logits;

            if (x_buf && out_buf) {
                kernel_matmul_q4_fma_offsets(ctx->metal, ctx->shared_buf,
                                             (size_t)w_off, (size_t)s_off, (size_t)b_off,
                                             x_buf, out_buf,
                                             (uint32_t)M, (uint32_t)K, (uint32_t)group_size);
                return;
            }
            // Fallthrough to CPU if buffers don't match
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

// Convert BF16 norm weight to float
static void bf16_to_float_vec(float *out, const void *bf16_data, int n) {
    const uint16_t *src = bf16_data;
    for (int i = 0; i < n; i++) {
        out[i] = bf16_to_f32(src[i]);
    }
}

// --- Forward pass ---

static void forward_layer(InferenceContext *ctx, int layer_idx) {
    const ModelConfig *cfg = model_config(ctx->model);
    ScratchBuffers *s = &ctx->scratch;
    int H = cfg->hidden_size;

    // Complete deferred expert accumulation from previous layer.
    // This blocks until the previous layer's expert GPU work finishes.
    // The current layer's attention can't start until we have the correct hidden state.
#ifdef PLATFORM_MACOS
    complete_deferred_experts(ctx);
#endif

    // Get layer norm weight (BF16 → float)
    char name[128];
    snprintf(name, sizeof(name), "layers.%d.input_layernorm.weight", layer_idx);
    size_t norm_size;
    const void *norm_bf16 = model_get_weight(ctx->model, name, &norm_size);
    if (!norm_bf16) return; // weights not loaded

    float *norm_weight = arena_alloc(&ctx->arena, (size_t)H * sizeof(float));
    bf16_to_float_vec(norm_weight, norm_bf16, H);

    // 1. Pre-attention RMSNorm
    memcpy(s->residual, s->hidden, (size_t)H * sizeof(float));
    cpu_rmsnorm(s->norm_out, s->hidden, norm_weight, H, cfg->rms_norm_eps);

    // 2-9. Attention (SWA or DeltaNet depending on layer type)
    float *attn_result = arena_alloc(&ctx->arena, (size_t)H * sizeof(float));

    // Set GPU input handle so attention can read norm_out from GPU directly
#ifdef PLATFORM_MACOS
    if (ctx->use_gpu)
        attention_gpu_set_input(ctx->attn_gpu, ctx->gpu_norm_out, s->norm_out);
#endif

    if (cfg->layer_types && cfg->layer_types[layer_idx] == LAYER_FULL_ATTENTION) {
        attention_swa_forward(attn_result, s->norm_out, ctx->model, cfg,
                             ctx->cache, ctx->attn_gpu,
                             layer_idx, ctx->kv_idx, ctx->position);
        ctx->kv_idx++;
    } else {
        attention_deltanet_forward(attn_result, s->norm_out, ctx->model, cfg,
                                   ctx->cache, ctx->attn_gpu,
                                   layer_idx, ctx->dn_idx, ctx->position);
        ctx->dn_idx++;
    }

    // Add attention output to residual
    for (int i = 0; i < H; i++) {
        s->hidden[i] = s->residual[i] + attn_result[i];
    }

    // 10. Post-attention layernorm
    // Save post-attention hidden as new residual for MLP block
    memcpy(s->residual, s->hidden, (size_t)H * sizeof(float));
    snprintf(name, sizeof(name), "layers.%d.post_attention_layernorm.weight", layer_idx);
    const void *post_norm_bf16 = model_get_weight(ctx->model, name, &norm_size);
    if (post_norm_bf16) {
        bf16_to_float_vec(norm_weight, post_norm_bf16, H);
        cpu_rmsnorm(s->norm_out, s->hidden, norm_weight, H, cfg->rms_norm_eps);
    }

    // 11. MoE gate: gate_logits = norm_out @ gate_weight
    int num_experts = cfg->num_experts;
    int K_active = cfg->num_experts_per_tok;

    snprintf(name, sizeof(name), "layers.%d.mlp.gate.weight", layer_idx);
    char sname[128], bname[128];
    snprintf(sname, sizeof(sname), "layers.%d.mlp.gate.scales", layer_idx);
    snprintf(bname, sizeof(bname), "layers.%d.mlp.gate.biases", layer_idx);

    q4_matmul(ctx, s->gate_logits, s->norm_out, name, sname, bname,
              num_experts, H, 64);

    // 12. Top-K selection
    memcpy(s->gate_probs, s->gate_logits, (size_t)num_experts * sizeof(float));
    cpu_softmax(s->gate_probs, num_experts);

    // Simple top-K selection
    int top_indices[16];
    float top_weights[16];
    for (int k = 0; k < K_active && k < 16; k++) {
        float best = -1e30f;
        int best_idx = 0;
        for (int e = 0; e < num_experts; e++) {
            bool skip = false;
            for (int j = 0; j < k; j++) {
                if (top_indices[j] == e) { skip = true; break; }
            }
            if (skip) continue;
            if (s->gate_probs[e] > best) {
                best = s->gate_probs[e];
                best_idx = e;
            }
        }
        top_indices[k] = best_idx;
        top_weights[k] = best;
    }

    // Renormalize weights
    float wsum = 0.0f;
    for (int k = 0; k < K_active; k++) wsum += top_weights[k];
    if (wsum > 0.0f) {
        for (int k = 0; k < K_active; k++) top_weights[k] /= wsum;
    }

    // Issue parallel pread() for selected experts into staging buffers.
    // This runs concurrently with the shared expert FFN below.
    // Only use pread for large experts (>2MB) where mmap page faults are costly.
    // Small experts (35B: ~1MB) are fast via mmap since they fit in page cache.
    size_t expert_stride = model_get_expert_stride(ctx->model, layer_idx);
    int expert_fd = model_get_expert_fd(ctx->model, layer_idx);
#ifdef PLATFORM_MACOS
    bool pread_experts = false;
    if (ctx->expert_io && expert_fd >= 0 && expert_stride > (2 * 1024 * 1024) &&
        K_active <= ctx->num_expert_slots &&
        expert_stride <= ctx->expert_staging_size) {
        size_t offsets[16];
        size_t sizes[16];
        void  *dests[16];
        for (int k = 0; k < K_active; k++) {
            offsets[k] = (size_t)top_indices[k] * expert_stride;
            sizes[k]   = expert_stride;
            dests[k]   = ctx->cpu_expert_staging[k];
        }
        expert_io_fetch(ctx->expert_io, expert_fd, offsets, sizes, dests, K_active);
        pread_experts = true;
    } else
#endif
    {
        // Fallback: madvise prefetch for mmap path
        for (int k = 0; k < K_active; k++) {
            size_t stride;
            const void *expert_data = model_get_expert(ctx->model, layer_idx,
                                                        top_indices[k], &stride);
            if (expert_data) {
                mmap_pool_prefetch((void *)expert_data, stride);
            }
        }
    }

    // 13. Shared expert FFN (runs while pread I/O is in flight)
    int moe_dim = cfg->shared_expert_intermediate_size;
    float *gate_out = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
    float *up_out   = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
    float *ffn_mid  = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
    bool shared_on_gpu = false;

#ifdef PLATFORM_MACOS
    if (ctx->use_gpu) {
        // GPU path: fused gate+up+SwiGLU + batched down_proj (1 commit total)
        char gw[128], gs_n[128], gb_n[128], uw[128], us_n[128], ub_n[128];
        char dw[128], ds_n[128], db_n[128];
        snprintf(gw, sizeof(gw), "layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
        snprintf(gs_n, sizeof(gs_n), "layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
        snprintf(gb_n, sizeof(gb_n), "layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
        snprintf(uw, sizeof(uw), "layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
        snprintf(us_n, sizeof(us_n), "layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
        snprintf(ub_n, sizeof(ub_n), "layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
        snprintf(dw, sizeof(dw), "layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
        snprintf(ds_n, sizeof(ds_n), "layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
        snprintf(db_n, sizeof(db_n), "layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);

        long gw_off = model_get_weight_offset(ctx->model, gw);
        long gs_off = model_get_weight_offset(ctx->model, gs_n);
        long gb_off = model_get_weight_offset(ctx->model, gb_n);
        long uw_off = model_get_weight_offset(ctx->model, uw);
        long us_off = model_get_weight_offset(ctx->model, us_n);
        long ub_off = model_get_weight_offset(ctx->model, ub_n);
        long dw_off = model_get_weight_offset(ctx->model, dw);
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
            memcpy(s->expert_out, ctx->cpu_expert_result, (size_t)H * sizeof(float));
            shared_on_gpu = true;
        }
    }
#endif

    if (!shared_on_gpu) {
        snprintf(name, sizeof(name), "layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
        snprintf(sname, sizeof(sname), "layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
        snprintf(bname, sizeof(bname), "layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
        q4_matmul(ctx, gate_out, s->norm_out, name, sname, bname, moe_dim, H, 64);

        snprintf(name, sizeof(name), "layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
        snprintf(sname, sizeof(sname), "layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
        snprintf(bname, sizeof(bname), "layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
        q4_matmul(ctx, up_out, s->norm_out, name, sname, bname, moe_dim, H, 64);

        cpu_silu_mul(ffn_mid, gate_out, up_out, moe_dim);

        snprintf(name, sizeof(name), "layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
        snprintf(sname, sizeof(sname), "layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
        snprintf(bname, sizeof(bname), "layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);
        q4_matmul(ctx, s->expert_out, ffn_mid, name, sname, bname, H, moe_dim, 64);
    }

    // Apply shared_expert_gate: sigmoid gate that weights the shared expert
    float shared_gate_val = 1.0f;
    snprintf(name, sizeof(name), "layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    snprintf(sname, sizeof(sname), "layers.%d.mlp.shared_expert_gate.scales", layer_idx);
    snprintf(bname, sizeof(bname), "layers.%d.mlp.shared_expert_gate.biases", layer_idx);
    float gate_scalar = 0.0f;
    q4_matmul(ctx, &gate_scalar, s->norm_out, name, sname, bname, 1, H, 64);
    shared_gate_val = 1.0f / (1.0f + expf(-gate_scalar)); // sigmoid

    // Add gated shared expert output to residual
    for (int i = 0; i < H; i++) s->hidden[i] = s->residual[i] + shared_gate_val * s->expert_out[i];

    // 14. Routed experts
    moe_dim = cfg->moe_intermediate_size;
    if (moe_dim != cfg->shared_expert_intermediate_size) {
        gate_out = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
        up_out   = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
        ffn_mid  = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
    }

    // Expert layout: gate_proj(w,s,b), up_proj(w,s,b), down_proj(w,s,b)
    // Each projection: weight[moe_dim, H/8] + scales[moe_dim, G] + biases[moe_dim, G]
    int group_size = 64;
    int num_groups_h = H / group_size;
    size_t w_size = (size_t)moe_dim * (size_t)(H / 8) * 4;      // U32
    size_t s_size = (size_t)moe_dim * (size_t)num_groups_h * 2;  // BF16
    size_t b_size = s_size;
    size_t proj_size = w_size + s_size + b_size;  // one projection

    int down_groups = moe_dim / group_size;
    size_t dw_size = (size_t)H * (size_t)(moe_dim / 8) * 4;
    size_t ds_size = (size_t)H * (size_t)down_groups * 2;

#ifdef PLATFORM_MACOS
    if (ctx->use_gpu && K_active <= ctx->num_expert_slots && pread_experts) {
        // pread-based path: expert data is in staging buffers (Metal unified memory)
        // Wait for parallel pread() to complete
        expert_io_wait(ctx->expert_io);

        void *batch = kernel_begin_batch(ctx->metal);
        int batch_count = 0;

        for (int k = 0; k < K_active; k++) {
            // Offsets within the staging buffer (expert data starts at 0)
            size_t down_base = proj_size * 2;

            // Fused gate+up+SwiGLU using staging buffer as weight source
            kernel_batch_fused_gate_up_swiglu_offsets(batch,
                                                      ctx->gpu_expert_staging[k],
                                                      0,
                                                      w_size,
                                                      w_size + s_size,
                                                      proj_size,
                                                      proj_size + w_size,
                                                      proj_size + w_size + s_size,
                                                      ctx->gpu_norm_out,
                                                      ctx->gpu_ffn_mid_slots[k],
                                                      (uint32_t)moe_dim, (uint32_t)H,
                                                      (uint32_t)group_size);

            // down_proj using staging buffer
            kernel_batch_q4_fma_offsets(batch,
                                        ctx->gpu_expert_staging[k],
                                        down_base,
                                        down_base + dw_size,
                                        down_base + dw_size + ds_size,
                                        ctx->gpu_ffn_mid_slots[k],
                                        ctx->gpu_result_slots[k],
                                        (uint32_t)H, (uint32_t)moe_dim,
                                        (uint32_t)group_size);
            batch_count++;
        }

        ctx->prev_expert_signal = kernel_end_batch_deferred(batch);
        for (int k = 0; k < batch_count; k++) {
            ctx->prev_expert_weights[k] = top_weights[k];
        }
        ctx->prev_expert_count = batch_count;
    } else if (ctx->use_gpu && K_active <= ctx->num_expert_slots) {
        // mmap-based GPU path (fallback when pread not available)
        void *expert_buf = model_get_expert_metal_buf(ctx->model, layer_idx);
        if (expert_buf) {
            void *batch = kernel_begin_batch(ctx->metal);
            int batch_count = 0;

            for (int k = 0; k < K_active; k++) {
                int expert_idx = top_indices[k];
                size_t stride;
                const void *expert_data = model_get_expert(ctx->model, layer_idx,
                                                            expert_idx, &stride);
                if (!expert_data) continue;

                size_t base = (size_t)expert_idx * stride;
                size_t down_base = base + proj_size * 2;

                kernel_batch_fused_gate_up_swiglu_offsets(batch, expert_buf,
                                                          base,
                                                          base + w_size,
                                                          base + w_size + s_size,
                                                          base + proj_size,
                                                          base + proj_size + w_size,
                                                          base + proj_size + w_size + s_size,
                                                          ctx->gpu_norm_out,
                                                          ctx->gpu_ffn_mid_slots[k],
                                                          (uint32_t)moe_dim, (uint32_t)H,
                                                          (uint32_t)group_size);

                kernel_batch_q4_fma_offsets(batch, expert_buf,
                                            down_base,
                                            down_base + dw_size,
                                            down_base + dw_size + ds_size,
                                            ctx->gpu_ffn_mid_slots[k],
                                            ctx->gpu_result_slots[k],
                                            (uint32_t)H, (uint32_t)moe_dim,
                                            (uint32_t)group_size);
                batch_count++;
            }

            ctx->prev_expert_signal = kernel_end_batch_deferred(batch);
            for (int k = 0; k < batch_count; k++) {
                ctx->prev_expert_weights[k] = top_weights[k];
            }
            ctx->prev_expert_count = batch_count;
        }
    } else
#endif
    for (int k = 0; k < K_active; k++) {
        int expert_idx = top_indices[k];
        float weight = top_weights[k];

        size_t stride;
        const void *expert_data = model_get_expert(ctx->model, layer_idx,
                                                    expert_idx, &stride);
        if (!expert_data) continue;

        {
            // CPU fallback
            const char *ptr = expert_data;

            const void *gw = ptr; ptr += w_size;
            const void *gs = ptr; ptr += s_size;
            const void *gb = ptr; ptr += b_size;
            dequant_matmul_q4(gate_out, gw, gs, gb, s->norm_out, moe_dim, H, group_size);

            const void *uw = ptr; ptr += w_size;
            const void *us = ptr; ptr += s_size;
            const void *ub = ptr; ptr += b_size;
            dequant_matmul_q4(up_out, uw, us, ub, s->norm_out, moe_dim, H, group_size);

            cpu_silu_mul(ffn_mid, gate_out, up_out, moe_dim);

            const void *dw = ptr; ptr += dw_size;
            const void *ds = ptr; ptr += ds_size;
            const void *db = ptr;
            (void)db;

            float *expert_result = arena_alloc(&ctx->arena, (size_t)H * sizeof(float));
            dequant_matmul_q4(expert_result, dw, ds, ptr, ffn_mid, H, moe_dim, group_size);

            for (int i = 0; i < H; i++) {
                s->hidden[i] += weight * expert_result[i];
            }
        }
    }

    // Cross-layer prefetch: hint next layer's shared weights while results
    // are being accumulated. Overlaps SSD reads with CPU work.
    int next_layer = layer_idx + 1;
    if (next_layer < cfg->num_hidden_layers) {
        // Prefetch next layer's norm weights (will be needed first)
        char pf_name[128];
        size_t pf_size;
        snprintf(pf_name, sizeof(pf_name), "layers.%d.input_layernorm.weight", next_layer);
        const void *pf = model_get_weight(ctx->model, pf_name, &pf_size);
        if (pf) mmap_pool_prefetch((void *)pf, pf_size);

        // Prefetch next layer's shared expert gate_proj (largest shared weight)
        snprintf(pf_name, sizeof(pf_name), "layers.%d.mlp.shared_expert.gate_proj.weight", next_layer);
        pf = model_get_weight(ctx->model, pf_name, &pf_size);
        if (pf) mmap_pool_prefetch((void *)pf, pf_size);
    }

    arena_reset(&ctx->arena);
}

static void embed_token(InferenceContext *ctx, int32_t token_id) {
    const ModelConfig *cfg = model_config(ctx->model);
    size_t ws, ss, bs;
    const void *emb_w = model_get_weight(ctx->model, "embed_tokens.weight", &ws);
    const void *emb_s = model_get_weight(ctx->model, "embed_tokens.scales", &ss);
    const void *emb_b = model_get_weight(ctx->model, "embed_tokens.biases", &bs);

    if (!emb_w || !emb_s || !emb_b) {
        memset(ctx->scratch.hidden, 0, (size_t)cfg->hidden_size * sizeof(float));
        return;
    }

    int K = cfg->hidden_size;
    int group_size = 64;
    int K_packed = K / 8;
    int num_groups = K / group_size;

    const uint32_t *row_w = (const uint32_t *)emb_w + (size_t)token_id * K_packed;
    const uint16_t *row_s = (const uint16_t *)emb_s + (size_t)token_id * num_groups;
    const uint16_t *row_b = (const uint16_t *)emb_b + (size_t)token_id * num_groups;

    dequant_row_q4(ctx->scratch.hidden, row_w, row_s, row_b, K, group_size);
}

static void compute_logits(InferenceContext *ctx) {
    const ModelConfig *cfg = model_config(ctx->model);

    // Final RMSNorm before logit projection
    char norm_name[] = "norm.weight";
    size_t nsz;
    const void *norm_bf16 = model_get_weight(ctx->model, norm_name, &nsz);
    if (norm_bf16) {
        int H = cfg->hidden_size;
        float *norm_w = malloc((size_t)H * sizeof(float));
        bf16_to_float_vec(norm_w, norm_bf16, H);
        float *tmp = malloc((size_t)H * sizeof(float));
        cpu_rmsnorm(tmp, ctx->scratch.hidden, norm_w, H, cfg->rms_norm_eps);
        memcpy(ctx->scratch.hidden, tmp, (size_t)H * sizeof(float));
        free(tmp);
        free(norm_w);
    }

    // lm_head has "language_model." prefix in the weight index
    q4_matmul(ctx, ctx->scratch.logits, ctx->scratch.hidden,
              "language_model.lm_head.weight",
              "language_model.lm_head.scales",
              "language_model.lm_head.biases",
              cfg->vocab_size, cfg->hidden_size, 64);
}

int inference_generate(InferenceContext *ctx,
                       const int32_t *prompt_tokens, int num_prompt_tokens,
                       int max_tokens,
                       float temperature, float top_p, int top_k,
                       TokenCallback callback, void *userdata) {
    const ModelConfig *cfg = model_config(ctx->model);
    const Tokenizer *tok = model_tokenizer(ctx->model);

    Sampler *sampler = sampler_create(temperature, top_p, top_k, 1.1f, 64);

    LOG_INFO("inference: generating (prompt=%d tokens, max=%d)",
             num_prompt_tokens, max_tokens);

    uint64_t t_start = timer_now_ns();
    int generated = 0;

    // Process prompt tokens (prefill)
    for (int i = 0; i < num_prompt_tokens; i++) {
        ctx->position = i;
        ctx->kv_idx = 0;
        ctx->dn_idx = 0;
        embed_token(ctx, prompt_tokens[i]);

        for (int l = 0; l < cfg->num_hidden_layers; l++) {
            forward_layer(ctx, l);
        }
#ifdef PLATFORM_MACOS
        complete_deferred_experts(ctx);
#endif
    }

    uint64_t t_prefill = timer_now_ns();
    LOG_INFO("inference: prefill done in %.1f ms",
             timer_elapsed_ms(t_start, t_prefill));

    // Compute logits from the last prefill hidden state
    compute_logits(ctx);

    // Generate tokens
    int32_t next_token = -1;

    for (int t = 0; t < max_tokens; t++) {
        if (t > 0) {
            ctx->position = num_prompt_tokens + t - 1;
            ctx->kv_idx = 0;
            ctx->dn_idx = 0;

            embed_token(ctx, next_token);

            for (int l = 0; l < cfg->num_hidden_layers; l++) {
                forward_layer(ctx, l);
            }
#ifdef PLATFORM_MACOS
            complete_deferred_experts(ctx);
#endif

            compute_logits(ctx);
        }

        // Sample next token from logits
        next_token = sampler_sample(sampler, ctx->scratch.logits, cfg->vocab_size);
        sampler_accept(sampler, next_token);
        generated++;

        // Decode and callback
        const char *text = tokenizer_decode(tok, next_token);
        if (callback) {
            callback(next_token, text, userdata);
        }

        // Stop on EOS or im_end
        if (next_token == tokenizer_eos_id(tok) ||
            next_token == tokenizer_im_end_id(tok)) {
            LOG_INFO("inference: EOS at token %d", generated);
            break;
        }
    }

    uint64_t t_end = timer_now_ns();
    double gen_time = timer_elapsed_ms(t_prefill, t_end);
    double tok_per_sec = generated > 0 ? (double)generated / (gen_time / 1000.0) : 0.0;

    LOG_INFO("inference: generated %d tokens in %.1f ms (%.1f tok/s)",
             generated, gen_time, tok_per_sec);

    sampler_free(sampler);
    return generated;
}

void inference_free(InferenceContext *ctx) {
    if (!ctx) return;
    attention_gpu_free(ctx->attn_gpu);
    cache_free(ctx->cache);

#ifdef PLATFORM_MACOS
    if (ctx->use_gpu) {
        // These were allocated by Metal, not malloc — null them before free_scratch
        ctx->scratch.hidden = NULL;
        ctx->scratch.norm_out = NULL;
        ctx->scratch.logits = NULL;
        metal_free_buffer(ctx->gpu_hidden);
        metal_free_buffer(ctx->gpu_norm_out);
        metal_free_buffer(ctx->gpu_out);
        metal_free_buffer(ctx->gpu_logits);
        metal_free_buffer(ctx->gpu_gate_out);
        metal_free_buffer(ctx->gpu_up_out);
        metal_free_buffer(ctx->gpu_ffn_mid);
        metal_free_buffer(ctx->gpu_expert_result);
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

    free_scratch(&ctx->scratch);
    arena_destroy(&ctx->arena);
    free(ctx);
}
