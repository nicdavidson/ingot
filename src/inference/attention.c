#define _POSIX_C_SOURCE 200809L

#include "inference/attention.h"
#include "inference/dequant.h"
#include "util/log.h"

#ifdef PLATFORM_MACOS
#include "util/timer.h"
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include "compute/kernels.h"
#include "compute/metal_context.h"
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Diagnostic gate from inference.c — only fires on first token
extern int _diag_first_token;

// --- GPU scratch for attention projections ---

// Max simultaneous projection outputs for batching
#define ATTN_GPU_SLOTS 5

// Max sequence length for pre-allocated SWA scores buffer
#define SWA_MAX_SEQ 8192

struct AttentionGPU {
#ifdef PLATFORM_MACOS
    MetalContext *metal;
    void *shared_buf;       // Model's shared weight Metal buffer
    void *gpu_x;            // Input buffer — set by caller via attention_gpu_set_input
    float *cpu_x;           // CPU pointer into gpu_x (unified memory)
    // Multiple output slots for batched projections
    void *gpu_out[ATTN_GPU_SLOTS];
    float *cpu_out[ATTN_GPU_SLOTS];
    int max_out_size;       // Size of each gpu_out slot in floats
#endif

    // --- DeltaNet pre-allocated scratch buffers (CPU-side, plain malloc) ---
    float *dn_qkv;            // [conv_dim]
    float *dn_b_raw;          // [num_v_heads]
    float *dn_a_raw;          // [num_v_heads]
    float *dn_z;              // [z_dim]
    float *dn_beta;           // [num_v_heads]
    float *dn_A_log;          // [num_v_heads]
    float *dn_dt_bias;        // [num_v_heads]
    float *dn_g;              // [num_v_heads]
    float *dn_core_out;       // [value_dim]
    float *dn_norm_w;         // [head_v_dim]
    float *dn_conv_weight;    // [conv_dim * kernel_size]
    float *dn_kv_scratch;     // [num_k_heads * head_v_dim]
    float *dn_delta_scratch;  // [num_k_heads * head_v_dim]

    // DeltaNet dimension sizes
    int dn_conv_dim;
    int dn_value_dim;
    int dn_key_dim;
    int dn_num_k_heads;
    int dn_num_v_heads;
    int dn_head_k_dim;
    int dn_head_v_dim;
    int dn_kernel_size;

    // --- SWA pre-allocated scratch buffers (CPU-side, plain malloc) ---
    float *swa_q_full;        // [q_proj_dim]
    float *swa_k;             // [kv_dim]
    float *swa_v;             // [kv_dim]
    float *swa_q;             // [q_dim]
    float *swa_gate;          // [q_dim]
    float *swa_q_norm_w;      // [head_dim]
    float *swa_k_norm_w;      // [head_dim]
    float *swa_head_out;      // [q_dim]
    float *swa_scores;        // [max_seq] — reused across heads

    // SWA dimension sizes
    int swa_q_proj_dim;
    int swa_q_dim;
    int swa_kv_dim;
    int swa_head_dim;
    int swa_num_heads;
    int swa_num_kv_heads;

    int dummy;              // Keep struct non-empty on non-macOS
};

AttentionGPU *attention_gpu_create(const Model *model, const ModelConfig *cfg) {
    AttentionGPU *gpu = calloc(1, sizeof(AttentionGPU));
    if (!gpu) return NULL;

#ifdef PLATFORM_MACOS
    gpu->metal = model_get_metal(model);
    gpu->shared_buf = model_get_metal_shared_buf(model);
    if (!gpu->metal || !gpu->shared_buf) return gpu;

    int H = cfg->hidden_size;

    // Compute max output dimension across all projection types:
    // SWA: q_proj_dim (may be 2x q_dim if gated), kv_dim, H
    // DeltaNet: conv_dim (key_dim*2+value_dim), z_dim, H, num_v_heads
    int q_dim = cfg->num_attention_heads * cfg->head_dim;
    int q_proj_dim = cfg->attn_output_gate ? q_dim * 2 : q_dim;
    int kv_dim = cfg->num_key_value_heads * cfg->head_dim;

    int key_dim = cfg->linear_attn.linear_num_key_heads *
                  cfg->linear_attn.linear_key_head_dim;
    int value_dim = cfg->linear_attn.linear_num_value_heads *
                    cfg->linear_attn.linear_value_head_dim;
    int conv_dim = key_dim * 2 + value_dim;
    int z_dim = cfg->linear_attn.linear_num_value_heads *
                cfg->linear_attn.linear_value_head_dim;

    int max_out = q_proj_dim;
    if (kv_dim > max_out) max_out = kv_dim;
    if (conv_dim > max_out) max_out = conv_dim;
    if (z_dim > max_out) max_out = z_dim;
    if (H > max_out) max_out = H;

    bool all_ok = true;
    for (int i = 0; i < ATTN_GPU_SLOTS; i++) {
        gpu->gpu_out[i] = metal_alloc_buffer(gpu->metal, (size_t)max_out * sizeof(float));
        if (gpu->gpu_out[i]) {
            gpu->cpu_out[i] = metal_buffer_contents(gpu->gpu_out[i]);
        } else {
            all_ok = false;
            break;
        }
    }
    if (all_ok) {
        gpu->max_out_size = max_out;
        LOG_INFO("attention: GPU buffers allocated (%d slots x %d floats)",
                 ATTN_GPU_SLOTS, max_out);
    } else {
        for (int i = 0; i < ATTN_GPU_SLOTS; i++) {
            if (gpu->gpu_out[i]) metal_free_buffer(gpu->gpu_out[i]);
            gpu->gpu_out[i] = NULL;
        }
    }
#endif

    // --- Allocate DeltaNet scratch buffers ---
    {
        int num_k_heads = cfg->linear_attn.linear_num_key_heads;
        int num_v_heads = cfg->linear_attn.linear_num_value_heads;
        int head_k_dim  = cfg->linear_attn.linear_key_head_dim;
        int head_v_dim  = cfg->linear_attn.linear_value_head_dim;
        int key_dim_    = num_k_heads * head_k_dim;
        int value_dim_  = num_v_heads * head_v_dim;
        int conv_dim_   = key_dim_ * 2 + value_dim_;
        int kernel_size = cfg->linear_attn.linear_conv_kernel_dim;

        gpu->dn_conv_dim    = conv_dim_;
        gpu->dn_value_dim   = value_dim_;
        gpu->dn_key_dim     = key_dim_;
        gpu->dn_num_k_heads = num_k_heads;
        gpu->dn_num_v_heads = num_v_heads;
        gpu->dn_head_k_dim  = head_k_dim;
        gpu->dn_head_v_dim  = head_v_dim;
        gpu->dn_kernel_size = kernel_size;

        gpu->dn_qkv           = malloc((size_t)conv_dim_ * sizeof(float));
        gpu->dn_b_raw         = malloc((size_t)num_v_heads * sizeof(float));
        gpu->dn_a_raw         = malloc((size_t)num_v_heads * sizeof(float));
        gpu->dn_z             = malloc((size_t)value_dim_ * sizeof(float));
        gpu->dn_beta          = malloc((size_t)num_v_heads * sizeof(float));
        gpu->dn_A_log         = malloc((size_t)num_v_heads * sizeof(float));
        gpu->dn_dt_bias       = malloc((size_t)num_v_heads * sizeof(float));
        gpu->dn_g             = malloc((size_t)num_v_heads * sizeof(float));
        gpu->dn_core_out      = malloc((size_t)value_dim_ * sizeof(float));
        gpu->dn_norm_w        = malloc((size_t)head_v_dim * sizeof(float));
        gpu->dn_conv_weight   = malloc((size_t)conv_dim_ * (size_t)kernel_size * sizeof(float));
        gpu->dn_kv_scratch    = malloc((size_t)num_k_heads * (size_t)head_v_dim * sizeof(float));
        gpu->dn_delta_scratch = malloc((size_t)num_k_heads * (size_t)head_v_dim * sizeof(float));
    }

    // --- Allocate SWA scratch buffers ---
    {
        int num_heads    = cfg->num_attention_heads;
        int num_kv_heads = cfg->num_key_value_heads;
        int head_dim     = cfg->head_dim;
        int q_dim_       = num_heads * head_dim;
        int kv_dim_      = num_kv_heads * head_dim;
        int q_proj_dim_  = cfg->attn_output_gate ? q_dim_ * 2 : q_dim_;

        gpu->swa_q_proj_dim  = q_proj_dim_;
        gpu->swa_q_dim       = q_dim_;
        gpu->swa_kv_dim      = kv_dim_;
        gpu->swa_head_dim    = head_dim;
        gpu->swa_num_heads   = num_heads;
        gpu->swa_num_kv_heads = num_kv_heads;

        gpu->swa_q_full   = malloc((size_t)q_proj_dim_ * sizeof(float));
        gpu->swa_k        = malloc((size_t)kv_dim_ * sizeof(float));
        gpu->swa_v        = malloc((size_t)kv_dim_ * sizeof(float));
        gpu->swa_q        = malloc((size_t)q_dim_ * sizeof(float));
        gpu->swa_gate     = malloc((size_t)q_dim_ * sizeof(float));
        gpu->swa_q_norm_w = malloc((size_t)head_dim * sizeof(float));
        gpu->swa_k_norm_w = malloc((size_t)head_dim * sizeof(float));
        gpu->swa_head_out = malloc((size_t)q_dim_ * sizeof(float));
        gpu->swa_scores   = malloc((size_t)SWA_MAX_SEQ * sizeof(float));
    }

    return gpu;
}

void attention_gpu_free(AttentionGPU *gpu) {
    if (!gpu) return;
#ifdef PLATFORM_MACOS
    for (int i = 0; i < ATTN_GPU_SLOTS; i++) {
        if (gpu->gpu_out[i]) metal_free_buffer(gpu->gpu_out[i]);
    }
    // gpu_x is borrowed, not owned
#endif

    // Free DeltaNet scratch
    free(gpu->dn_qkv);
    free(gpu->dn_b_raw);
    free(gpu->dn_a_raw);
    free(gpu->dn_z);
    free(gpu->dn_beta);
    free(gpu->dn_A_log);
    free(gpu->dn_dt_bias);
    free(gpu->dn_g);
    free(gpu->dn_core_out);
    free(gpu->dn_norm_w);
    free(gpu->dn_conv_weight);
    free(gpu->dn_kv_scratch);
    free(gpu->dn_delta_scratch);

    // Free SWA scratch
    free(gpu->swa_q_full);
    free(gpu->swa_k);
    free(gpu->swa_v);
    free(gpu->swa_q);
    free(gpu->swa_gate);
    free(gpu->swa_q_norm_w);
    free(gpu->swa_k_norm_w);
    free(gpu->swa_head_out);
    free(gpu->swa_scores);

    free(gpu);
}

void attention_gpu_set_input(AttentionGPU *gpu, void *gpu_buf, float *cpu_ptr) {
    if (!gpu) return;
#ifdef PLATFORM_MACOS
    gpu->gpu_x = gpu_buf;
    gpu->cpu_x = cpu_ptr;
#else
    (void)gpu_buf;
    (void)cpu_ptr;
#endif
}

// --- Helpers ---

// Batch multiple projections sharing the same input into one command buffer.
// bases[i] is the weight name prefix, outs[i] is where result goes, Ms[i] is output dim.
// All read from gpu->gpu_x, write to gpu_out[0..n-1], then memcpy to outs[i].
#ifdef PLATFORM_MACOS
static bool q4_proj_batch(float *outs[], const float *x, const Model *model,
                          AttentionGPU *gpu, const char *bases[], int Ms[], int n,
                          int K) {
    if (!gpu || !gpu->gpu_x || !gpu->gpu_out[0] || n > ATTN_GPU_SLOTS) return false;

    // Verify all output dims fit and all weights exist
    for (int i = 0; i < n; i++) {
        if (Ms[i] > gpu->max_out_size) return false;
    }

    // Build weight name lookup
    char wn[128], sn[128], bn[128];
    void *batch = kernel_begin_batch(gpu->metal);

    if (x != gpu->cpu_x) {
        memcpy(gpu->cpu_x, x, (size_t)K * sizeof(float));
    }

    for (int i = 0; i < n; i++) {
        snprintf(wn, sizeof(wn), "%s.weight", bases[i]);
        snprintf(sn, sizeof(sn), "%s.scales", bases[i]);
        snprintf(bn, sizeof(bn), "%s.biases", bases[i]);

        long w_off = model_get_weight_offset(model, wn);
        long s_off = model_get_weight_offset(model, sn);
        long b_off = model_get_weight_offset(model, bn);
        if (w_off < 0 || s_off < 0 || b_off < 0) {
            kernel_end_batch(batch);
            return false;
        }

        kernel_batch_q4_fma_offsets(batch, gpu->shared_buf,
                                     (size_t)w_off, (size_t)s_off, (size_t)b_off,
                                     gpu->gpu_x, gpu->gpu_out[i],
                                     (uint32_t)Ms[i], (uint32_t)K, 64);
    }

    kernel_end_batch(batch);  // Single commit for all projections

    // Copy results from unified memory
    for (int i = 0; i < n; i++) {
        memcpy(outs[i], gpu->cpu_out[i], (size_t)Ms[i] * sizeof(float));
    }
    return true;
}
#endif

static void q4_proj(float *out, const float *x, const Model *model,
                    AttentionGPU *gpu, const char *base, int M, int K) {
    char wn[128], sn[128], bn[128];
    snprintf(wn, sizeof(wn), "%s.weight", base);
    snprintf(sn, sizeof(sn), "%s.scales", base);
    snprintf(bn, sizeof(bn), "%s.biases", base);

#ifdef PLATFORM_MACOS
    if (gpu && gpu->gpu_x && gpu->gpu_out[0] && M <= gpu->max_out_size) {
        long w_off = model_get_weight_offset(model, wn);
        long s_off = model_get_weight_offset(model, sn);
        long b_off = model_get_weight_offset(model, bn);
        if (w_off >= 0 && s_off >= 0 && b_off >= 0) {
            if (x != gpu->cpu_x) {
                memcpy(gpu->cpu_x, x, (size_t)K * sizeof(float));
            }
            kernel_matmul_q4_fma_offsets(gpu->metal, gpu->shared_buf,
                                         (size_t)w_off, (size_t)s_off, (size_t)b_off,
                                         gpu->gpu_x, gpu->gpu_out[0],
                                         (uint32_t)M, (uint32_t)K, 64);
            memcpy(out, gpu->cpu_out[0], (size_t)M * sizeof(float));
            return;
        }
    }
#endif

    size_t ws, ss, bs;
    const void *w = model_get_weight(model, wn, &ws);
    const void *s = model_get_weight(model, sn, &ss);
    const void *b = model_get_weight(model, bn, &bs);

    if (!w || !s || !b) {
        memset(out, 0, (size_t)M * sizeof(float));
        return;
    }
    dequant_matmul_q4(out, w, s, b, x, M, K, 64);
}

#ifdef __ARM_NEON
static void load_bf16_neon(float *out, const Model *model, const char *name, int n) {
    size_t sz;
    const void *data = model_get_weight(model, name, &sz);
    if (!data) { memset(out, 0, (size_t)n * sizeof(float)); return; }
    const uint16_t *bf = data;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        uint16x8_t v = vld1q_u16(bf + i);
        uint16x4_t lo = vget_low_u16(v);
        uint16x4_t hi = vget_high_u16(v);
        uint32x4_t lo32 = vshll_n_u16(lo, 16);
        uint32x4_t hi32 = vshll_n_u16(hi, 16);
        vst1q_f32(out + i, vreinterpretq_f32_u32(lo32));
        vst1q_f32(out + i + 4, vreinterpretq_f32_u32(hi32));
    }
    for (; i < n; i++) out[i] = bf16_to_f32(bf[i]);
}
#define load_bf16 load_bf16_neon
#else
static void load_bf16(float *out, const Model *model, const char *name, int n) {
    size_t sz;
    const void *data = model_get_weight(model, name, &sz);
    if (!data) { memset(out, 0, (size_t)n * sizeof(float)); return; }
    const uint16_t *bf = data;
    for (int i = 0; i < n; i++) out[i] = bf16_to_f32(bf[i]);
}
#endif

static void load_f32(float *out, const Model *model, const char *name, int n) {
    size_t sz;
    const void *data = model_get_weight(model, name, &sz);
    if (!data) { memset(out, 0, (size_t)n * sizeof(float)); return; }
    memcpy(out, data, (size_t)n * sizeof(float));
}

static void softmax_vec(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static float softplus(float x) {
    if (x > 20.0f) return x;  // avoid overflow
    return logf(1.0f + expf(x));
}

// L2 normalize a vector in-place
static void l2norm(float *x, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss + 1e-6f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// --- SWA (Full Attention) ---

void attention_swa_forward(
    float *attn_out, const float *hidden,
    const Model *model, const ModelConfig *cfg,
    InferenceCache *cache, AttentionGPU *gpu,
    int layer_idx, int kv_layer_idx, int position)
{
    int H = cfg->hidden_size;
    int num_heads = cfg->num_attention_heads;
    int num_kv_heads = cfg->num_key_value_heads;
    int head_dim = cfg->head_dim;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int rotary_dim = (int)((float)head_dim * cfg->rope.partial_rotary_factor);

    // When attn_output_gate is enabled, q_proj is 2x wider:
    // first half = query, second half = gate for output gating
    int q_proj_dim = cfg->attn_output_gate ? q_dim * 2 : q_dim;

    char base[128];

    // Use pre-allocated scratch if gpu is available, else fall back to calloc
    float *q_full, *k, *v, *q, *gate, *q_norm_w, *k_norm_w, *head_out;
    int use_scratch = (gpu != NULL && gpu->swa_q_full != NULL);

    if (use_scratch) {
        q_full = gpu->swa_q_full;
        memset(q_full, 0, (size_t)q_proj_dim * sizeof(float));
        k = gpu->swa_k;
        memset(k, 0, (size_t)kv_dim * sizeof(float));
        v = gpu->swa_v;
        memset(v, 0, (size_t)kv_dim * sizeof(float));
    } else {
        q_full = calloc((size_t)q_proj_dim, sizeof(float));
        k = calloc((size_t)kv_dim, sizeof(float));
        v = calloc((size_t)kv_dim, sizeof(float));
    }

    // Q/K/V projections — batch into single GPU commit
    char base_q[128], base_k[128], base_v[128];
    snprintf(base_q, sizeof(base_q), "layers.%d.self_attn.q_proj", layer_idx);
    snprintf(base_k, sizeof(base_k), "layers.%d.self_attn.k_proj", layer_idx);
    snprintf(base_v, sizeof(base_v), "layers.%d.self_attn.v_proj", layer_idx);

#ifdef PLATFORM_MACOS
    {
        float *batch_outs[] = { q_full, k, v };
        const char *batch_bases[] = { base_q, base_k, base_v };
        int batch_ms[] = { q_proj_dim, kv_dim, kv_dim };
        if (!q4_proj_batch(batch_outs, hidden, model, gpu, batch_bases, batch_ms, 3, H)) {
            q4_proj(q_full, hidden, model, gpu, base_q, q_proj_dim, H);
            q4_proj(k, hidden, model, gpu, base_k, kv_dim, H);
            q4_proj(v, hidden, model, gpu, base_v, kv_dim, H);
        }
    }
#else
    q4_proj(q_full, hidden, model, gpu, base_q, q_proj_dim, H);
    q4_proj(k, hidden, model, gpu, base_k, kv_dim, H);
    q4_proj(v, hidden, model, gpu, base_v, kv_dim, H);
#endif

    // Split q_proj into query and gate
    // q_full is laid out as [num_heads, head_dim*2] when gated
    // We need to split each head's chunk: first head_dim = query, second = gate
    if (use_scratch) {
        q = gpu->swa_q;
        memset(q, 0, (size_t)q_dim * sizeof(float));
        gate = cfg->attn_output_gate ? gpu->swa_gate : NULL;
    } else {
        q = calloc((size_t)q_dim, sizeof(float));
        gate = cfg->attn_output_gate ? calloc((size_t)q_dim, sizeof(float)) : NULL;
    }

    if (cfg->attn_output_gate) {
        for (int h = 0; h < num_heads; h++) {
            memcpy(q + h * head_dim, q_full + h * head_dim * 2,
                   (size_t)head_dim * sizeof(float));
            memcpy(gate + h * head_dim, q_full + h * head_dim * 2 + head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    } else {
        memcpy(q, q_full, (size_t)q_dim * sizeof(float));
    }
    if (!use_scratch) free(q_full);

    // QK normalization (per-head RMSNorm)
    if (use_scratch) {
        q_norm_w = gpu->swa_q_norm_w;
        k_norm_w = gpu->swa_k_norm_w;
    } else {
        q_norm_w = calloc((size_t)head_dim, sizeof(float));
        k_norm_w = calloc((size_t)head_dim, sizeof(float));
    }
    snprintf(base, sizeof(base), "layers.%d.self_attn.q_norm.weight", layer_idx);
    load_bf16(q_norm_w, model, base, head_dim);
    snprintf(base, sizeof(base), "layers.%d.self_attn.k_norm.weight", layer_idx);
    load_bf16(k_norm_w, model, base, head_dim);

    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i++) ss += qh[i] * qh[i];
        float rms = sqrtf(ss / (float)head_dim + cfg->rms_norm_eps);
        for (int i = 0; i < head_dim; i++) qh[i] = (qh[i] / rms) * q_norm_w[i];
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i++) ss += kh[i] * kh[i];
        float rms = sqrtf(ss / (float)head_dim + cfg->rms_norm_eps);
        for (int i = 0; i < head_dim; i++) kh[i] = (kh[i] / rms) * k_norm_w[i];
    }

    // RoPE (partial rotary)
    float theta_base = (float)cfg->rope.rope_theta;
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)rotary_dim);
            float angle = (float)position * freq;
            float c = cosf(angle), sn_val = sinf(angle);
            int idx = h * head_dim + i * 2;
            float x0 = q[idx], x1 = q[idx + 1];
            q[idx]     = x0 * c - x1 * sn_val;
            q[idx + 1] = x0 * sn_val + x1 * c;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)rotary_dim);
            float angle = (float)position * freq;
            float c = cosf(angle), sn_val = sinf(angle);
            int idx = h * head_dim + i * 2;
            float x0 = k[idx], x1 = k[idx + 1];
            k[idx]     = x0 * c - x1 * sn_val;
            k[idx + 1] = x0 * sn_val + x1 * c;
        }
    }

    // Cache KV
    cache_kv_append(cache, kv_layer_idx, k, v);

    const float *cached_k, *cached_v;
    int seq_len;
    cache_kv_get(cache, kv_layer_idx, &cached_k, &cached_v, &seq_len);

    // GQA Attention
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_group = num_heads / num_kv_heads;

    if (use_scratch) {
        head_out = gpu->swa_head_out;
        memset(head_out, 0, (size_t)q_dim * sizeof(float));
    } else {
        head_out = calloc((size_t)q_dim, sizeof(float));
    }

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / kv_group;
        const float *qi = q + h * head_dim;

        float *scores;
        if (use_scratch && seq_len <= SWA_MAX_SEQ) {
            scores = gpu->swa_scores;
            memset(scores, 0, (size_t)seq_len * sizeof(float));
        } else {
            scores = calloc((size_t)seq_len, sizeof(float));
        }

        for (int t = 0; t < seq_len; t++) {
            const float *ki = cached_k + t * num_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qi[d] * ki[d];
            scores[t] = dot * scale;
        }
        softmax_vec(scores, seq_len);

        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                sum += scores[t] * cached_v[t * num_kv_heads * head_dim + kv_h * head_dim + d];
            }
            head_out[h * head_dim + d] = sum;
        }

        if (!(use_scratch && seq_len <= SWA_MAX_SEQ)) {
            free(scores);
        }
    }

    // Apply output gate: attn_out = sigmoid(gate) * head_out
    if (gate) {
        // gate is [num_heads, head_dim], reshape to [q_dim] = [num_heads * head_dim]
        for (int i = 0; i < q_dim; i++) {
            head_out[i] *= 1.0f / (1.0f + expf(-gate[i]));
        }
    }

    // Output projection
    snprintf(base, sizeof(base), "layers.%d.self_attn.o_proj", layer_idx);
    q4_proj(attn_out, head_out, model, gpu, base, H, q_dim);

    if (!use_scratch) {
        free(q); free(k); free(v);
        free(q_norm_w); free(k_norm_w);
        free(head_out);
        free(gate);
    }
}

// --- Gated DeltaNet (Linear Attention / SSM) ---
//
// Implements the recurrent gated delta rule from HuggingFace transformers.
// For single-token generation, the recurrent form is:
//
//   S = S * exp(g)                        — decay state
//   kv_mem = (S * k_t).sum(key_dim)       — read from state
//   delta = (v_t - kv_mem) * beta_t       — error-corrected update
//   S = S + k_t.outer(delta)              — write to state
//   output = (S * q_t).sum(key_dim)       — query the state


#ifdef PLATFORM_MACOS
static double _dn_acc_proj = 0, _dn_acc_conv = 0, _dn_acc_recur = 0;
static double _dn_acc_norm = 0, _dn_acc_oproj = 0, _dn_acc_loads = 0;
static int _dn_layer_count = 0;
void attention_dn_timing_report(int token_num) {
    double total = _dn_acc_proj + _dn_acc_conv + _dn_acc_recur + _dn_acc_norm + _dn_acc_oproj + _dn_acc_loads;
    fprintf(stderr, "[DN-TIMING tok%d] proj=%.1f conv=%.1f recur=%.1f norm=%.1f oproj=%.1f loads=%.1f TOTAL=%.1f ms (%d layers)\n",
            token_num, _dn_acc_proj, _dn_acc_conv, _dn_acc_recur, _dn_acc_norm, _dn_acc_oproj, _dn_acc_loads, total, _dn_layer_count);
    _dn_acc_proj = _dn_acc_conv = _dn_acc_recur = _dn_acc_norm = _dn_acc_oproj = _dn_acc_loads = 0;
    _dn_layer_count = 0;
}
#endif
void attention_deltanet_forward(
    float *attn_out, const float *hidden,
    const Model *model, const ModelConfig *cfg,
    InferenceCache *cache, AttentionGPU *gpu,
    int layer_idx, int dn_layer_idx, int position)
{
    (void)position;

    int H = cfg->hidden_size;
    int num_k_heads = cfg->linear_attn.linear_num_key_heads;
    int num_v_heads = cfg->linear_attn.linear_num_value_heads;
    int head_k_dim = cfg->linear_attn.linear_key_head_dim;
    int head_v_dim = cfg->linear_attn.linear_value_head_dim;
    int key_dim = num_k_heads * head_k_dim;   // total key dimension
    int value_dim = num_v_heads * head_v_dim;  // total value dimension
    int conv_dim = key_dim * 2 + value_dim;    // QKV fused dimension


#ifdef PLATFORM_MACOS
    uint64_t _dt0 = timer_now_ns(), _dt1;
#endif
    char base[128], wname[128];

    // Determine if we can use pre-allocated scratch
    int use_scratch = (gpu != NULL && gpu->dn_qkv != NULL);

    // 1-5. Batch all projections from hidden (qkv, b, a, z) in one GPU commit
    float *qkv, *b_raw, *a_raw, *z;
    int z_dim = num_v_heads * head_v_dim;

    if (use_scratch) {
        qkv = gpu->dn_qkv;
        memset(qkv, 0, (size_t)conv_dim * sizeof(float));
        b_raw = gpu->dn_b_raw;
        memset(b_raw, 0, (size_t)num_v_heads * sizeof(float));
        a_raw = gpu->dn_a_raw;
        memset(a_raw, 0, (size_t)num_v_heads * sizeof(float));
        z = gpu->dn_z;
        memset(z, 0, (size_t)z_dim * sizeof(float));
    } else {
        qkv = calloc((size_t)conv_dim, sizeof(float));
        b_raw = calloc((size_t)num_v_heads, sizeof(float));
        a_raw = calloc((size_t)num_v_heads, sizeof(float));
        z = calloc((size_t)z_dim, sizeof(float));
    }

    char base_qkv[128], base_b[128], base_a[128], base_z[128];
    snprintf(base_qkv, sizeof(base_qkv), "layers.%d.linear_attn.in_proj_qkv", layer_idx);
    snprintf(base_b, sizeof(base_b), "layers.%d.linear_attn.in_proj_b", layer_idx);
    snprintf(base_a, sizeof(base_a), "layers.%d.linear_attn.in_proj_a", layer_idx);
    snprintf(base_z, sizeof(base_z), "layers.%d.linear_attn.in_proj_z", layer_idx);

#ifdef PLATFORM_MACOS
    {
        float *batch_outs[] = { qkv, b_raw, a_raw, z };
        const char *batch_bases[] = { base_qkv, base_b, base_a, base_z };
        int batch_ms[] = { conv_dim, num_v_heads, num_v_heads, z_dim };
        if (!q4_proj_batch(batch_outs, hidden, model, gpu, batch_bases, batch_ms, 4, H)) {
            // Fallback to individual calls
            q4_proj(qkv, hidden, model, gpu, base_qkv, conv_dim, H);
            q4_proj(b_raw, hidden, model, gpu, base_b, num_v_heads, H);
            q4_proj(a_raw, hidden, model, gpu, base_a, num_v_heads, H);
            q4_proj(z, hidden, model, gpu, base_z, z_dim, H);
        }
    }
#else
    q4_proj(qkv, hidden, model, gpu, base_qkv, conv_dim, H);
    q4_proj(b_raw, hidden, model, gpu, base_b, num_v_heads, H);
    q4_proj(a_raw, hidden, model, gpu, base_a, num_v_heads, H);
    q4_proj(z, hidden, model, gpu, base_z, z_dim, H);
#endif

    // DIAG: after QKV projection
    if (_diag_first_token && layer_idx == 0) {
        float ss = 0.0f;
        for (int i = 0; i < conv_dim; i++) ss += qkv[i] * qkv[i];
        fprintf(stderr, "[DN-DIAG L%d] qkv_proj rms=%.6f first4=[%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, sqrtf(ss/(float)conv_dim), qkv[0], qkv[1], qkv[2], qkv[3]);
        float ss_z = 0.0f;
        for (int i = 0; i < z_dim; i++) ss_z += z[i] * z[i];
        fprintf(stderr, "[DN-DIAG L%d] z_proj rms=%.6f first4=[%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, sqrtf(ss_z/(float)z_dim), z[0], z[1], z[2], z[3]);
        fprintf(stderr, "[DN-DIAG L%d] beta_raw first4=[%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, b_raw[0], b_raw[1], b_raw[2], b_raw[3]);
        fprintf(stderr, "[DN-DIAG L%d] alpha_raw first4=[%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, a_raw[0], a_raw[1], a_raw[2], a_raw[3]);
    }

#ifdef PLATFORM_MACOS
    _dt1 = timer_now_ns(); _dn_acc_proj += timer_elapsed_ms(_dt0, _dt1); _dt0 = _dt1;
#endif
    // 2. Causal conv1d with state tracking
    int kernel_size = cfg->linear_attn.linear_conv_kernel_dim;
    float *conv_weight;
    if (use_scratch) {
        conv_weight = gpu->dn_conv_weight;
    } else {
        conv_weight = calloc((size_t)conv_dim * (size_t)kernel_size, sizeof(float));
    }
    snprintf(wname, sizeof(wname), "layers.%d.linear_attn.conv1d.weight", layer_idx);
    load_bf16(conv_weight, model, wname, conv_dim * kernel_size);

    float *conv_state = cache_dn_conv_get(cache, dn_layer_idx);
    int state_len = kernel_size - 1;

    for (int c = 0; c < conv_dim; c++) {
        float *cs = conv_state + c * state_len;
        float *cw = conv_weight + c * kernel_size;

        float sum = 0.0f;
        for (int k = 0; k < state_len; k++) {
            sum += cs[k] * cw[k];
        }
        sum += qkv[c] * cw[state_len];

        for (int k = 0; k < state_len - 1; k++) {
            cs[k] = cs[k + 1];
        }
        if (state_len > 0) cs[state_len - 1] = qkv[c];

        qkv[c] = sum / (1.0f + expf(-sum));
    }

    // DIAG: after conv1d+SiLU
    if (_diag_first_token && layer_idx == 0) {
        float ss = 0.0f;
        for (int i = 0; i < conv_dim; i++) ss += qkv[i] * qkv[i];
        fprintf(stderr, "[DN-DIAG L%d] after conv1d rms=%.6f first4=[%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, sqrtf(ss/(float)conv_dim), qkv[0], qkv[1], qkv[2], qkv[3]);
    }

    // 3. Split QKV
    float *query = qkv;
    float *key = qkv + key_dim;
    float *value = qkv + key_dim * 2;

    // 4. Compute gates
    float *beta;
    if (use_scratch) {
        beta = gpu->dn_beta;
    } else {
        beta = calloc((size_t)num_v_heads, sizeof(float));
    }
    for (int i = 0; i < num_v_heads; i++) {
        beta[i] = 1.0f / (1.0f + expf(-b_raw[i]));
    }

    float *A_log;
    if (use_scratch) {
        A_log = gpu->dn_A_log;
    } else {
        A_log = calloc((size_t)num_v_heads, sizeof(float));
    }
    snprintf(wname, sizeof(wname), "layers.%d.linear_attn.A_log", layer_idx);
    load_f32(A_log, model, wname, num_v_heads);

    float *dt_bias_w;
    if (use_scratch) {
        dt_bias_w = gpu->dn_dt_bias;
    } else {
        dt_bias_w = calloc((size_t)num_v_heads, sizeof(float));
    }
    snprintf(wname, sizeof(wname), "layers.%d.linear_attn.dt_bias", layer_idx);
    load_bf16(dt_bias_w, model, wname, num_v_heads);

    float *g;
    if (use_scratch) {
        g = gpu->dn_g;
    } else {
        g = calloc((size_t)num_v_heads, sizeof(float));
    }
    for (int i = 0; i < num_v_heads; i++) {
        g[i] = -expf(A_log[i]) * softplus(a_raw[i] + dt_bias_w[i]);
    }

    // 6. RMS normalize Q and K per head (bare, no weights)
    // Matches flash-moe exactly: Q = rms_norm(q) * inv_scale^2, K = rms_norm(k) * inv_scale
    // where inv_scale = 1/sqrt(head_k_dim)
    {
        float inv_scale = 1.0f / sqrtf((float)head_k_dim);
        float q_post_scale = inv_scale * inv_scale;  // 1/head_k_dim
        for (int h = 0; h < num_k_heads; h++) {
            float *qh = query + h * head_k_dim;
            float *kh = key + h * head_k_dim;

            // RMS norm Q (bare) then scale by inv_scale^2
            float ss_q = 0.0f;
            for (int d = 0; d < head_k_dim; d++) ss_q += qh[d] * qh[d];
            float inv_rms_q = 1.0f / sqrtf(ss_q / (float)head_k_dim + 1e-6f);
            for (int d = 0; d < head_k_dim; d++) qh[d] = qh[d] * inv_rms_q * q_post_scale;

            // RMS norm K (bare) then scale by inv_scale
            float ss_k = 0.0f;
            for (int d = 0; d < head_k_dim; d++) ss_k += kh[d] * kh[d];
            float inv_rms_k = 1.0f / sqrtf(ss_k / (float)head_k_dim + 1e-6f);
            for (int d = 0; d < head_k_dim; d++) kh[d] = kh[d] * inv_rms_k * inv_scale;
        }
    }

    // DIAG: after Q/K normalization
    if (_diag_first_token && layer_idx == 0) {
        float ss_q = 0.0f, ss_k = 0.0f;
        for (int i = 0; i < key_dim; i++) ss_q += query[i] * query[i];
        for (int i = 0; i < key_dim; i++) ss_k += key[i] * key[i];
        fprintf(stderr, "[DN-DIAG L%d] after QK_norm: Q rms=%.6f first4=[%.8f,%.8f,%.8f,%.8f]\n",
                layer_idx, sqrtf(ss_q/(float)key_dim), query[0], query[1], query[2], query[3]);
        fprintf(stderr, "[DN-DIAG L%d] after QK_norm: K rms=%.6f first4=[%.8f,%.8f,%.8f,%.8f]\n",
                layer_idx, sqrtf(ss_k/(float)key_dim), key[0], key[1], key[2], key[3]);
        float ss_v = 0.0f;
        for (int i = 0; i < value_dim; i++) ss_v += value[i] * value[i];
        fprintf(stderr, "[DN-DIAG L%d] V rms=%.6f first4=[%.8f,%.8f,%.8f,%.8f]\n",
                layer_idx, sqrtf(ss_v/(float)value_dim), value[0], value[1], value[2], value[3]);
    }

#ifdef PLATFORM_MACOS
    _dt1 = timer_now_ns(); _dn_acc_conv += timer_elapsed_ms(_dt0, _dt1); _dt0 = _dt1;
#endif
    // 8. Recurrent state update (per head)
    // State S is [num_k_heads, head_k_dim, head_v_dim]
    // But HF uses num_v_heads for the outer loop — heads may be grouped
    // For Qwen 3.5: num_k_heads=16, num_v_heads=64, head_k_dim=128, head_v_dim=128
    // This means 4 value heads per key head

    float *state = cache_dn_get(cache, dn_layer_idx);

    // After q4_proj_batch completes and copies results, gpu_out[0..3] are free
    float *core_out;
#ifdef PLATFORM_MACOS
    if (gpu && gpu->cpu_out[0]) {
        core_out = gpu->cpu_out[0];  // Metal unified memory — no copy needed for o_proj
        memset(core_out, 0, (size_t)value_dim * sizeof(float));
    } else
#endif
    {
        if (use_scratch) {
            core_out = gpu->dn_core_out;  // pre-allocated scratch
        } else {
            core_out = calloc((size_t)value_dim, sizeof(float));
        }
        memset(core_out, 0, (size_t)value_dim * sizeof(float));
    }

    int v_per_k = num_v_heads / num_k_heads; // value heads per key head
    int state_size = head_k_dim * head_v_dim;

#ifdef PLATFORM_MACOS
    // GCD-parallelized recurrence: dispatch key heads to separate cores
    // Each key head processes its v_per_k value heads sequentially
    // Use pre-allocated scratch per key head to avoid calloc inside dispatch blocks
    float *kv_scratch, *delta_scratch;
    if (use_scratch) {
        kv_scratch = gpu->dn_kv_scratch;
        delta_scratch = gpu->dn_delta_scratch;
    } else {
        kv_scratch = calloc((size_t)num_k_heads * head_v_dim, sizeof(float));
        delta_scratch = calloc((size_t)num_k_heads * head_v_dim, sizeof(float));
    }

    dispatch_apply((size_t)num_k_heads,
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t kh_idx) {
        int kh = (int)kh_idx;
        float *q_h = query + kh * head_k_dim;
        float *k_h = key + kh * head_k_dim;
        float *my_kv = kv_scratch + kh * head_v_dim;
        float *my_delta = delta_scratch + kh * head_v_dim;

        for (int vh_local = 0; vh_local < v_per_k; vh_local++) {
            int vh = kh * v_per_k + vh_local;
            float *v_h = value + vh * head_v_dim;
            float g_h = g[vh];
            float beta_h = beta[vh];
            float *S = state + ((size_t)kh * v_per_k + vh_local) *
                       (size_t)state_size;
            float decay_val = expf(g_h);

            cblas_sscal(state_size, decay_val, S, 1);

            cblas_sgemv(CblasRowMajor, CblasTrans,
                        head_k_dim, head_v_dim,
                        1.0f, S, head_v_dim,
                        k_h, 1,
                        0.0f, my_kv, 1);

            for (int vd = 0; vd < head_v_dim; vd++) {
                my_delta[vd] = (v_h[vd] - my_kv[vd]) * beta_h;
            }

            cblas_sger(CblasRowMajor,
                       head_k_dim, head_v_dim,
                       1.0f, k_h, 1, my_delta, 1,
                       S, head_v_dim);

            float *out_h = core_out + vh * head_v_dim;
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        head_k_dim, head_v_dim,
                        1.0f, S, head_v_dim,
                        q_h, 1,
                        1.0f, out_h, 1);
        }
    });

    if (!use_scratch) {
        free(kv_scratch);
        free(delta_scratch);
    }

#else
    // Portable scalar fallback for non-Apple platforms
    for (int kh = 0; kh < num_k_heads; kh++) {
        float *q_h = query + kh * head_k_dim;
        float *k_h = key + kh * head_k_dim;

        for (int vh_local = 0; vh_local < v_per_k; vh_local++) {
            int vh = kh * v_per_k + vh_local;
            float *v_h = value + vh * head_v_dim;
            float g_h = g[vh];
            float beta_h = beta[vh];
            float *S = state + ((size_t)kh * v_per_k + vh_local) *
                       (size_t)state_size;
            float decay = expf(g_h);

            for (int i = 0; i < state_size; i++) S[i] *= decay;

            float *kv_mem = calloc((size_t)head_v_dim, sizeof(float));
            for (int kd = 0; kd < head_k_dim; kd++) {
                for (int vd = 0; vd < head_v_dim; vd++) {
                    kv_mem[vd] += S[kd * head_v_dim + vd] * k_h[kd];
                }
            }

            float *delta = calloc((size_t)head_v_dim, sizeof(float));
            for (int vd = 0; vd < head_v_dim; vd++) {
                delta[vd] = (v_h[vd] - kv_mem[vd]) * beta_h;
            }

            for (int kd = 0; kd < head_k_dim; kd++) {
                for (int vd = 0; vd < head_v_dim; vd++) {
                    S[kd * head_v_dim + vd] += k_h[kd] * delta[vd];
                }
            }

            float *out_h = core_out + vh * head_v_dim;
            for (int kd = 0; kd < head_k_dim; kd++) {
                for (int vd = 0; vd < head_v_dim; vd++) {
                    out_h[vd] += S[kd * head_v_dim + vd] * q_h[kd];
                }
            }

            free(kv_mem);
            free(delta);
        }
    }
#endif

    // DIAG: after recurrence
    if (_diag_first_token && layer_idx == 0) {
        float ss = 0.0f;
        for (int i = 0; i < value_dim; i++) ss += core_out[i] * core_out[i];
        fprintf(stderr, "[DN-DIAG L%d] core_out rms=%.6f first4=[%.8f,%.8f,%.8f,%.8f]\n",
                layer_idx, sqrtf(ss/(float)value_dim), core_out[0], core_out[1], core_out[2], core_out[3]);
    }

#ifdef PLATFORM_MACOS
    _dt1 = timer_now_ns(); _dn_acc_recur += timer_elapsed_ms(_dt0, _dt1); _dt0 = _dt1;
#endif
    // 9. RMSNormGated: rmsnorm(core_out) * silu(z)
    float *norm_w;
    if (use_scratch) {
        norm_w = gpu->dn_norm_w;
    } else {
        norm_w = calloc((size_t)head_v_dim, sizeof(float));
    }
    snprintf(wname, sizeof(wname), "layers.%d.linear_attn.norm.weight", layer_idx);
    load_bf16(norm_w, model, wname, head_v_dim);

    // Apply per-head RMSNorm then multiply by SiLU(z)
    for (int vh = 0; vh < num_v_heads; vh++) {
        float *out_h = core_out + vh * head_v_dim;
        float *z_h = z + vh * head_v_dim;

        // RMSNorm
        float ss = 0.0f;
        for (int i = 0; i < head_v_dim; i++) ss += out_h[i] * out_h[i];
        float rms = sqrtf(ss / (float)head_v_dim + 1e-6f);
        for (int i = 0; i < head_v_dim; i++) {
            float normed = (out_h[i] / rms) * norm_w[i];
            float silu_z = z_h[i] / (1.0f + expf(-z_h[i]));
            out_h[i] = normed * silu_z;
        }
    }

    // DIAG: after gated RMSNorm
    if (_diag_first_token && layer_idx == 0) {
        float ss = 0.0f;
        for (int i = 0; i < value_dim; i++) ss += core_out[i] * core_out[i];
        fprintf(stderr, "[DN-DIAG L%d] after gated_norm rms=%.6f first4=[%.8f,%.8f,%.8f,%.8f]\n",
                layer_idx, sqrtf(ss/(float)value_dim), core_out[0], core_out[1], core_out[2], core_out[3]);
    }

#ifdef PLATFORM_MACOS
    _dt1 = timer_now_ns(); _dn_acc_norm += timer_elapsed_ms(_dt0, _dt1); _dt0 = _dt1;
#endif
    // 10. Output projection
    snprintf(base, sizeof(base), "layers.%d.linear_attn.out_proj", layer_idx);

    // Zero-copy o_proj: if core_out is in GPU unified memory, skip the memcpy
#ifdef PLATFORM_MACOS
    if (gpu && gpu->gpu_out[0] && core_out == gpu->cpu_out[0]) {
        // core_out is in GPU memory — set as input for o_proj, skip memcpy
        attention_gpu_set_input(gpu, gpu->gpu_out[0], gpu->cpu_out[0]);
    }
#endif
    q4_proj(attn_out, core_out, model, gpu, base, H, value_dim);

    // DIAG: after o_proj
    if (_diag_first_token && layer_idx == 0) {
        float ss = 0.0f;
        for (int i = 0; i < H; i++) ss += attn_out[i] * attn_out[i];
        fprintf(stderr, "[DN-DIAG L%d] o_proj out rms=%.6f first4=[%.8f,%.8f,%.8f,%.8f]\n",
                layer_idx, sqrtf(ss/(float)H), attn_out[0], attn_out[1], attn_out[2], attn_out[3]);
    }

#ifdef PLATFORM_MACOS
    _dt1 = timer_now_ns(); _dn_acc_oproj += timer_elapsed_ms(_dt0, _dt1);
    _dn_layer_count++;
#endif

    if (!use_scratch) {
        free(qkv); free(b_raw); free(beta); free(a_raw);
        free(A_log); free(dt_bias_w); free(g); free(z);
        free(norm_w); free(conv_weight);
        // core_out: only free if it was calloc'd (not gpu unified memory or scratch)
#ifdef PLATFORM_MACOS
        if (!(gpu && gpu->cpu_out[0] && core_out == gpu->cpu_out[0])) {
            free(core_out);
        }
#else
        free(core_out);
#endif
    } else {
        // Scratch mode: core_out might be gpu->cpu_out[0] (unified memory) — don't free that
        // All other buffers are pre-allocated scratch — don't free
    }
}


// --- DeepSeek V4 MLA (Multi-head Latent Attention) ---
//
// Q: LoRA low-rank projection (wq_a → RMSNorm → wq_b), then per-head inline norm.
// KV: Single-head latent (wkv → kv_norm). RoPE on last qk_rope_head_dim dims.
// Scoring: Full dot product Q[h,head_dim] @ K[head_dim] with sliding window mask.
// Values: Pre-RoPE KV latent (O projection absorbs V up-projection).
// O: Grouped LoRA (per-group wo_a → concat → wo_b).

void attention_v4_mla_forward(
    float *attn_out, const float *hidden,
    const Model *model, const ModelConfig *cfg,
    InferenceCache *cache, AttentionGPU *gpu,
    int layer_idx, int kv_layer_idx, int position)
{
    int H = cfg->hidden_size;
    int num_heads = cfg->num_attention_heads;
    int head_dim = cfg->head_dim;
    int q_lora_rank = cfg->v4.q_lora_rank;
    int o_lora_rank = cfg->v4.o_lora_rank;
    int o_groups = cfg->v4.o_groups;
    int qk_rope_dim = cfg->v4.qk_rope_head_dim;
    int nope_dim = head_dim - qk_rope_dim;
    int window_size = cfg->v4.window_size;
    int q_dim = num_heads * head_dim;
    int group_in_dim = (num_heads / o_groups) * head_dim;
    float eps = cfg->rms_norm_eps;
    // V4 uses different RoPE bases per layer type:
    //   compress_ratio == 0  → standard rope_theta (10000)
    //   compress_ratio  > 0  → compress_rope_theta (160000), with YaRN scaling
    //                          (YaRN itself is not yet implemented; affects long
    //                          contexts more than short ones).
    int compress_ratio = cfg->v4.compress_ratios
        ? cfg->v4.compress_ratios[layer_idx] : 0;
    float theta_base = compress_ratio > 0
        ? (float)cfg->v4.compress_rope_theta
        : (float)cfg->rope.rope_theta;

    char base[128], wname[128];

    // 1. Q projection (low-rank): hidden → wq_a → RMSNorm → wq_b → [num_heads * head_dim]
    float *q_low = calloc((size_t)q_lora_rank, sizeof(float));

    snprintf(base, sizeof(base), "layers.%d.attn.wq_a", layer_idx);
    q4_proj(q_low, hidden, model, gpu, base, q_lora_rank, H);

    float *q_norm_w = calloc((size_t)q_lora_rank, sizeof(float));
    snprintf(wname, sizeof(wname), "layers.%d.attn.q_norm.weight", layer_idx);
    load_bf16(q_norm_w, model, wname, q_lora_rank);
    {
        float ss = 0.0f;
#ifdef PLATFORM_MACOS
        vDSP_svesq(q_low, 1, &ss, (vDSP_Length)q_lora_rank);
#else
        for (int i = 0; i < q_lora_rank; i++) ss += q_low[i] * q_low[i];
#endif
        float rms = sqrtf(ss / (float)q_lora_rank + eps);
        for (int i = 0; i < q_lora_rank; i++)
            q_low[i] = (q_low[i] / rms) * q_norm_w[i];
    }
    free(q_norm_w);

    float *q = calloc((size_t)q_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.attn.wq_b", layer_idx);
    q4_proj(q, q_low, model, gpu, base, q_dim, q_lora_rank);
    free(q_low);

    // 2. Per-head inline Q norm (bare RMSNorm, no learned weight)
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        float ss = 0.0f;
#ifdef PLATFORM_MACOS
        vDSP_svesq(qh, 1, &ss, (vDSP_Length)head_dim);
#else
        for (int d = 0; d < head_dim; d++) ss += qh[d] * qh[d];
#endif
        float inv_rms = 1.0f / sqrtf(ss / (float)head_dim + 1e-6f);
#ifdef PLATFORM_MACOS
        vDSP_vsmul(qh, 1, &inv_rms, qh, 1, (vDSP_Length)head_dim);
#else
        for (int d = 0; d < head_dim; d++) qh[d] *= inv_rms;
#endif
    }

    // 3. RoPE on Q (last qk_rope_dim dims of each head)
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < qk_rope_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)qk_rope_dim);
            float angle = (float)position * freq;
            float c = cosf(angle), s = sinf(angle);
            int idx = h * head_dim + nope_dim + i * 2;
            float x0 = q[idx], x1 = q[idx + 1];
            q[idx]     = x0 * c - x1 * s;
            q[idx + 1] = x0 * s + x1 * c;
        }
    }

    // 4. KV projection (MQA, single head): hidden → wkv → kv_norm → [head_dim]
    float *kv = calloc((size_t)head_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.attn.wkv", layer_idx);
    q4_proj(kv, hidden, model, gpu, base, head_dim, H);

    float *kv_norm_w = calloc((size_t)head_dim, sizeof(float));
    snprintf(wname, sizeof(wname), "layers.%d.attn.kv_norm.weight", layer_idx);
    load_bf16(kv_norm_w, model, wname, head_dim);
    {
        float ss = 0.0f;
#ifdef PLATFORM_MACOS
        vDSP_svesq(kv, 1, &ss, (vDSP_Length)head_dim);
#else
        for (int d = 0; d < head_dim; d++) ss += kv[d] * kv[d];
#endif
        float rms = sqrtf(ss / (float)head_dim + eps);
        for (int d = 0; d < head_dim; d++)
            kv[d] = (kv[d] / rms) * kv_norm_w[d];
    }
    free(kv_norm_w);

    // 5. RoPE on KV (last qk_rope_dim dims). V4 MLA uses the SAME RoPE'd kv
    // as both K (scoring) and V (values); the output's last rope_dim slice is
    // then un-rotated by the inverse-RoPE step after attention.
    for (int i = 0; i < qk_rope_dim / 2; i++) {
        float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)qk_rope_dim);
        float angle = (float)position * freq;
        float c = cosf(angle), s = sinf(angle);
        int idx = nope_dim + i * 2;
        float x0 = kv[idx], x1 = kv[idx + 1];
        kv[idx]     = x0 * c - x1 * s;
        kv[idx + 1] = x0 * s + x1 * c;
    }

    // 6. Cache: K and V are the same RoPE'd vector.
    cache_kv_append(cache, kv_layer_idx, kv, kv);
    free(kv);

    const float *cached_k, *cached_v;
    int seq_len;
    cache_kv_get(cache, kv_layer_idx, &cached_k, &cached_v, &seq_len);

    // 7. Attention: all heads attend to single KV head (MQA), with V4's
    // learnable per-head attention sink (sparse_attn_kernel in reference adds
    // exp(attn_sink[h] - scores_max) to the softmax denominator only — the
    // sink contributes nothing to the numerator).
    float scale = 1.0f / sqrtf((float)head_dim);

    int window_start = 0;
    if (seq_len > window_size)
        window_start = seq_len - window_size;

    char sink_name[128];
    snprintf(sink_name, sizeof(sink_name), "layers.%d.attn.attn_sink", layer_idx);
    size_t sink_sz;
    const void *attn_sink_raw = model_get_weight(model, sink_name, &sink_sz);
    const float *attn_sink = (const float *)attn_sink_raw;

    float *head_out = calloc((size_t)q_dim, sizeof(float));
    float *scores = calloc((size_t)seq_len, sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        const float *qi = q + h * head_dim;

        for (int t = window_start; t < seq_len; t++) {
            const float *kt = cached_k + (size_t)t * (size_t)head_dim;
            float dot = 0.0f;
#ifdef PLATFORM_MACOS
            vDSP_dotpr(qi, 1, kt, 1, &dot, (vDSP_Length)head_dim);
#else
            for (int d = 0; d < head_dim; d++) dot += qi[d] * kt[d];
#endif
            scores[t] = dot * scale;
        }

        // Online softmax with attn_sink as a virtual position with zero value.
        float max_v = attn_sink ? attn_sink[h] : -1e30f;
        for (int t = window_start; t < seq_len; t++)
            if (scores[t] > max_v) max_v = scores[t];

        float denom = 0.0f;
        for (int t = window_start; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_v);
            denom += scores[t];
        }
        if (attn_sink)
            denom += expf(attn_sink[h] - max_v);   // sink only adds to denom
        float inv_denom = 1.0f / denom;
        for (int t = window_start; t < seq_len; t++)
            scores[t] *= inv_denom;

        float *out_h = head_out + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int t = window_start; t < seq_len; t++)
                sum += scores[t] * cached_v[(size_t)t * (size_t)head_dim + d];
            out_h[d] = sum;
        }
    }

    // 7b. Output-side RoPE inverse on the last qk_rope_dim of each head.
    // Reference: apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True).
    // Inverse rotation = same formulas as forward RoPE but with sin negated.
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < qk_rope_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)qk_rope_dim);
            float angle = (float)position * freq;
            float c = cosf(angle), s = sinf(angle);
            int idx = h * head_dim + nope_dim + i * 2;
            float x0 = head_out[idx], x1 = head_out[idx + 1];
            head_out[idx]     =  x0 * c + x1 * s;   // s sign flipped vs forward RoPE
            head_out[idx + 1] = -x0 * s + x1 * c;
        }
    }

    free(q);
    free(scores);

    // 8. O projection (grouped LoRA)
    // head_out [64, 512] → [8 groups of 4096] → per-group wo_a → concat → wo_b
    float *o_low = calloc((size_t)(o_groups * o_lora_rank), sizeof(float));

    {
        char wo_a_w[128], wo_a_s[128], wo_a_b[128];
        snprintf(wo_a_w, sizeof(wo_a_w), "layers.%d.attn.wo_a.weight", layer_idx);
        snprintf(wo_a_s, sizeof(wo_a_s), "layers.%d.attn.wo_a.scales", layer_idx);
        snprintf(wo_a_b, sizeof(wo_a_b), "layers.%d.attn.wo_a.biases", layer_idx);

        size_t ws, ss, bs;
        const void *w = model_get_weight(model, wo_a_w, &ws);
        const void *s = model_get_weight(model, wo_a_s, &ss);
        const void *b = model_get_weight(model, wo_a_b, &bs);

        if (w && s && b) {
            int M_g = o_lora_rank;
            int K_g = group_in_dim;
            int gs = 64;
            size_t w_row_bytes = (size_t)(K_g / 8) * sizeof(uint32_t);
            size_t s_row_bytes = (size_t)(K_g / gs) * sizeof(uint16_t);

            for (int g = 0; g < o_groups; g++) {
                const char *wg = (const char *)w + (size_t)(g * M_g) * w_row_bytes;
                const char *sg = (const char *)s + (size_t)(g * M_g) * s_row_bytes;
                const char *bg = (const char *)b + (size_t)(g * M_g) * s_row_bytes;

                dequant_matmul_q4(o_low + g * M_g, wg, sg, bg,
                                  head_out + g * K_g, M_g, K_g, gs);
            }
        } else {
            LOG_WARN("v4_mla: missing wo_a weights for layer %d", layer_idx);
        }
    }

    free(head_out);

    snprintf(base, sizeof(base), "layers.%d.attn.wo_b", layer_idx);
    q4_proj(attn_out, o_low, model, gpu, base, H, o_groups * o_lora_rank);
    free(o_low);
}
