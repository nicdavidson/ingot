#define _POSIX_C_SOURCE 200809L

#include "inference/attention.h"
#include "inference/dequant.h"
#include "util/log.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Helpers ---

static void q4_matmul_named(float *out, const float *x, const Model *model,
                            const char *base, int M, int K, int group_size) {
    char wn[128], sn[128], bn[128];
    snprintf(wn, sizeof(wn), "%s.weight", base);
    snprintf(sn, sizeof(sn), "%s.scales", base);
    snprintf(bn, sizeof(bn), "%s.biases", base);

    size_t ws, ss, bs;
    const void *w = model_get_weight(model, wn, &ws);
    const void *s = model_get_weight(model, sn, &ss);
    const void *b = model_get_weight(model, bn, &bs);

    if (!w || !s || !b) {
        memset(out, 0, (size_t)M * sizeof(float));
        return;
    }
    dequant_matmul_q4(out, w, s, b, x, M, K, group_size);
}

static void load_bf16_weight(float *out, const Model *model, const char *name, int n) {
    size_t sz;
    const void *data = model_get_weight(model, name, &sz);
    if (!data) { memset(out, 0, (size_t)n * sizeof(float)); return; }
    const uint16_t *bf = data;
    for (int i = 0; i < n; i++) out[i] = bf16_to_f32(bf[i]);
}

__attribute__((unused))
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] / rms) * w[i];
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// --- SWA (Full Attention) ---

void attention_swa_forward(
    float *attn_out, const float *hidden,
    const Model *model, const ModelConfig *cfg,
    InferenceCache *cache, int layer_idx, int kv_layer_idx, int position)
{
    int H = cfg->hidden_size;
    int num_heads = cfg->num_attention_heads;
    int num_kv_heads = cfg->num_key_value_heads;
    int head_dim = cfg->head_dim;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int rotary_dim = (int)((float)head_dim * cfg->rope.partial_rotary_factor);

    char base[128];

    // Q projection
    float *q = calloc((size_t)q_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.self_attn.q_proj", layer_idx);
    q4_matmul_named(q, hidden, model, base, q_dim, H, 64);

    // K projection
    float *k = calloc((size_t)kv_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.self_attn.k_proj", layer_idx);
    q4_matmul_named(k, hidden, model, base, kv_dim, H, 64);

    // V projection
    float *v = calloc((size_t)kv_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.self_attn.v_proj", layer_idx);
    q4_matmul_named(v, hidden, model, base, kv_dim, H, 64);

    // QK normalization
    float *q_norm_w = calloc((size_t)head_dim, sizeof(float));
    float *k_norm_w = calloc((size_t)head_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.self_attn.q_norm.weight", layer_idx);
    load_bf16_weight(q_norm_w, model, base, head_dim);
    snprintf(base, sizeof(base), "layers.%d.self_attn.k_norm.weight", layer_idx);
    load_bf16_weight(k_norm_w, model, base, head_dim);

    // Apply QK norm per-head
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

    // RoPE
    float theta_base = (float)cfg->rope.rope_theta;
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)rotary_dim);
            float angle = (float)position * freq;
            float c = cosf(angle), s_val = sinf(angle);
            int idx = h * head_dim + i * 2;
            float x0 = q[idx], x1 = q[idx + 1];
            q[idx]     = x0 * c - x1 * s_val;
            q[idx + 1] = x0 * s_val + x1 * c;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)rotary_dim);
            float angle = (float)position * freq;
            float c = cosf(angle), s_val = sinf(angle);
            int idx = h * head_dim + i * 2;
            float x0 = k[idx], x1 = k[idx + 1];
            k[idx]     = x0 * c - x1 * s_val;
            k[idx + 1] = x0 * s_val + x1 * c;
        }
    }

    // Cache KV
    cache_kv_append(cache, kv_layer_idx, k, v);

    // Get cached K, V
    const float *cached_k, *cached_v;
    int seq_len;
    cache_kv_get(cache, kv_layer_idx, &cached_k, &cached_v, &seq_len);

    // Attention: Q @ K^T → softmax → @ V
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_group = num_heads / num_kv_heads;
    float *head_out = calloc((size_t)q_dim, sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / kv_group;
        const float *qi = q + h * head_dim;

        // Compute scores
        float *scores = calloc((size_t)seq_len, sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            const float *ki = cached_k + t * num_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qi[d] * ki[d];
            scores[t] = dot * scale;
        }

        softmax(scores, seq_len);

        // Weighted V
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                sum += scores[t] * cached_v[t * num_kv_heads * head_dim + kv_h * head_dim + d];
            }
            head_out[h * head_dim + d] = sum;
        }
        free(scores);
    }

    // Output projection
    snprintf(base, sizeof(base), "layers.%d.self_attn.o_proj", layer_idx);
    q4_matmul_named(attn_out, head_out, model, base, H, q_dim, 64);

    free(q); free(k); free(v);
    free(q_norm_w); free(k_norm_w);
    free(head_out);
}

// --- DeltaNet / Mamba-style Linear Attention ---
//
// Simplified implementation: uses the in_proj_qkv projection to transform
// the hidden state, applies a basic gated pass, and projects back.
// Full Mamba SSM state tracking is deferred to Metal optimization.

void attention_deltanet_forward(
    float *attn_out, const float *hidden,
    const Model *model, const ModelConfig *cfg,
    InferenceCache *cache, int layer_idx, int dn_layer_idx, int position)
{
    (void)cache; (void)dn_layer_idx; (void)position;

    int H = cfg->hidden_size;
    char base[128];

    // in_proj_qkv: [12288, 4096] → projects hidden to 3x expanded state
    // This is a fused QKV projection for the Mamba block
    int qkv_dim = cfg->linear_attn.linear_num_key_heads *
                  cfg->linear_attn.linear_key_head_dim * 3;
    // Fallback: check actual weight shape
    char wname[128];
    snprintf(wname, sizeof(wname), "layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    size_t ws;
    const void *w_check = model_get_weight(model, wname, &ws);
    if (!w_check) {
        memset(attn_out, 0, (size_t)H * sizeof(float));
        return;
    }

    // Determine output dim from weight size: shape[0] is output dim
    // Weight is packed [M, K/8] U32, so M = shape[0]
    // From the index we know it's [12288, 512] → 12288 output dim
    qkv_dim = 12288; // TODO: read from weight index

    float *qkv = calloc((size_t)qkv_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.linear_attn.in_proj_qkv", layer_idx);
    q4_matmul_named(qkv, hidden, model, base, qkv_dim, H, 64);

    // in_proj_z: gate projection [8192, 4096]
    int z_dim = 8192;
    float *z = calloc((size_t)z_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.linear_attn.in_proj_z", layer_idx);
    q4_matmul_named(z, hidden, model, base, z_dim, H, 64);

    // Apply SiLU gate: z = silu(z)
    for (int i = 0; i < z_dim; i++) {
        z[i] = z[i] / (1.0f + expf(-z[i]));
    }

    // Simplified: take the first H elements of qkv, multiply by gate,
    // and project back through out_proj.
    // This is NOT a proper Mamba implementation — it's a gated linear pass
    // that at least transforms the hidden state through the layer's weights.
    float *gated = calloc((size_t)z_dim, sizeof(float));

    // Normalize with the layer's norm weight
    float *norm_w = calloc(128, sizeof(float)); // linear_attn.norm is [128] BF16
    snprintf(base, sizeof(base), "layers.%d.linear_attn.norm.weight", layer_idx);
    load_bf16_weight(norm_w, model, base, 128);

    // Simple gated output: take z and multiply elementwise
    for (int i = 0; i < z_dim; i++) {
        // Use qkv values modulated by z gate
        float qkv_val = (i < qkv_dim) ? qkv[i] : 0.0f;
        gated[i] = qkv_val * z[i];
    }

    // Output projection: [4096, 8192]
    snprintf(base, sizeof(base), "layers.%d.linear_attn.out_proj", layer_idx);
    q4_matmul_named(attn_out, gated, model, base, H, z_dim, 64);

    free(qkv); free(z); free(gated); free(norm_w);
}
