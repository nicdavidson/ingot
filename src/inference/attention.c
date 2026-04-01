#define _POSIX_C_SOURCE 200809L

#include "inference/attention.h"
#include "inference/dequant.h"
#include "util/log.h"

#ifdef PLATFORM_MACOS
#include <Accelerate/Accelerate.h>
#include "compute/kernels.h"
#include "compute/metal_context.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Helpers ---

static void q4_proj(float *out, const float *x, const Model *model,
                    const char *base, int M, int K) {
    char wn[128], sn[128], bn[128];
    snprintf(wn, sizeof(wn), "%s.weight", base);
    snprintf(sn, sizeof(sn), "%s.scales", base);
    snprintf(bn, sizeof(bn), "%s.biases", base);

#ifdef PLATFORM_MACOS
    // GPU dispatch via temporary Metal buffers.
    // We allocate temp GPU buffers because the input may already live inside
    // another Metal buffer (unified memory), so newBufferWithBytesNoCopy fails.
    MetalContext *metal = model_get_metal(model);
    void *shared_buf = model_get_metal_shared_buf(model);
    if (metal && shared_buf) {
        long w_off = model_get_weight_offset(model, wn);
        long s_off = model_get_weight_offset(model, sn);
        long b_off = model_get_weight_offset(model, bn);
        if (w_off >= 0 && s_off >= 0 && b_off >= 0) {
            void *x_gpu = metal_alloc_buffer(metal, (size_t)K * sizeof(float));
            void *out_gpu = metal_alloc_buffer(metal, (size_t)M * sizeof(float));
            if (x_gpu && out_gpu) {
                memcpy(metal_buffer_contents(x_gpu), x, (size_t)K * sizeof(float));
                kernel_matmul_q4_fma_offsets(metal, shared_buf,
                                             (size_t)w_off, (size_t)s_off, (size_t)b_off,
                                             x_gpu, out_gpu,
                                             (uint32_t)M, (uint32_t)K, 64);
                memcpy(out, metal_buffer_contents(out_gpu), (size_t)M * sizeof(float));
                metal_free_buffer(x_gpu);
                metal_free_buffer(out_gpu);
                return;
            }
            if (x_gpu) metal_free_buffer(x_gpu);
            if (out_gpu) metal_free_buffer(out_gpu);
            LOG_ERROR("attn q4_proj: GPU alloc failed for %s (M=%d K=%d)", base, M, K);
        } else {
            LOG_ERROR("attn q4_proj: weight offset missing for %s (w=%ld s=%ld b=%ld)",
                      base, w_off, s_off, b_off);
        }
    } else {
        static int warn_count = 0;
        if (warn_count++ < 3)
            LOG_ERROR("attn q4_proj: no metal=%p shared_buf=%p", (void *)metal, shared_buf);
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

static void load_bf16(float *out, const Model *model, const char *name, int n) {
    size_t sz;
    const void *data = model_get_weight(model, name, &sz);
    if (!data) { memset(out, 0, (size_t)n * sizeof(float)); return; }
    const uint16_t *bf = data;
    for (int i = 0; i < n; i++) out[i] = bf16_to_f32(bf[i]);
}

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
    InferenceCache *cache, int layer_idx, int kv_layer_idx, int position)
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

    float *q_full = calloc((size_t)q_proj_dim, sizeof(float));
    float *k = calloc((size_t)kv_dim, sizeof(float));
    float *v = calloc((size_t)kv_dim, sizeof(float));

    // Q/K/V projections
    snprintf(base, sizeof(base), "layers.%d.self_attn.q_proj", layer_idx);
    q4_proj(q_full, hidden, model, base, q_proj_dim, H);
    snprintf(base, sizeof(base), "layers.%d.self_attn.k_proj", layer_idx);
    q4_proj(k, hidden, model, base, kv_dim, H);
    snprintf(base, sizeof(base), "layers.%d.self_attn.v_proj", layer_idx);
    q4_proj(v, hidden, model, base, kv_dim, H);

    // Split q_proj into query and gate
    // q_full is laid out as [num_heads, head_dim*2] when gated
    // We need to split each head's chunk: first head_dim = query, second = gate
    float *q = calloc((size_t)q_dim, sizeof(float));
    float *gate = cfg->attn_output_gate ? calloc((size_t)q_dim, sizeof(float)) : NULL;

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
    free(q_full);

    // QK normalization (per-head RMSNorm)
    float *q_norm_w = calloc((size_t)head_dim, sizeof(float));
    float *k_norm_w = calloc((size_t)head_dim, sizeof(float));
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
    float *head_out = calloc((size_t)q_dim, sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / kv_group;
        const float *qi = q + h * head_dim;

        float *scores = calloc((size_t)seq_len, sizeof(float));
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
        free(scores);
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
    q4_proj(attn_out, head_out, model, base, H, q_dim);

    free(q); free(k); free(v);
    free(q_norm_w); free(k_norm_w);
    free(head_out);
    free(gate);
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

void attention_deltanet_forward(
    float *attn_out, const float *hidden,
    const Model *model, const ModelConfig *cfg,
    InferenceCache *cache, int layer_idx, int dn_layer_idx, int position)
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

    char base[128], wname[128];

    // 1. QKV projection: hidden → [key_dim + key_dim + value_dim]
    float *qkv = calloc((size_t)conv_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.linear_attn.in_proj_qkv", layer_idx);
    q4_proj(qkv, hidden, model, base, conv_dim, H);

    // 2. Causal conv1d with state tracking
    // Conv1d weight is [conv_dim, kernel_size, 1] BF16 (depthwise)
    // For single-token: shift conv_state left, append new qkv, convolve, SiLU
    int kernel_size = cfg->linear_attn.linear_conv_kernel_dim; // typically 4
    float *conv_weight = calloc((size_t)conv_dim * (size_t)kernel_size, sizeof(float));
    snprintf(wname, sizeof(wname), "layers.%d.linear_attn.conv1d.weight", layer_idx);
    load_bf16(conv_weight, model, wname, conv_dim * kernel_size);

    float *conv_state = cache_dn_conv_get(cache, dn_layer_idx);
    int state_len = kernel_size - 1; // conv_state holds last (kernel_size-1) values

    // For each channel: convolve over [conv_state..., current_qkv]
    for (int c = 0; c < conv_dim; c++) {
        float *cs = conv_state + c * state_len;
        float *cw = conv_weight + c * kernel_size;

        // Convolution: sum over kernel window
        float sum = 0.0f;
        for (int k = 0; k < state_len; k++) {
            sum += cs[k] * cw[k];
        }
        sum += qkv[c] * cw[state_len]; // current value × last kernel weight

        // Shift state: drop oldest, append current
        for (int k = 0; k < state_len - 1; k++) {
            cs[k] = cs[k + 1];
        }
        if (state_len > 0) cs[state_len - 1] = qkv[c];

        // SiLU activation
        qkv[c] = sum / (1.0f + expf(-sum));
    }

    // 3. Split QKV
    float *query = qkv;                          // [key_dim]
    float *key = qkv + key_dim;                  // [key_dim]
    float *value = qkv + key_dim * 2;            // [value_dim]

    // 4. Compute gates
    // beta = sigmoid(in_proj_b(hidden))
    float *b_raw = calloc((size_t)num_v_heads, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.linear_attn.in_proj_b", layer_idx);
    q4_proj(b_raw, hidden, model, base, num_v_heads, H);
    float *beta = calloc((size_t)num_v_heads, sizeof(float));
    for (int i = 0; i < num_v_heads; i++) {
        beta[i] = 1.0f / (1.0f + expf(-b_raw[i])); // sigmoid
    }

    // g = -exp(A_log) * softplus(in_proj_a(hidden) + dt_bias)
    float *a_raw = calloc((size_t)num_v_heads, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.linear_attn.in_proj_a", layer_idx);
    q4_proj(a_raw, hidden, model, base, num_v_heads, H);

    float *A_log = calloc((size_t)num_v_heads, sizeof(float));
    snprintf(wname, sizeof(wname), "layers.%d.linear_attn.A_log", layer_idx);
    load_f32(A_log, model, wname, num_v_heads);

    float *dt_bias_w = calloc((size_t)num_v_heads, sizeof(float));
    snprintf(wname, sizeof(wname), "layers.%d.linear_attn.dt_bias", layer_idx);
    load_bf16(dt_bias_w, model, wname, num_v_heads);

    float *g = calloc((size_t)num_v_heads, sizeof(float));
    for (int i = 0; i < num_v_heads; i++) {
        g[i] = -expf(A_log[i]) * softplus(a_raw[i] + dt_bias_w[i]);
    }

    // 5. Z gate for output gating
    int z_dim = num_v_heads * head_v_dim;
    float *z = calloc((size_t)z_dim, sizeof(float));
    snprintf(base, sizeof(base), "layers.%d.linear_attn.in_proj_z", layer_idx);
    q4_proj(z, hidden, model, base, z_dim, H);

    // 6. L2 normalize Q and K per head
    // Note: use_qk_l2norm_in_kernel=True is the default for Qwen 3.5
    for (int h = 0; h < num_k_heads; h++) {
        l2norm(query + h * head_k_dim, head_k_dim);
        l2norm(key + h * head_k_dim, head_k_dim);
    }

    // Prefetch expert weights for the next MoE layer (optimization hint)
    // This happens during attention compute so the expert pages are ready

    // 7. Scale Q
    float q_scale = 1.0f / sqrtf((float)head_k_dim);
    for (int i = 0; i < key_dim; i++) query[i] *= q_scale;

    // 8. Recurrent state update (per head)
    // State S is [num_k_heads, head_k_dim, head_v_dim]
    // But HF uses num_v_heads for the outer loop — heads may be grouped
    // For Qwen 3.5: num_k_heads=16, num_v_heads=64, head_k_dim=128, head_v_dim=128
    // This means 4 value heads per key head

    float *state = cache_dn_get(cache, dn_layer_idx);
    float *core_out = calloc((size_t)value_dim, sizeof(float));

    int v_per_k = num_v_heads / num_k_heads; // value heads per key head
    int state_size = head_k_dim * head_v_dim;

    for (int kh = 0; kh < num_k_heads; kh++) {
        float *q_h = query + kh * head_k_dim;
        float *k_h = key + kh * head_k_dim;

        for (int vh_local = 0; vh_local < v_per_k; vh_local++) {
            int vh = kh * v_per_k + vh_local;
            float *v_h = value + vh * head_v_dim;
            float g_h = g[vh];
            float beta_h = beta[vh];

            // State for this head pair: S[head_k_dim, head_v_dim] row-major
            float *S = state + ((size_t)kh * v_per_k + vh_local) *
                       (size_t)state_size;

            float decay = expf(g_h);

#ifdef PLATFORM_MACOS
            // Accelerate BLAS path — avoids GPU kernel launch overhead
            // for these small dense matrices (128x128)

            // Decay state: S *= decay
            cblas_sscal(state_size, decay, S, 1);

            // Read from state: kv_mem = S^T @ k
            // S is [head_k_dim, head_v_dim] row-major
            // We want kv_mem[vd] = sum_kd(S[kd][vd] * k[kd]) = S^T @ k
            float *kv_mem = calloc((size_t)head_v_dim, sizeof(float));
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        head_k_dim, head_v_dim,
                        1.0f, S, head_v_dim,
                        k_h, 1,
                        0.0f, kv_mem, 1);

            // Delta: (v - kv_mem) * beta
            float *delta = calloc((size_t)head_v_dim, sizeof(float));
            for (int vd = 0; vd < head_v_dim; vd++) {
                delta[vd] = (v_h[vd] - kv_mem[vd]) * beta_h;
            }

            // Write to state: S += k outer delta
            cblas_sger(CblasRowMajor,
                       head_k_dim, head_v_dim,
                       1.0f, k_h, 1, delta, 1,
                       S, head_v_dim);

            // Query state: out = S^T @ q
            float *out_h = core_out + vh * head_v_dim;
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        head_k_dim, head_v_dim,
                        1.0f, S, head_v_dim,
                        q_h, 1,
                        1.0f, out_h, 1);  // accumulate (beta=1)

            free(kv_mem);
            free(delta);
#else
            // Portable scalar fallback for non-Apple platforms

            // Decay state
            for (int i = 0; i < state_size; i++) {
                S[i] *= decay;
            }

            // Read from state: kv_mem[vd] = sum_kd(S[kd][vd] * k[kd])
            float *kv_mem = calloc((size_t)head_v_dim, sizeof(float));
            for (int kd = 0; kd < head_k_dim; kd++) {
                for (int vd = 0; vd < head_v_dim; vd++) {
                    kv_mem[vd] += S[kd * head_v_dim + vd] * k_h[kd];
                }
            }

            // Delta: (v - kv_mem) * beta
            float *delta = calloc((size_t)head_v_dim, sizeof(float));
            for (int vd = 0; vd < head_v_dim; vd++) {
                delta[vd] = (v_h[vd] - kv_mem[vd]) * beta_h;
            }

            // Write to state: S += k.outer(delta)
            for (int kd = 0; kd < head_k_dim; kd++) {
                for (int vd = 0; vd < head_v_dim; vd++) {
                    S[kd * head_v_dim + vd] += k_h[kd] * delta[vd];
                }
            }

            // Query state: out[vd] = sum_kd(S[kd][vd] * q[kd])
            float *out_h = core_out + vh * head_v_dim;
            for (int kd = 0; kd < head_k_dim; kd++) {
                for (int vd = 0; vd < head_v_dim; vd++) {
                    out_h[vd] += S[kd * head_v_dim + vd] * q_h[kd];
                }
            }

            free(kv_mem);
            free(delta);
#endif
        }
    }

    // 9. RMSNormGated: rmsnorm(core_out) * silu(z)
    float *norm_w = calloc((size_t)head_v_dim, sizeof(float));
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

    // 10. Output projection
    snprintf(base, sizeof(base), "layers.%d.linear_attn.out_proj", layer_idx);
    q4_proj(attn_out, core_out, model, base, H, value_dim);

    free(qkv); free(b_raw); free(beta); free(a_raw);
    free(A_log); free(dt_bias_w); free(g); free(z);
    free(core_out); free(norm_w); free(conv_weight);
}
