#define _POSIX_C_SOURCE 200809L

#include "inference/inference.h"
#include "inference/kv_cache.h"
#include "inference/sampler.h"
#include "inference/dequant.h"
#include "config/config.h"
#include "util/log.h"
#include "util/timer.h"
#include "util/arena.h"

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

    LOG_INFO("inference: context created (max_seq=%d)", max_seq);
    return ctx;
}

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
static void q4_matmul(InferenceContext *ctx, float *out, const float *x,
                      const char *w_name, const char *s_name, const char *b_name,
                      int M, int K, int group_size) {
    size_t ws, ss, bs;
    const void *w = model_get_weight(ctx->model, w_name, &ws);
    const void *s = model_get_weight(ctx->model, s_name, &ss);
    const void *b = model_get_weight(ctx->model, b_name, &bs);

    if (!w || !s || !b) {
        // Weight not found — zero output
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

    // 2-4. Attention projections (Q, K, V)
    // Note: for now, skip actual attention since weight naming isn't finalized
    // We'll run the MoE path to test expert weight loading first

    // Skip to MoE (just add residual back for now)
    // This will be filled in when attention weights are properly mapped

    // 10. Post-attention layernorm
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

    // 13. Shared expert FFN
    int moe_dim = cfg->shared_expert_intermediate_size;
    float *gate_out = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
    float *up_out   = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
    float *ffn_mid  = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));

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

    // Add shared expert output to hidden
    for (int i = 0; i < H; i++) s->hidden[i] = s->residual[i] + s->expert_out[i];

    // 14. Routed experts
    moe_dim = cfg->moe_intermediate_size;
    if (moe_dim != cfg->shared_expert_intermediate_size) {
        gate_out = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
        up_out   = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
        ffn_mid  = arena_alloc(&ctx->arena, (size_t)moe_dim * sizeof(float));
    }

    for (int k = 0; k < K_active; k++) {
        int expert_idx = top_indices[k];
        float weight = top_weights[k];

        // Get expert data from mmap'd file
        size_t stride;
        const void *expert_data = model_get_expert(ctx->model, layer_idx,
                                                    expert_idx, &stride);
        if (!expert_data) continue;

        // Expert layout: gate_proj(w,s,b), up_proj(w,s,b), down_proj(w,s,b)
        // Each projection: weight[moe_dim, H/8] + scales[moe_dim, G] + biases[moe_dim, G]
        int group_size = 64;
        int num_groups = H / group_size;
        size_t w_size = (size_t)moe_dim * (size_t)(H / 8) * 4;       // U32
        size_t s_size = (size_t)moe_dim * (size_t)num_groups * 2;     // BF16
        size_t b_size = s_size;

        const char *ptr = expert_data;

        // gate_proj
        const void *gw = ptr; ptr += w_size;
        const void *gs = ptr; ptr += s_size;
        const void *gb = ptr; ptr += b_size;
        dequant_matmul_q4(gate_out, gw, gs, gb, s->norm_out, moe_dim, H, group_size);

        // up_proj
        const void *uw = ptr; ptr += w_size;
        const void *us = ptr; ptr += s_size;
        const void *ub = ptr; ptr += b_size;
        dequant_matmul_q4(up_out, uw, us, ub, s->norm_out, moe_dim, H, group_size);

        // SiLU(gate) * up
        cpu_silu_mul(ffn_mid, gate_out, up_out, moe_dim);

        // down_proj: [H, moe_dim]
        int down_groups = moe_dim / group_size;
        size_t dw_size = (size_t)H * (size_t)(moe_dim / 8) * 4;
        size_t ds_size = (size_t)H * (size_t)down_groups * 2;
        (void)dw_size;

        const void *dw = ptr; ptr += dw_size;
        const void *ds = ptr; ptr += ds_size;
        const void *db = ptr; // remaining
        (void)db;

        float *expert_result = arena_alloc(&ctx->arena, (size_t)H * sizeof(float));
        dequant_matmul_q4(expert_result, dw, ds, ptr, ffn_mid, H, moe_dim, group_size);

        // Weighted add to hidden
        for (int i = 0; i < H; i++) {
            s->hidden[i] += weight * expert_result[i];
        }
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
    q4_matmul(ctx, ctx->scratch.logits, ctx->scratch.hidden,
              "lm_head.weight", "lm_head.scales", "lm_head.biases",
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
        embed_token(ctx, prompt_tokens[i]);

        for (int l = 0; l < cfg->num_hidden_layers; l++) {
            forward_layer(ctx, l);
        }
    }

    uint64_t t_prefill = timer_now_ns();
    LOG_INFO("inference: prefill done in %.1f ms",
             timer_elapsed_ms(t_start, t_prefill));

    // Generate tokens
    int32_t next_token = prompt_tokens[num_prompt_tokens - 1];

    for (int t = 0; t < max_tokens; t++) {
        ctx->position = num_prompt_tokens + t;

        embed_token(ctx, next_token);

        // Forward pass through all layers
        for (int l = 0; l < cfg->num_hidden_layers; l++) {
            forward_layer(ctx, l);
        }

        // Project to vocab logits
        compute_logits(ctx);

        // Sample next token from logits
        next_token = sampler_sample(sampler, ctx->scratch.logits, cfg->vocab_size);
        sampler_accept(sampler, next_token);
        generated++;

        // Decode and callback
        const char *text = tokenizer_decode(tok, next_token);
        if (callback) {
            callback(next_token, text, userdata);
        }

        // Stop on EOS
        if (next_token == tokenizer_eos_id(tok)) {
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
    cache_free(ctx->cache);
    free_scratch(&ctx->scratch);
    arena_destroy(&ctx->arena);
    free(ctx);
}
