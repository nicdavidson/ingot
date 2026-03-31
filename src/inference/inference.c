#define _POSIX_C_SOURCE 200809L

#include "inference/inference.h"
#include "inference/kv_cache.h"
#include "inference/sampler.h"
#include "config/config.h"
#include "util/log.h"
#include "util/timer.h"
#include "util/arena.h"

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

// CPU reference implementations for the forward pass.
// These will be replaced by Metal kernel calls on macOS,
// but serve as ground truth for testing.

// CPU reference implementations — used until Metal kernels are wired in.
// Suppress unused warnings for now; these are called from forward_layer()
// once weight loading is implemented.
__attribute__((unused))
static void cpu_rmsnorm(float *out, const float *x, const float *weight,
                        int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] / rms) * weight[i];
}

__attribute__((unused))
static void cpu_matmul(float *out, const float *a, const float *x,
                       int M, int K) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            sum += a[i * K + j] * x[j];
        }
        out[i] = sum;
    }
}

__attribute__((unused))
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

__attribute__((unused))
static void cpu_silu(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
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

// Forward pass for a single layer.
// This is a placeholder that shows the correct data flow.
// The actual weight lookups depend on the weight file format.
static void forward_layer(InferenceContext *ctx, int layer_idx) {
    (void)ctx;
    (void)layer_idx;
    // TODO: implement when weight format is finalized
    // The flow is:
    //
    // 1. pre_attn_norm = rmsnorm(hidden, attn_norm_weight)
    // 2. q = pre_attn_norm @ wq
    // 3. k = pre_attn_norm @ wk
    // 4. v = pre_attn_norm @ wv
    //
    // If full_attention layer:
    //   5a. rope(q, k)
    //   6a. cache_kv_append(k, v)
    //   7a. attn_out = attention(q, cached_k, cached_v)
    //
    // If linear_attention layer:
    //   5b. beta = sigmoid(pre_attn_norm @ w_gate)
    //   6b. deltanet_update(S, k, v, beta)
    //   7b. attn_out = deltanet_query(S, q)
    //
    // 8. if attn_output_gate: attn_out *= sigmoid(pre_attn_norm @ w_output_gate)
    // 9. hidden += attn_out @ wo
    //
    // MoE:
    // 10. pre_moe_norm = rmsnorm(hidden, moe_norm_weight)
    // 11. gate_logits = pre_moe_norm @ gate_weight
    // 12. top_k_experts, weights = topk(softmax(gate_logits))
    // 13. shared_expert_out = ffn(pre_moe_norm, shared_expert_weights)
    // 14. for each selected expert:
    //       expert_out += weight[k] * ffn(pre_moe_norm, expert_weights[k])
    // 15. hidden += shared_expert_out + expert_out
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
        // TODO: embedding lookup + forward pass
        // For now this is a stub — the actual implementation needs
        // the weight file format to be finalized.
        ctx->position = i;

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

        // Forward pass
        for (int l = 0; l < cfg->num_hidden_layers; l++) {
            forward_layer(ctx, l);
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
