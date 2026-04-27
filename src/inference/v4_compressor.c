// V4 Compressor — produces compressed KV positions for layers with
// compress_ratio > 0. Mirrors DeepSeek-V4-Flash inference/model.py::Compressor.
//
// CSA layers (compress_ratio=4): overlap=true. wkv/wgate output 2*head_dim;
// state buffers are [2*ratio=8, 2*head_dim=1024]. Each compressed token
// pulls "first half" dims from the previous window and "second half" dims
// from the current window (overlap_transform in the reference).
//
// HCA layers (compress_ratio=128): overlap=false. wkv/wgate output head_dim;
// state buffers are [ratio=128, head_dim=512]. Plain block-pool every 128
// tokens. For prompts under 128 tokens this never fires.
//
// Caching: each compressor holds its own [cache_capacity, head_dim] buffer
// of compressed KV vectors (RoPE-rotated). Attention reads these alongside
// the windowed KV cache.

#define _POSIX_C_SOURCE 200809L

#include "inference/v4_compressor.h"
#include "inference/dequant.h"
#include "util/log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PLATFORM_MACOS
#include <Accelerate/Accelerate.h>
#endif

// bf16 → f32 (matches dequant.c). Inline copy avoids exposing a header.
static inline float bf16_to_f32_local(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

typedef struct {
    int  compress_ratio;        // 4 (CSA) or 128 (HCA), 0 if layer has none
    int  state_slots;           // coff*ratio: 8 (CSA) or 128 (HCA)
    int  proj_dim;              // coff*head_dim: 1024 (CSA) or 512 (HCA)
    int  head_dim;              // 512
    int  rope_head_dim;         // 64
    bool overlap;               // ratio == 4

    // Rolling state buffers, sized [state_slots, proj_dim]. CSA uses
    // [0:ratio] for the "previous window" overlap stash and [ratio:2*ratio]
    // for the current window; HCA uses [0:ratio] only.
    float *kv_state;
    float *score_state;

    // Compressed KV cache: [cache_capacity, head_dim], RoPE-rotated.
    float *cache;
    int    cache_capacity;
    int    cache_count;
} LayerCompressor;

struct V4Compressor {
    int               num_layers;
    int               head_dim;
    int               rope_head_dim;
    int               hidden_size;
    int               max_seq_len;
    LayerCompressor  *layers;   // [num_layers]
    double            csa_rope_theta;
};

// ---- Helpers ----

static void rmsnorm_with_weight(float *out, const float *x,
                                const float *w, int n, float eps) {
    float ss = 0.0f;
#ifdef PLATFORM_MACOS
    vDSP_svesq(x, 1, &ss, (vDSP_Length)n);
#else
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
#endif
    float inv_rms = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * w[i];
}

// Q4 affine matmul wrapper (Qwen-style int4 + BF16 affine — used by all
// compressor weights, which sit in model_weights.bin alongside other shared
// quant). out[M] = dequant(W) @ x[K]. Falls back to zeros if any of the
// triplet is missing.
static void compressor_q4_proj(const Model *model, const char *base,
                               float *out, const float *x, int M, int K) {
    char wn[160], sn[160], bn[160];
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
    dequant_matmul_q4(out, w, s, b, x, M, K, 64);
}

// Apply rotary embedding to the last `rope_dim` of one head_dim-vector.
// Forward rotation (matches v4_forward / attention_v4_mla_forward).
static void apply_rope_inplace(float *kv, int head_dim, int rope_dim,
                               int position, float theta_base) {
    int nope_dim = head_dim - rope_dim;
    for (int i = 0; i < rope_dim / 2; i++) {
        float freq = 1.0f / powf(theta_base, (float)(i * 2) / (float)rope_dim);
        float angle = (float)position * freq;
        float c = cosf(angle), s = sinf(angle);
        int idx = nope_dim + i * 2;
        float x0 = kv[idx], x1 = kv[idx + 1];
        kv[idx]     = x0 * c - x1 * s;
        kv[idx + 1] = x0 * s + x1 * c;
    }
}

// Numerically stable softmax over `n` floats in-place. Treats -inf as
// contributing 0 (already true for std exp/-inf, but be explicit).
static void softmax_inplace(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) x[i] *= inv;
    }
}

// ---- Lifecycle ----

V4Compressor *v4_compressor_create(const ModelConfig *cfg, int max_seq_len) {
    if (cfg->arch != ARCH_DEEPSEEK_V4) return NULL;

    V4Compressor *c = calloc(1, sizeof(V4Compressor));
    c->num_layers    = cfg->num_hidden_layers;
    c->head_dim      = cfg->head_dim;
    c->rope_head_dim = cfg->v4.qk_rope_head_dim;
    c->hidden_size   = cfg->hidden_size;
    c->max_seq_len   = max_seq_len;
    c->csa_rope_theta = cfg->v4.compress_rope_theta;
    c->layers        = calloc((size_t)c->num_layers, sizeof(LayerCompressor));

    int total_layers = 0, total_kb = 0;
    for (int l = 0; l < c->num_layers; l++) {
        int ratio = cfg->v4.compress_ratios ? cfg->v4.compress_ratios[l] : 0;
        if (ratio <= 0) continue;
        LayerCompressor *lc = &c->layers[l];
        lc->compress_ratio = ratio;
        lc->head_dim       = c->head_dim;
        lc->rope_head_dim  = c->rope_head_dim;
        lc->overlap        = (ratio == 4);
        int coff           = lc->overlap ? 2 : 1;
        lc->state_slots    = coff * ratio;
        lc->proj_dim       = coff * c->head_dim;
        lc->cache_capacity = max_seq_len / ratio + 1;

        size_t state_floats = (size_t)lc->state_slots * (size_t)lc->proj_dim;
        lc->kv_state    = calloc(state_floats, sizeof(float));
        lc->score_state = malloc(state_floats * sizeof(float));
        // score_state must start at -inf so untouched slots don't pollute the
        // softmax over present-only positions.
        for (size_t i = 0; i < state_floats; i++) lc->score_state[i] = -INFINITY;

        size_t cache_floats = (size_t)lc->cache_capacity * (size_t)c->head_dim;
        lc->cache = calloc(cache_floats, sizeof(float));

        total_layers++;
        total_kb += (int)((state_floats * 2 + cache_floats) * sizeof(float) / 1024);
    }
    LOG_INFO("v4_compressor: %d layers, %d KB total state+cache (max_seq=%d)",
             total_layers, total_kb, max_seq_len);
    return c;
}

void v4_compressor_reset(V4Compressor *c) {
    if (!c) return;
    for (int l = 0; l < c->num_layers; l++) {
        LayerCompressor *lc = &c->layers[l];
        if (lc->compress_ratio == 0) continue;
        size_t state_floats = (size_t)lc->state_slots * (size_t)lc->proj_dim;
        memset(lc->kv_state, 0, state_floats * sizeof(float));
        for (size_t i = 0; i < state_floats; i++) lc->score_state[i] = -INFINITY;
        lc->cache_count = 0;
    }
}

void v4_compressor_free(V4Compressor *c) {
    if (!c) return;
    for (int l = 0; l < c->num_layers; l++) {
        LayerCompressor *lc = &c->layers[l];
        free(lc->kv_state);
        free(lc->score_state);
        free(lc->cache);
    }
    free(c->layers);
    free(c);
}

// ---- Per-step forward ----

// Slice the rolling state into a flat [n_pool, head_dim] working buffer for
// the gated softmax pool. For CSA (overlap=true) the pool is built by taking
// the FIRST head_dim of state[0:ratio] (== "first half" dims of the previous
// window) followed by the SECOND head_dim of state[ratio:2*ratio] (== "second
// half" dims of the current window). For HCA (overlap=false) the pool is
// just state[0:ratio, :head_dim] — same head_dim slice as everything else.
static void build_pool(const LayerCompressor *lc,
                       const float *src,         // kv_state or score_state
                       float *dst,               // [pool_size, head_dim]
                       int pool_size,
                       float fill_pad) {
    int d = lc->head_dim;
    int proj = lc->proj_dim;
    int ratio = lc->compress_ratio;
    if (lc->overlap) {
        // First half: state[0:ratio, 0:d]
        for (int i = 0; i < ratio; i++) {
            const float *row = src + i * proj;
            for (int k = 0; k < d; k++) dst[i * d + k] = row[k];
        }
        // Second half: state[ratio:2*ratio, d:2d]
        for (int i = 0; i < ratio; i++) {
            const float *row = src + (ratio + i) * proj + d;
            for (int k = 0; k < d; k++) dst[(ratio + i) * d + k] = row[k];
        }
    } else {
        for (int i = 0; i < pool_size; i++) {
            const float *row = src + i * proj;
            for (int k = 0; k < d; k++) dst[i * d + k] = row[k];
        }
    }
    (void)fill_pad;
}

int v4_compressor_step(V4Compressor *c, const Model *model,
                       const ModelConfig *cfg, int layer_idx,
                       const float *hidden_post_norm, int position) {
    if (!c) return 0;
    LayerCompressor *lc = &c->layers[layer_idx];
    if (lc->compress_ratio == 0) return 0;

    int H = c->hidden_size;
    int D = lc->proj_dim;        // 1024 (CSA) or 512 (HCA)
    int head_dim = lc->head_dim;
    int rope_dim = lc->rope_head_dim;
    int ratio = lc->compress_ratio;
    int slot_in_window = position % ratio;
    bool should_compress = ((position + 1) % ratio) == 0;

    // 1. Project hidden → kv [D] and score [D].
    char base[160];
    snprintf(base, sizeof(base), "layers.%d.attn.compressor.wkv", layer_idx);
    float *kv = malloc((size_t)D * sizeof(float));
    compressor_q4_proj(model, base, kv, hidden_post_norm, D, H);

    snprintf(base, sizeof(base), "layers.%d.attn.compressor.wgate", layer_idx);
    float *score = malloc((size_t)D * sizeof(float));
    compressor_q4_proj(model, base, score, hidden_post_norm, D, H);

    // 2. Add ape[slot_in_window] to score (broadcast across D).
    char ape_name[160];
    snprintf(ape_name, sizeof(ape_name), "layers.%d.attn.compressor.ape", layer_idx);
    size_t ape_sz;
    const void *ape_raw = model_get_weight(model, ape_name, &ape_sz);
    if (ape_raw) {
        // ape shape: [ratio, D] BF16. Add row [slot_in_window].
        const uint16_t *ape16 = (const uint16_t *)ape_raw;
        const uint16_t *ape_row = ape16 + (size_t)slot_in_window * (size_t)D;
        for (int i = 0; i < D; i++)
            score[i] += bf16_to_f32_local(ape_row[i]);
    }

    // 3. Stash this token into the rolling state.
    int stash_slot = lc->overlap ? (ratio + slot_in_window) : slot_in_window;
    memcpy(lc->kv_state    + (size_t)stash_slot * D, kv,    (size_t)D * sizeof(float));
    memcpy(lc->score_state + (size_t)stash_slot * D, score, (size_t)D * sizeof(float));

    free(kv);
    free(score);

    if (!should_compress) return lc->cache_count;

    // 4. Build the gated pool. For CSA: pool_size = 2*ratio = 8, head_dim wide.
    //    For HCA: pool_size = ratio, head_dim wide.
    int pool_size = lc->state_slots;
    float *pool_kv = malloc((size_t)pool_size * (size_t)head_dim * sizeof(float));
    float *pool_score = malloc((size_t)pool_size * (size_t)head_dim * sizeof(float));
    build_pool(lc, lc->kv_state,    pool_kv,    pool_size, 0.0f);
    build_pool(lc, lc->score_state, pool_score, pool_size, -INFINITY);

    // 5. softmax along pool axis (per output dim k), then weighted sum.
    //    Reference: kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1)
    //    softmax happens along the pool axis INDEPENDENTLY for each head_dim k.
    float *out_kv = calloc((size_t)head_dim, sizeof(float));
    float *col = malloc((size_t)pool_size * sizeof(float));
    for (int k = 0; k < head_dim; k++) {
        for (int i = 0; i < pool_size; i++)
            col[i] = pool_score[i * head_dim + k];
        softmax_inplace(col, pool_size);
        float sum = 0.0f;
        for (int i = 0; i < pool_size; i++)
            sum += col[i] * pool_kv[i * head_dim + k];
        out_kv[k] = sum;
    }
    free(col);
    free(pool_kv);
    free(pool_score);

    // 6. RMSNorm (compressor.norm.weight, BF16 [head_dim]).
    char nname[160];
    snprintf(nname, sizeof(nname), "layers.%d.attn.compressor.norm.weight", layer_idx);
    size_t norm_sz;
    const void *norm_raw = model_get_weight(model, nname, &norm_sz);
    if (norm_raw) {
        float *norm_w = malloc((size_t)head_dim * sizeof(float));
        const uint16_t *norm16 = (const uint16_t *)norm_raw;
        for (int i = 0; i < head_dim; i++)
            norm_w[i] = bf16_to_f32_local(norm16[i]);
        float *tmp = malloc((size_t)head_dim * sizeof(float));
        rmsnorm_with_weight(tmp, out_kv, norm_w, head_dim, cfg->rms_norm_eps);
        memcpy(out_kv, tmp, (size_t)head_dim * sizeof(float));
        free(tmp);
        free(norm_w);
    }

    // 7. RoPE the last rope_dim of out_kv at compressed position.
    //    Reference (decode): freqs_cis[start_pos + 1 - ratio]
    //    For prefill: freqs_cis[:cutoff:ratio][block_idx] = freqs_cis[block_idx*ratio]
    //    Both reduce to: position-of-first-token-in-window = position+1-ratio.
    int compressed_position = position + 1 - ratio;
    if (compressed_position < 0) compressed_position = 0;
    apply_rope_inplace(out_kv, head_dim, rope_dim, compressed_position,
                       (float)c->csa_rope_theta);

    // 7b. FP8 simulation on the non-rope dims (QAT precision). Reference
    // applies act_quant(kv[..., :-rd], 64, ...) here. (rotate=True path
    // exists for indexer's own compressor — V4-Flash main attention compressor
    // uses rotate=False which is this branch.)
    fp8_act_quant_inplace(out_kv, head_dim - rope_dim, 64);

    // 8. Append to cache. For CSA also rotate state: state[:ratio] = state[ratio:].
    if (lc->cache_count < lc->cache_capacity) {
        memcpy(lc->cache + (size_t)lc->cache_count * head_dim, out_kv,
               (size_t)head_dim * sizeof(float));
        lc->cache_count++;
    }
    free(out_kv);

    if (lc->overlap) {
        // Copy current window slots (ratio..2*ratio) into prev-window slots
        // (0..ratio) so the NEXT window's pool sees them as "first half"
        // overlap. The current window slots will be overwritten by the next
        // ratio tokens.
        size_t row_bytes = (size_t)D * sizeof(float);
        memcpy(lc->kv_state,
               lc->kv_state + (size_t)ratio * D,
               (size_t)ratio * row_bytes);
        memcpy(lc->score_state,
               lc->score_state + (size_t)ratio * D,
               (size_t)ratio * row_bytes);
    }

    return lc->cache_count;
}

const float *v4_compressor_cache(const V4Compressor *c, int layer_idx,
                                 int *count_out) {
    if (!c) { if (count_out) *count_out = 0; return NULL; }
    const LayerCompressor *lc = &c->layers[layer_idx];
    if (lc->compress_ratio == 0) {
        if (count_out) *count_out = 0;
        return NULL;
    }
    if (count_out) *count_out = lc->cache_count;
    return lc->cache;
}
