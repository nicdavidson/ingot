#include "inference/kv_cache.h"
#include "util/log.h"

#include <stdlib.h>
#include <string.h>

InferenceCache *cache_create(const ModelConfig *cfg, int max_seq_len) {
    InferenceCache *cache = calloc(1, sizeof(InferenceCache));
    if (!cache) return NULL;

    // Count SWA and DeltaNet layers
    int num_kv = 0, num_dn = 0;
    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        if (cfg->layer_types[i] == LAYER_FULL_ATTENTION) num_kv++;
        else num_dn++;
    }

    // Allocate KV layers (full attention / SWA)
    cache->num_kv_layers = num_kv;
    cache->kv_layers = calloc((size_t)num_kv, sizeof(KVLayer));
    for (int i = 0; i < num_kv; i++) {
        KVLayer *kv = &cache->kv_layers[i];
        kv->max_seq = max_seq_len;
        kv->num_kv_heads = cfg->num_key_value_heads;
        kv->head_dim = cfg->head_dim;
        size_t entry_size = (size_t)max_seq_len * (size_t)cfg->num_key_value_heads *
                            (size_t)cfg->head_dim * sizeof(float);
        kv->k = calloc(1, entry_size);
        kv->v = calloc(1, entry_size);
    }

    // Allocate DeltaNet states (linear attention)
    // State is [num_v_heads, head_k_dim, head_v_dim] per layer
    // (one recurrent state matrix per value head)
    cache->num_dn_layers = num_dn;
    cache->dn_layers = calloc((size_t)num_dn, sizeof(DeltaNetState));
    for (int i = 0; i < num_dn; i++) {
        DeltaNetState *dn = &cache->dn_layers[i];
        dn->num_heads = cfg->linear_attn.linear_num_value_heads;
        dn->key_dim = cfg->linear_attn.linear_key_head_dim;
        dn->value_dim = cfg->linear_attn.linear_value_head_dim;
        size_t state_size = (size_t)dn->num_heads *
                            (size_t)dn->key_dim *
                            (size_t)dn->value_dim * sizeof(float);
        dn->S = calloc(1, state_size);
    }

    size_t kv_mem = (size_t)num_kv * 2 * (size_t)max_seq_len *
                    (size_t)cfg->num_key_value_heads *
                    (size_t)cfg->head_dim * sizeof(float);
    size_t dn_mem = (size_t)num_dn *
                    (size_t)cfg->linear_attn.linear_num_value_heads *
                    (size_t)cfg->linear_attn.linear_key_head_dim *
                    (size_t)cfg->linear_attn.linear_value_head_dim * sizeof(float);

    LOG_INFO("cache: %d KV layers (%zu MB) + %d DeltaNet layers (%zu MB)",
             num_kv, kv_mem / (1024 * 1024), num_dn, dn_mem / (1024 * 1024));

    return cache;
}

void cache_kv_append(InferenceCache *cache, int layer_idx,
                     const float *k, const float *v) {
    KVLayer *kv = &cache->kv_layers[layer_idx];
    if (kv->seq_len >= kv->max_seq) {
        // Sliding window: shift everything left by 1
        size_t entry_bytes = (size_t)kv->num_kv_heads * (size_t)kv->head_dim * sizeof(float);
        memmove(kv->k, kv->k + kv->num_kv_heads * kv->head_dim,
                entry_bytes * (size_t)(kv->max_seq - 1));
        memmove(kv->v, kv->v + kv->num_kv_heads * kv->head_dim,
                entry_bytes * (size_t)(kv->max_seq - 1));
        kv->seq_len = kv->max_seq - 1;
    }

    size_t offset = (size_t)kv->seq_len * (size_t)kv->num_kv_heads * (size_t)kv->head_dim;
    size_t entry_floats = (size_t)kv->num_kv_heads * (size_t)kv->head_dim;
    memcpy(kv->k + offset, k, entry_floats * sizeof(float));
    memcpy(kv->v + offset, v, entry_floats * sizeof(float));
    kv->seq_len++;
}

void cache_kv_get(const InferenceCache *cache, int layer_idx,
                  const float **k, const float **v, int *out_seq_len) {
    const KVLayer *kv = &cache->kv_layers[layer_idx];
    *k = kv->k;
    *v = kv->v;
    *out_seq_len = kv->seq_len;
}

float *cache_dn_get(InferenceCache *cache, int layer_idx) {
    return cache->dn_layers[layer_idx].S;
}

void cache_reset(InferenceCache *cache) {
    for (int i = 0; i < cache->num_kv_layers; i++) {
        KVLayer *kv = &cache->kv_layers[i];
        size_t sz = (size_t)kv->max_seq * (size_t)kv->num_kv_heads *
                    (size_t)kv->head_dim * sizeof(float);
        memset(kv->k, 0, sz);
        memset(kv->v, 0, sz);
        kv->seq_len = 0;
    }
    for (int i = 0; i < cache->num_dn_layers; i++) {
        DeltaNetState *dn = &cache->dn_layers[i];
        size_t sz = (size_t)dn->num_heads * (size_t)dn->key_dim *
                    (size_t)dn->value_dim * sizeof(float);
        memset(dn->S, 0, sz);
    }
}

void cache_free(InferenceCache *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->num_kv_layers; i++) {
        free(cache->kv_layers[i].k);
        free(cache->kv_layers[i].v);
    }
    free(cache->kv_layers);
    for (int i = 0; i < cache->num_dn_layers; i++) {
        free(cache->dn_layers[i].S);
    }
    free(cache->dn_layers);
    free(cache);
}
