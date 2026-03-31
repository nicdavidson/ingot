#ifndef INGOT_KV_CACHE_H
#define INGOT_KV_CACHE_H

#include "config/config.h"

#include <stddef.h>
#include <stdint.h>

// KV Cache for full attention (SWA) layers.
// Stores K and V tensors for each SWA layer.
// Sliding window means we only keep the last W positions.

typedef struct {
    float  *k;            // [max_seq, num_kv_heads, head_dim]
    float  *v;            // [max_seq, num_kv_heads, head_dim]
    int     seq_len;      // current number of cached positions
    int     max_seq;      // maximum positions
    int     num_kv_heads;
    int     head_dim;
} KVLayer;

// DeltaNet recurrent state for linear attention layers.
// One state matrix per head per layer, plus conv1d state.
typedef struct {
    float  *S;           // [num_heads, key_dim, value_dim]
    float  *conv_state;  // [conv_dim, kernel_size-1] — causal conv history
    int     num_heads;
    int     key_dim;
    int     value_dim;
    int     conv_dim;    // key_dim*2 + value_dim (QKV fused)
    int     kernel_size;
} DeltaNetState;

// Full cache for all layers
typedef struct {
    KVLayer       *kv_layers;       // one per full_attention layer
    int            num_kv_layers;

    DeltaNetState *dn_layers;       // one per linear_attention layer
    int            num_dn_layers;
} InferenceCache;

// Create cache based on model config.
// max_seq_len is the maximum context length to support.
InferenceCache *cache_create(const ModelConfig *cfg, int max_seq_len);

// Append a KV entry for a given SWA layer at the current position.
void cache_kv_append(InferenceCache *cache, int layer_idx,
                     const float *k, const float *v);

// Get K and V pointers for a given SWA layer.
// Returns the current sequence length through *out_seq_len.
void cache_kv_get(const InferenceCache *cache, int layer_idx,
                  const float **k, const float **v, int *out_seq_len);

// Get DeltaNet recurrent state for a given linear attention layer.
float *cache_dn_get(InferenceCache *cache, int layer_idx);

// Get DeltaNet conv1d state for a given linear attention layer.
float *cache_dn_conv_get(InferenceCache *cache, int layer_idx);

// Reset cache (new conversation).
void cache_reset(InferenceCache *cache);

// Free cache.
void cache_free(InferenceCache *cache);

#endif
