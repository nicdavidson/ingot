#ifndef INGOT_ATTENTION_H
#define INGOT_ATTENTION_H

#include "config/config.h"
#include "inference/kv_cache.h"
#include "model/model.h"

// Full (SWA) attention forward pass for a single layer.
// Reads Q/K/V projection weights, applies RoPE, caches KV,
// computes attention, and writes attn_out.
void attention_swa_forward(
    float       *attn_out,     // [num_heads * head_dim] output
    const float *hidden,       // [hidden_size] input
    const Model *model,
    const ModelConfig *cfg,
    InferenceCache *cache,
    int          layer_idx,    // global layer index
    int          kv_layer_idx, // index into KV cache (SWA layers only)
    int          position      // token position
);

// DeltaNet (linear attention / Mamba-style) forward pass.
// Uses recurrent state instead of KV cache.
void attention_deltanet_forward(
    float       *attn_out,     // [hidden_size] output
    const float *hidden,       // [hidden_size] input
    const Model *model,
    const ModelConfig *cfg,
    InferenceCache *cache,
    int          layer_idx,    // global layer index
    int          dn_layer_idx, // index into DeltaNet state
    int          position      // token position
);

#endif
