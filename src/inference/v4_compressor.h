#ifndef INGOT_V4_COMPRESSOR_H
#define INGOT_V4_COMPRESSOR_H

#include <stdbool.h>
#include "config/config.h"
#include "model/model.h"

// V4 Compressor module — produces compressed KV positions for layers with
// compress_ratio > 0 (CSA: ratio=4 with overlap, HCA: ratio=128 no overlap).
//
// Reference: DeepSeek-V4-Flash inference/model.py::Compressor
//
// One V4Compressor holds per-layer state (rolling window buffers) and the
// compressed KV cache for every layer that has a compressor configured.
typedef struct V4Compressor V4Compressor;

// Create the compressor for a model. Allocates state for every layer with
// compress_ratio > 0; layers with compress_ratio == 0 get a null entry.
// max_seq_len is used to size the per-layer compressed caches
// (cache_capacity = max_seq_len / compress_ratio per layer).
V4Compressor *v4_compressor_create(const ModelConfig *cfg, int max_seq_len);

// Reset all per-layer rolling state and cache counts to zero (start of new
// conversation / cache miss).
void v4_compressor_reset(V4Compressor *c);

void v4_compressor_free(V4Compressor *c);

// Process one token at one layer's compressor. Reads hidden_post_norm
// (output of attn_norm for the current layer / current token) and updates
// the rolling state; emits a new compressed KV entry into the layer's
// cache when (position+1) % compress_ratio == 0.
//
// Returns: total number of valid compressed entries currently in the layer's
// cache (after this step). Returns 0 if the layer has no compressor.
int v4_compressor_step(V4Compressor *c, const Model *model,
                       const ModelConfig *cfg, int layer_idx,
                       const float *hidden_post_norm, int position);

// Returns pointer to the compressed KV cache for this layer.
// Each entry is `head_dim` floats; the last rope_head_dim slice has been
// RoPE-rotated at the layer's compressed RoPE base. Sets *count_out to the
// number of valid entries. Returns NULL with *count_out=0 for layers with
// no compressor.
const float *v4_compressor_cache(const V4Compressor *c, int layer_idx,
                                 int *count_out);

#endif
