#ifndef INGOT_ATTENTION_H
#define INGOT_ATTENTION_H

#include "config/config.h"
#include "inference/kv_cache.h"
#include "model/model.h"

// Opaque GPU scratch for attention projections (allocated once, reused)
typedef struct AttentionGPU AttentionGPU;

// Allocate/free persistent GPU buffers for attention.
// gpu may be NULL on non-macOS or if Metal is unavailable.
AttentionGPU *attention_gpu_create(const Model *model, const ModelConfig *cfg);
void attention_gpu_free(AttentionGPU *gpu);

// Set the input GPU buffer handle (e.g. gpu_norm_out from InferenceContext)
// so that q4_proj can read input directly from GPU without memcpy.
void attention_gpu_set_input(AttentionGPU *gpu, void *gpu_buf, float *cpu_ptr);

// Full (SWA) attention forward pass for a single layer.
void attention_swa_forward(
    float       *attn_out,     // [num_heads * head_dim] output
    const float *hidden,       // [hidden_size] input
    const Model *model,
    const ModelConfig *cfg,
    InferenceCache *cache,
    AttentionGPU *gpu,         // reusable GPU buffers (may be NULL)
    int          layer_idx,
    int          kv_layer_idx,
    int          position
);

// DeltaNet (linear attention / Mamba-style) forward pass.
void attention_deltanet_forward(
    float       *attn_out,     // [hidden_size] output
    const float *hidden,       // [hidden_size] input
    const Model *model,
    const ModelConfig *cfg,
    InferenceCache *cache,
    AttentionGPU *gpu,         // reusable GPU buffers (may be NULL)
    int          layer_idx,
    int          dn_layer_idx,
    int          position
);

#endif
