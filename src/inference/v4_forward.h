#ifndef INGOT_V4_FORWARD_H
#define INGOT_V4_FORWARD_H

#include "config/config.h"
#include "model/model.h"
#include "inference/kv_cache.h"
#include "inference/attention.h"

#ifdef PLATFORM_MACOS
#include "compute/metal_context.h"
#endif

// V4 inference context — holds hyper-connection state and scratch buffers
typedef struct V4InferenceContext V4InferenceContext;

// Create V4 context. Allocates HC state, scratch buffers, and GPU resources.
V4InferenceContext *v4_inference_create(Model *model, const ModelConfig *cfg,
                                        InferenceCache *cache, AttentionGPU *attn_gpu);

// Initialize HC state from a single embedding vector.
// Expands [hidden_size] → [hc_mult * hidden_size] (4 identical copies).
void v4_init_hc_state(V4InferenceContext *v4, const float *embedding, int hidden_size);

// Process one V4 layer (HC-wrapped attention + HC-wrapped MoE).
// token_id is needed for hash routing (first num_hash_layers layers).
void v4_forward_layer(V4InferenceContext *v4, int layer_idx, int position, int token_id);

// Final HC reduction + RMSNorm → logits.
// Writes logits to out_logits[vocab_size].
void v4_compute_logits(V4InferenceContext *v4, float *out_logits);

// Free V4 context.
void v4_inference_free(V4InferenceContext *v4);

// Reset per-conversation state (currently: clear the compressor's rolling
// window buffers and compressed-KV caches). Call when the KV cache is reset.
void v4_reset_state(V4InferenceContext *v4);

// V4 timing instrumentation
void v4_timing_reset(void);
void v4_timing_report(int token_num);

#endif
