#ifndef INGOT_INFERENCE_H
#define INGOT_INFERENCE_H

#include "model/model.h"
#include "inference/kv_cache.h"

#include <stdint.h>

typedef struct InferenceContext InferenceContext;

// Create inference context for a loaded model.
InferenceContext *inference_create(Model *model);

// Generate tokens. Calls callback for each generated token.
// Returns total tokens generated.
typedef void (*TokenCallback)(int32_t token_id, const char *text, void *userdata);

int inference_generate(InferenceContext *ctx,
                       const int32_t *prompt_tokens, int num_prompt_tokens,
                       int max_tokens,
                       float temperature, float top_p, int top_k,
                       TokenCallback callback, void *userdata);

// --- Split prefill/generate API (for system prompt caching) ---

// Prefill tokens into the context without generating.
// start_position = position index of the first token (0 for fresh context,
//                  or snapshot->position for continuation after cache restore).
void inference_prefill(InferenceContext *ctx,
                       const int32_t *tokens, int num_tokens,
                       int start_position);

// Generate from the current state (prefill must have been done already).
// total_prompt_tokens = total number of tokens prefilled (for position tracking).
int inference_generate_tokens(InferenceContext *ctx,
                              int total_prompt_tokens,
                              int max_tokens,
                              float temperature, float top_p, int top_k,
                              TokenCallback callback, void *userdata);

// Access the underlying cache (for snapshot save/restore).
InferenceCache *inference_get_cache(InferenceContext *ctx);

// Reset V4-specific per-conversation state (compressor rolling buffers).
// No-op for non-V4 models. Call after cache_reset() on a cache miss.
void inference_reset_v4_state(InferenceContext *ctx);

// Free inference context.
void inference_free(InferenceContext *ctx);

#endif
