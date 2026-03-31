#ifndef INGOT_INFERENCE_H
#define INGOT_INFERENCE_H

#include "model/model.h"

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

// Free inference context.
void inference_free(InferenceContext *ctx);

#endif
