#ifndef INGOT_SAMPLER_H
#define INGOT_SAMPLER_H

#include <stdint.h>

typedef struct {
    float  temperature;
    float  top_p;
    int    top_k;
    float  repetition_penalty;
    int   *last_tokens;     // circular buffer for rep penalty
    int    last_tokens_size;
    int    last_tokens_pos;
} Sampler;

// Create sampler with given parameters.
Sampler *sampler_create(float temperature, float top_p, int top_k,
                        float repetition_penalty, int rep_window);

// Sample a token from logits[vocab_size].
// Applies temperature, top-k, top-p, and repetition penalty.
int32_t sampler_sample(Sampler *s, float *logits, int vocab_size);

// Record a generated token (for repetition penalty).
void sampler_accept(Sampler *s, int32_t token);

// Reset sampler state (new generation).
void sampler_reset(Sampler *s);

// Free sampler.
void sampler_free(Sampler *s);

#endif
