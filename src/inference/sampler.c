#include "inference/sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

Sampler *sampler_create(float temperature, float top_p, int top_k,
                        float repetition_penalty, int rep_window) {
    Sampler *s = calloc(1, sizeof(Sampler));
    s->temperature = temperature;
    s->top_p = top_p;
    s->top_k = top_k;
    s->repetition_penalty = repetition_penalty;
    s->last_tokens_size = rep_window;
    s->last_tokens = calloc((size_t)rep_window, sizeof(int));
    return s;
}

// Comparison function for sorting logits descending
typedef struct { float val; int idx; } LogitPair;

static int logit_cmp(const void *a, const void *b) {
    float diff = ((const LogitPair *)b)->val - ((const LogitPair *)a)->val;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

int32_t sampler_sample(Sampler *s, float *logits, int vocab_size) {
    // Apply repetition penalty
    if (s->repetition_penalty != 1.0f) {
        for (int i = 0; i < s->last_tokens_size; i++) {
            int tok = s->last_tokens[i];
            if (tok <= 0) continue;
            if (logits[tok] > 0) {
                logits[tok] /= s->repetition_penalty;
            } else {
                logits[tok] *= s->repetition_penalty;
            }
        }
    }

    // Temperature = 0: greedy
    if (s->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return (int32_t)best;
    }

    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= s->temperature;
    }

    // Top-K: keep only top_k logits
    int n = vocab_size;
    LogitPair *pairs = malloc((size_t)n * sizeof(LogitPair));
    for (int i = 0; i < n; i++) {
        pairs[i] = (LogitPair){ .val = logits[i], .idx = i };
    }
    qsort(pairs, (size_t)n, sizeof(LogitPair), logit_cmp);

    if (s->top_k > 0 && s->top_k < n) {
        n = s->top_k;
    }

    // Softmax over top-K
    float max_val = pairs[0].val;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        pairs[i].val = expf(pairs[i].val - max_val);
        sum += pairs[i].val;
    }
    for (int i = 0; i < n; i++) {
        pairs[i].val /= sum;
    }

    // Top-P (nucleus sampling)
    if (s->top_p < 1.0f) {
        float cumsum = 0.0f;
        for (int i = 0; i < n; i++) {
            cumsum += pairs[i].val;
            if (cumsum >= s->top_p) {
                n = i + 1;
                // Renormalize
                sum = cumsum;
                for (int j = 0; j < n; j++) pairs[j].val /= sum;
                break;
            }
        }
    }

    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int selected = pairs[0].idx;
    for (int i = 0; i < n; i++) {
        cumsum += pairs[i].val;
        if (r <= cumsum) {
            selected = pairs[i].idx;
            break;
        }
    }

    free(pairs);
    return (int32_t)selected;
}

void sampler_accept(Sampler *s, int32_t token) {
    s->last_tokens[s->last_tokens_pos] = (int)token;
    s->last_tokens_pos = (s->last_tokens_pos + 1) % s->last_tokens_size;
}

void sampler_reset(Sampler *s) {
    memset(s->last_tokens, 0, (size_t)s->last_tokens_size * sizeof(int));
    s->last_tokens_pos = 0;
}

void sampler_free(Sampler *s) {
    if (!s) return;
    free(s->last_tokens);
    free(s);
}
