#ifndef INGOT_MODEL_H
#define INGOT_MODEL_H

#include "config/config.h"
#include "tokenizer/tokenizer.h"

#include <stdbool.h>

typedef struct Model Model;

// Load model from directory. Parses config, loads tokenizer,
// mmaps shared weights and expert files.
// Returns NULL on failure.
Model *model_load(const char *model_dir);

// Free model and all associated resources.
void model_free(Model *model);

// Get model config.
const ModelConfig *model_config(const Model *model);

// Get model tokenizer.
const Tokenizer *model_tokenizer(const Model *model);

// Get a pointer to a shared weight tensor by name.
// Returns pointer into mmap'd region, or NULL if not found.
// The name should match the weight_index.json key (e.g., "layers.0.input_layernorm.weight").
const void *model_get_weight(const Model *model, const char *name, size_t *out_size);

// Get a pointer to expert weight data for a specific expert in a layer.
// Returns pointer into mmap'd expert file, or NULL if not found.
const void *model_get_expert(const Model *model, int layer_idx, int expert_idx,
                             size_t *out_stride);

#endif
