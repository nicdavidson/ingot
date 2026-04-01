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

// Get the byte offset of a weight within model_weights.bin (for Metal buffer offsets).
// Returns -1 if not found.
long model_get_weight_offset(const Model *model, const char *name);

#ifdef PLATFORM_MACOS
#include "compute/metal_context.h"

// Get Metal context for GPU dispatch.
MetalContext *model_get_metal(const Model *model);

// Get Metal buffer wrapping shared weights (zero-copy).
void *model_get_metal_shared_buf(const Model *model);

// Get Metal buffer wrapping an expert layer file (zero-copy).
// Returns NULL if not available.
void *model_get_expert_metal_buf(const Model *model, int layer_idx);
#endif

// Get a pointer to expert weight data for a specific expert in a layer.
// Returns pointer into mmap'd expert file, or NULL if not found.
const void *model_get_expert(const Model *model, int layer_idx, int expert_idx,
                             size_t *out_stride);

// Get the file descriptor for a layer's expert file (for pread-based I/O).
// Returns -1 if not available.
int model_get_expert_fd(const Model *model, int layer_idx);

// Get expert stride (bytes per expert) for a layer.
// Returns 0 if not found.
size_t model_get_expert_stride(const Model *model, int layer_idx);

#endif
