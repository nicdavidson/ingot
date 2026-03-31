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

#endif
