#define _POSIX_C_SOURCE 200809L

#include "model/model.h"
#include "model/mmap_pool.h"
#include "util/log.h"
#include "util/timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

// Weight index: maps weight names to their mmap'd locations
typedef struct {
    char  name[128];
    void *data;
    size_t size;
} WeightEntry;

struct Model {
    ModelConfig  config;
    Tokenizer   *tokenizer;
    MmapPool    *pool;

    // Shared weights (always resident)
    void        *shared_weights;
    size_t       shared_weights_size;

    // Expert files: one mmap per layer
    void       **expert_data;    // [num_layers]
    size_t      *expert_sizes;   // [num_layers]
    int          num_expert_files;

    // Weight index
    WeightEntry *weights;
    int          num_weights;
    int          max_weights;

#ifdef PLATFORM_MACOS
    void        *metal_ctx;        // MetalContext*
    void        *shared_mtl_buf;   // Metal buffer wrapping shared weights
#endif
};

// Scan for expert layer files (layer_00.bin, layer_01.bin, etc.)
static int load_expert_files(Model *model, const char *model_dir) {
    char expert_dir[1024];
    snprintf(expert_dir, sizeof(expert_dir), "%s/packed_experts", model_dir);

    DIR *dir = opendir(expert_dir);
    if (!dir) {
        LOG_WARN("model: no packed_experts/ directory found");
        return 0;
    }

    int count = 0;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strstr(ent->d_name, "layer_") && strstr(ent->d_name, ".bin")) {
            count++;
        }
    }
    closedir(dir);

    if (count == 0) return 0;

    model->expert_data = calloc((size_t)count, sizeof(void *));
    model->expert_sizes = calloc((size_t)count, sizeof(size_t));
    model->num_expert_files = count;

    // Reopen and map each file
    dir = opendir(expert_dir);
    int idx = 0;
    while ((ent = readdir(dir)) != NULL && idx < count) {
        if (strstr(ent->d_name, "layer_") && strstr(ent->d_name, ".bin")) {
            char path[2048];
            snprintf(path, sizeof(path), "%s/%s", expert_dir, ent->d_name);

            size_t size;
            void *data = mmap_pool_add(model->pool, path, &size);
            if (data) {
                model->expert_data[idx] = data;
                model->expert_sizes[idx] = size;
                idx++;
            }
        }
    }
    closedir(dir);

    LOG_INFO("model: mapped %d expert layer files", idx);
    return idx;
}

Model *model_load(const char *model_dir) {
    uint64_t t0 = timer_now_ns();

    Model *model = calloc(1, sizeof(Model));
    if (!model) return NULL;

    // Load config
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    if (!config_load(&model->config, config_path)) {
        LOG_ERROR("model: failed to load config from %s", config_path);
        free(model);
        return NULL;
    }
    config_print(&model->config);

    // Load tokenizer
    model->tokenizer = tokenizer_load(model_dir);
    if (!model->tokenizer) {
        LOG_ERROR("model: failed to load tokenizer from %s", model_dir);
        config_free(&model->config);
        free(model);
        return NULL;
    }

    // Create mmap pool (shared weights + up to 100 expert files)
    model->pool = mmap_pool_create(128);

    // Load shared weights
    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/model_weights.bin", model_dir);
    model->shared_weights = mmap_pool_add(model->pool, weights_path,
                                           &model->shared_weights_size);
    if (model->shared_weights) {
        LOG_INFO("model: shared weights = %zu MB",
                 model->shared_weights_size / (1024 * 1024));

        // Lock shared weights in RAM — these must never be paged out
#ifdef PLATFORM_MACOS
        mlock(model->shared_weights, model->shared_weights_size);
#endif
    } else {
        LOG_WARN("model: no model_weights.bin found (will need weights in another format)");
    }

    // Load expert files
    load_expert_files(model, model_dir);

#ifdef PLATFORM_MACOS
    // Initialize Metal
    extern MetalContext *metal_init(void);
    model->metal_ctx = metal_init();
    if (!model->metal_ctx) {
        LOG_ERROR("model: Metal initialization failed");
        model_free(model);
        return NULL;
    }

    // Wrap shared weights as Metal buffer (zero-copy)
    if (model->shared_weights) {
        extern void *metal_wrap_buffer(MetalContext *, void *, size_t);
        model->shared_mtl_buf = metal_wrap_buffer(
            model->metal_ctx,
            model->shared_weights,
            model->shared_weights_size);
    }
#endif

    uint64_t t1 = timer_now_ns();
    LOG_INFO("model: loaded in %.1f ms", timer_elapsed_ms(t0, t1));

    return model;
}

void model_free(Model *model) {
    if (!model) return;

#ifdef PLATFORM_MACOS
    if (model->shared_mtl_buf) {
        extern void metal_free_buffer(void *);
        metal_free_buffer(model->shared_mtl_buf);
    }
    if (model->metal_ctx) {
        extern void metal_free(MetalContext *);
        metal_free(model->metal_ctx);
    }
#endif

    tokenizer_free(model->tokenizer);
    config_free(&model->config);
    mmap_pool_free(model->pool);
    free(model->expert_data);
    free(model->expert_sizes);
    free(model->weights);
    free(model);
}

const ModelConfig *model_config(const Model *model) {
    return &model->config;
}

const Tokenizer *model_tokenizer(const Model *model) {
    return model->tokenizer;
}
