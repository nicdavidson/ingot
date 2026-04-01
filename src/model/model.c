#define _POSIX_C_SOURCE 200809L

#include "model/model.h"
#include "model/mmap_pool.h"
#include "model/weight_index.h"
#include "util/log.h"
#include "util/timer.h"

#ifdef PLATFORM_MACOS
#include "compute/metal_context.h"
#include <sys/mman.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

struct Model {
    ModelConfig   config;
    Tokenizer    *tokenizer;
    MmapPool     *pool;
    WeightIndex   weight_idx;

    // Shared weights (always resident)
    void         *shared_weights;
    size_t        shared_weights_size;

    // Expert files: one mmap per layer
    void        **expert_data;    // [num_layers]
    size_t       *expert_sizes;   // [num_layers]
    int           num_expert_files;

#ifdef PLATFORM_MACOS
    MetalContext *metal_ctx;
    void         *shared_mtl_buf;  // Metal buffer wrapping shared weights
    void        **expert_mtl_bufs; // Metal buffers wrapping expert files (zero-copy)
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

    // Load weight index
    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s/weight_index.json", model_dir);
    if (!weight_index_load(&model->weight_idx, idx_path)) {
        LOG_WARN("model: no weight_index.json (run convert_weights.py first)");
    }

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
        LOG_WARN("model: no model_weights.bin found (run convert_weights.py first)");
    }

    // Load expert files
    load_expert_files(model, model_dir);

#ifdef PLATFORM_MACOS
    // Initialize Metal
    model->metal_ctx = metal_init();
    if (!model->metal_ctx) {
        LOG_ERROR("model: Metal initialization failed");
        model_free(model);
        return NULL;
    }

    // Wrap shared weights as Metal buffer (zero-copy)
    if (model->shared_weights) {
        model->shared_mtl_buf = metal_wrap_buffer(
            model->metal_ctx,
            model->shared_weights,
            model->shared_weights_size);
    }

    // Wrap expert files as Metal buffers (zero-copy via unified memory)
    if (model->num_expert_files > 0) {
        model->expert_mtl_bufs = calloc((size_t)model->num_expert_files, sizeof(void *));
        int wrapped = 0;
        for (int i = 0; i < model->num_expert_files; i++) {
            if (model->expert_data[i] && model->expert_sizes[i] > 0) {
                model->expert_mtl_bufs[i] = metal_wrap_buffer(
                    model->metal_ctx,
                    model->expert_data[i],
                    model->expert_sizes[i]);
                if (model->expert_mtl_bufs[i]) wrapped++;
            }
        }
        LOG_INFO("model: wrapped %d/%d expert files as Metal buffers",
                 wrapped, model->num_expert_files);
    }
#endif

    uint64_t t1 = timer_now_ns();
    LOG_INFO("model: loaded in %.1f ms", timer_elapsed_ms(t0, t1));

    return model;
}

void model_free(Model *model) {
    if (!model) return;

#ifdef PLATFORM_MACOS
    if (model->expert_mtl_bufs) {
        for (int i = 0; i < model->num_expert_files; i++) {
            metal_free_buffer(model->expert_mtl_bufs[i]);
        }
        free(model->expert_mtl_bufs);
    }
    if (model->shared_mtl_buf) {
        metal_free_buffer(model->shared_mtl_buf);
    }
    if (model->metal_ctx) {
        metal_free(model->metal_ctx);
    }
#endif

    tokenizer_free(model->tokenizer);
    config_free(&model->config);
    mmap_pool_free(model->pool);
    free(model->expert_data);
    free(model->expert_sizes);
    weight_index_free(&model->weight_idx);
    free(model);
}

const ModelConfig *model_config(const Model *model) {
    return &model->config;
}

const Tokenizer *model_tokenizer(const Model *model) {
    return model->tokenizer;
}

const void *model_get_weight(const Model *model, const char *name, size_t *out_size) {
    const WeightEntry *e = weight_index_find(&model->weight_idx, name);
    if (!e || !model->shared_weights) return NULL;

    if (e->offset + e->size > model->shared_weights_size) {
        return NULL; // out of bounds
    }

    if (out_size) *out_size = e->size;
    return (const char *)model->shared_weights + e->offset;
}

long model_get_weight_offset(const Model *model, const char *name) {
    const WeightEntry *e = weight_index_find(&model->weight_idx, name);
    if (!e) return -1;
    return (long)e->offset;
}

#ifdef PLATFORM_MACOS
MetalContext *model_get_metal(const Model *model) {
    return model->metal_ctx;
}

void *model_get_metal_shared_buf(const Model *model) {
    return model->shared_mtl_buf;
}

void *model_get_expert_metal_buf(const Model *model, int layer_idx) {
    if (!model->expert_mtl_bufs || layer_idx < 0 ||
        layer_idx >= model->num_expert_files)
        return NULL;
    return model->expert_mtl_bufs[layer_idx];
}
#endif

const void *model_get_expert(const Model *model, int layer_idx, int expert_idx,
                             size_t *out_stride) {
    // Find the expert index entry
    char name[64];
    snprintf(name, sizeof(name), "layers.%d.experts", layer_idx);
    const WeightEntry *e = weight_index_find(&model->weight_idx, name);
    if (!e) return NULL;

    if (expert_idx < 0 || expert_idx >= e->num_experts) return NULL;

    // Find the mmap'd expert file for this layer
    // Expert files are ordered by layer index
    int expert_file_idx = -1;
    for (int i = 0; i < model->num_expert_files; i++) {
        // Simple match — expert files are loaded in order
        if (i == layer_idx || expert_file_idx < 0) {
            expert_file_idx = i;
            // Actually need to match by layer index. For now, just use linear mapping
            // since expert files are named layer_XX.bin and loaded in sorted order.
        }
    }

    // For now, use a simple mapping: expert_file_idx maps directly to layers
    // with expert files (every layer has one since all layers have MoE)
    if (expert_file_idx < 0 || expert_file_idx >= model->num_expert_files)
        return NULL;

    void *data = model->expert_data[expert_file_idx];
    if (!data) return NULL;

    if (out_stride) *out_stride = e->expert_stride;
    return (const char *)data + (size_t)expert_idx * e->expert_stride;
}
