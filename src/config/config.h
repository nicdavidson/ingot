#ifndef INGOT_CONFIG_H
#define INGOT_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

// Layer attention type — 3:1 ratio of linear (DeltaNet) to full (SWA)
typedef enum {
    LAYER_LINEAR_ATTENTION,
    LAYER_FULL_ATTENTION,
} LayerType;

// RoPE configuration
typedef struct {
    double rope_theta;
    float  partial_rotary_factor;
    bool   mrope_interleaved;
    int    mrope_section[3];
} RopeConfig;

// DeltaNet (linear attention) configuration
typedef struct {
    int linear_conv_kernel_dim;
    int linear_key_head_dim;
    int linear_num_key_heads;
    int linear_num_value_heads;
    int linear_value_head_dim;
} LinearAttnConfig;

// Model configuration — parsed from HuggingFace config.json
typedef struct {
    // Identity
    char model_name[256];

    // Core dimensions
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int vocab_size;
    int max_position_embeddings;

    // MoE
    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate_size;
    int shared_expert_intermediate_size;

    // Normalization
    float rms_norm_eps;

    // Special tokens
    int eos_token_id;

    // Layer pattern
    int              full_attention_interval;
    LayerType       *layer_types;  // array of num_hidden_layers

    // Attention sub-configs
    RopeConfig       rope;
    LinearAttnConfig linear_attn;
    bool             attn_output_gate;

    // MTP (multi-token prediction)
    int  mtp_num_hidden_layers;
} ModelConfig;

// Parse a HuggingFace config.json file into ModelConfig.
// Returns true on success. Caller must free with config_free().
bool config_load(ModelConfig *cfg, const char *path);

// Free dynamically allocated fields in config.
void config_free(ModelConfig *cfg);

// Print config summary to log.
void config_print(const ModelConfig *cfg);

#endif
