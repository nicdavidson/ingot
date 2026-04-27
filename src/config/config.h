#ifndef INGOT_CONFIG_H
#define INGOT_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

// Model architecture family
typedef enum {
    ARCH_QWEN35,
    ARCH_DEEPSEEK_V4,
} ModelArch;

// Layer attention type
typedef enum {
    LAYER_LINEAR_ATTENTION,   // Qwen 3.5: DeltaNet
    LAYER_FULL_ATTENTION,     // Qwen 3.5: SWA (sliding window with GQA)
    LAYER_V4_SLIDING_WINDOW,  // V4: sliding window only (compress_ratio=0)
    LAYER_V4_CSA,             // V4: compressed sparse attention (ratio=4)
    LAYER_V4_HCA,             // V4: heavily compressed attention (ratio=128)
} LayerType;

// RoPE configuration
typedef struct {
    double rope_theta;
    float  partial_rotary_factor;
    bool   mrope_interleaved;
    int    mrope_section[3];
} RopeConfig;

// DeltaNet (linear attention) configuration — Qwen 3.5 only
typedef struct {
    int linear_conv_kernel_dim;
    int linear_key_head_dim;
    int linear_num_key_heads;
    int linear_num_value_heads;
    int linear_value_head_dim;
} LinearAttnConfig;

// DeepSeek V4 attention configuration
typedef struct {
    int  q_lora_rank;
    int  o_lora_rank;
    int  o_groups;
    int  qk_rope_head_dim;

    int *compress_ratios;    // per-layer array
    double compress_rope_theta;

    int  index_head_dim;
    int  index_n_heads;
    int  index_topk;

    int  hc_mult;
    float hc_eps;
    int  hc_sinkhorn_iters;

    int  num_hash_layers;
    int  window_size;
    double route_scale;        // routed_scaling_factor (V4: 1.5)
} V4AttnConfig;

// Model configuration — parsed from HuggingFace config.json
typedef struct {
    // Identity
    char     model_name[256];
    ModelArch arch;

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
    LayerType       *layer_types;

    // Attention sub-configs
    RopeConfig       rope;
    LinearAttnConfig linear_attn;
    V4AttnConfig     v4;
    bool             attn_output_gate;

    // MTP (multi-token prediction)
    int  mtp_num_hidden_layers;
} ModelConfig;

bool config_load(ModelConfig *cfg, const char *path);
void config_free(ModelConfig *cfg);
void config_print(const ModelConfig *cfg);

#endif
