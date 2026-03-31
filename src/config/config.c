#define _POSIX_C_SOURCE 200809L

#include "config/config.h"
#include "util/json_parse.h"
#include "util/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read entire file into malloc'd buffer. Caller frees.
static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (len <= 0) { fclose(f); return NULL; }

    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t read = fread(buf, 1, (size_t)len, f);
    fclose(f);

    buf[read] = '\0';
    *out_len = read;
    return buf;
}

static void parse_rope(const JsonDoc *doc, int rope_idx, RopeConfig *rope) {
    *rope = (RopeConfig){
        .rope_theta          = 10000000.0,
        .partial_rotary_factor = 0.25f,
    };

    if (rope_idx < 0) return;

    int idx;
    if ((idx = json_get(doc, rope_idx, "rope_theta")) >= 0)
        rope->rope_theta = json_number(doc, idx);
    if ((idx = json_get(doc, rope_idx, "partial_rotary_factor")) >= 0)
        rope->partial_rotary_factor = (float)json_number(doc, idx);
    if ((idx = json_get(doc, rope_idx, "mrope_interleaved")) >= 0)
        rope->mrope_interleaved = json_bool(doc, idx);

    int sec_idx = json_get(doc, rope_idx, "mrope_section");
    if (sec_idx >= 0) {
        for (int i = 0; i < 3; i++) {
            int el = json_array_get(doc, sec_idx, i);
            if (el >= 0) rope->mrope_section[i] = json_int(doc, el);
        }
    }
}

static LayerType *parse_layer_types(const JsonDoc *doc, int arr_idx, int num_layers) {
    LayerType *types = calloc((size_t)num_layers, sizeof(LayerType));
    if (!types) return NULL;

    int arr_len = json_array_len(doc, arr_idx);

    for (int i = 0; i < num_layers; i++) {
        if (i < arr_len) {
            int el = json_array_get(doc, arr_idx, i);
            if (el >= 0) {
                char buf[64];
                if (json_string(doc, el, buf, sizeof(buf))) {
                    if (strcmp(buf, "full_attention") == 0)
                        types[i] = LAYER_FULL_ATTENTION;
                    else
                        types[i] = LAYER_LINEAR_ATTENTION;
                }
            }
        }
    }
    return types;
}

bool config_load(ModelConfig *cfg, const char *path) {
    memset(cfg, 0, sizeof(*cfg));

    size_t len;
    char *json = read_file(path, &len);
    if (!json) {
        LOG_ERROR("config: failed to read %s", path);
        return false;
    }

    // 512 tokens is plenty for these configs
    JsonToken tokens[512];
    JsonDoc doc;
    if (!json_parse(&doc, json, len, tokens, 512)) {
        LOG_ERROR("config: failed to parse JSON in %s", path);
        free(json);
        return false;
    }

    // Root must be an object
    if (doc.tokens[0].type != JSON_OBJECT) {
        LOG_ERROR("config: root is not an object");
        free(json);
        return false;
    }

    // All the fields we need live under "text_config"
    int text_idx = json_get(&doc, 0, "text_config");
    if (text_idx < 0) {
        LOG_ERROR("config: missing text_config");
        free(json);
        return false;
    }

    int idx;

    // Core dimensions
    if ((idx = json_get(&doc, text_idx, "hidden_size")) >= 0)
        cfg->hidden_size = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "num_hidden_layers")) >= 0)
        cfg->num_hidden_layers = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "num_attention_heads")) >= 0)
        cfg->num_attention_heads = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "num_key_value_heads")) >= 0)
        cfg->num_key_value_heads = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "head_dim")) >= 0)
        cfg->head_dim = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "vocab_size")) >= 0)
        cfg->vocab_size = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "max_position_embeddings")) >= 0)
        cfg->max_position_embeddings = json_int(&doc, idx);

    // MoE
    if ((idx = json_get(&doc, text_idx, "num_experts")) >= 0)
        cfg->num_experts = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "num_experts_per_tok")) >= 0)
        cfg->num_experts_per_tok = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "moe_intermediate_size")) >= 0)
        cfg->moe_intermediate_size = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "shared_expert_intermediate_size")) >= 0)
        cfg->shared_expert_intermediate_size = json_int(&doc, idx);

    // Normalization
    cfg->rms_norm_eps = 1e-6f;
    if ((idx = json_get(&doc, text_idx, "rms_norm_eps")) >= 0)
        cfg->rms_norm_eps = (float)json_number(&doc, idx);

    // Special tokens
    if ((idx = json_get(&doc, text_idx, "eos_token_id")) >= 0)
        cfg->eos_token_id = json_int(&doc, idx);

    // Layer pattern
    cfg->full_attention_interval = 4;
    if ((idx = json_get(&doc, text_idx, "full_attention_interval")) >= 0)
        cfg->full_attention_interval = json_int(&doc, idx);

    int lt_idx = json_get(&doc, text_idx, "layer_types");
    if (lt_idx >= 0 && cfg->num_hidden_layers > 0)
        cfg->layer_types = parse_layer_types(&doc, lt_idx, cfg->num_hidden_layers);

    // Attention config
    if ((idx = json_get(&doc, text_idx, "attn_output_gate")) >= 0)
        cfg->attn_output_gate = json_bool(&doc, idx);

    // Linear attention (DeltaNet) config
    if ((idx = json_get(&doc, text_idx, "linear_conv_kernel_dim")) >= 0)
        cfg->linear_attn.linear_conv_kernel_dim = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "linear_key_head_dim")) >= 0)
        cfg->linear_attn.linear_key_head_dim = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "linear_num_key_heads")) >= 0)
        cfg->linear_attn.linear_num_key_heads = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "linear_num_value_heads")) >= 0)
        cfg->linear_attn.linear_num_value_heads = json_int(&doc, idx);
    if ((idx = json_get(&doc, text_idx, "linear_value_head_dim")) >= 0)
        cfg->linear_attn.linear_value_head_dim = json_int(&doc, idx);

    // RoPE
    int rope_idx = json_get(&doc, text_idx, "rope_parameters");
    parse_rope(&doc, rope_idx, &cfg->rope);

    // MTP
    if ((idx = json_get(&doc, text_idx, "mtp_num_hidden_layers")) >= 0)
        cfg->mtp_num_hidden_layers = json_int(&doc, idx);

    // Derive model name from dimensions
    int total_params_approx = cfg->hidden_size * cfg->num_hidden_layers * cfg->num_experts / 10;
    (void)total_params_approx;
    snprintf(cfg->model_name, sizeof(cfg->model_name),
             "qwen3.5-%dL-%dE-%dH",
             cfg->num_hidden_layers, cfg->num_experts, cfg->hidden_size);

    free(json);

    LOG_INFO("config: loaded %s", cfg->model_name);
    return true;
}

void config_free(ModelConfig *cfg) {
    free(cfg->layer_types);
    cfg->layer_types = NULL;
}

void config_print(const ModelConfig *cfg) {
    LOG_INFO("config: model=%s", cfg->model_name);
    LOG_INFO("config: hidden_size=%d, layers=%d, heads=%d, kv_heads=%d",
             cfg->hidden_size, cfg->num_hidden_layers,
             cfg->num_attention_heads, cfg->num_key_value_heads);
    LOG_INFO("config: head_dim=%d, vocab=%d, max_pos=%d",
             cfg->head_dim, cfg->vocab_size, cfg->max_position_embeddings);
    LOG_INFO("config: experts=%d, active=%d, moe_dim=%d, shared_dim=%d",
             cfg->num_experts, cfg->num_experts_per_tok,
             cfg->moe_intermediate_size, cfg->shared_expert_intermediate_size);
    LOG_INFO("config: rope_theta=%.0f, partial_rotary=%.2f",
             cfg->rope.rope_theta, cfg->rope.partial_rotary_factor);

    int linear_count = 0, full_count = 0;
    if (cfg->layer_types) {
        for (int i = 0; i < cfg->num_hidden_layers; i++) {
            if (cfg->layer_types[i] == LAYER_LINEAR_ATTENTION) linear_count++;
            else full_count++;
        }
    }
    LOG_INFO("config: layer pattern: %d linear (DeltaNet) + %d full (SWA)",
             linear_count, full_count);
}
