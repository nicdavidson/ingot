#define _POSIX_C_SOURCE 200809L

#include "config/config.h"
#include "util/json_parse.h"
#include "util/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static LayerType *parse_qwen_layer_types(const JsonDoc *doc, int arr_idx, int num_layers) {
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

static LayerType *parse_v4_layer_types(const JsonDoc *doc, int arr_idx,
                                       int num_layers, int **compress_ratios_out) {
    LayerType *types = calloc((size_t)num_layers, sizeof(LayerType));
    int *ratios = calloc((size_t)num_layers, sizeof(int));
    if (!types || !ratios) { free(types); free(ratios); return NULL; }

    int arr_len = json_array_len(doc, arr_idx);

    for (int i = 0; i < num_layers && i < arr_len; i++) {
        int el = json_array_get(doc, arr_idx, i);
        if (el >= 0) {
            int ratio = json_int(doc, el);
            ratios[i] = ratio;
            if (ratio == 0)
                types[i] = LAYER_V4_SLIDING_WINDOW;
            else if (ratio <= 4)
                types[i] = LAYER_V4_CSA;
            else
                types[i] = LAYER_V4_HCA;
        }
    }

    *compress_ratios_out = ratios;
    return types;
}

static ModelArch detect_arch(const JsonDoc *doc) {
    int idx = json_get(doc, 0, "model_type");
    if (idx >= 0) {
        char buf[64];
        if (json_string(doc, idx, buf, sizeof(buf))) {
            if (strcmp(buf, "deepseek_v4") == 0)
                return ARCH_DEEPSEEK_V4;
        }
    }

    int arch_idx = json_get(doc, 0, "architectures");
    if (arch_idx >= 0) {
        int el = json_array_get(doc, arch_idx, 0);
        if (el >= 0) {
            char buf[128];
            if (json_string(doc, el, buf, sizeof(buf))) {
                if (strstr(buf, "DeepseekV4"))
                    return ARCH_DEEPSEEK_V4;
            }
        }
    }

    return ARCH_QWEN35;
}

static void parse_common(ModelConfig *cfg, const JsonDoc *doc, int obj) {
    int idx;

    if ((idx = json_get(doc, obj, "hidden_size")) >= 0)
        cfg->hidden_size = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "num_hidden_layers")) >= 0)
        cfg->num_hidden_layers = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "num_attention_heads")) >= 0)
        cfg->num_attention_heads = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "num_key_value_heads")) >= 0)
        cfg->num_key_value_heads = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "head_dim")) >= 0)
        cfg->head_dim = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "vocab_size")) >= 0)
        cfg->vocab_size = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "max_position_embeddings")) >= 0)
        cfg->max_position_embeddings = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "num_experts_per_tok")) >= 0)
        cfg->num_experts_per_tok = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "moe_intermediate_size")) >= 0)
        cfg->moe_intermediate_size = json_int(doc, idx);

    cfg->rms_norm_eps = 1e-6f;
    if ((idx = json_get(doc, obj, "rms_norm_eps")) >= 0)
        cfg->rms_norm_eps = (float)json_number(doc, idx);

    if ((idx = json_get(doc, obj, "eos_token_id")) >= 0)
        cfg->eos_token_id = json_int(doc, idx);

    if ((idx = json_get(doc, obj, "mtp_num_hidden_layers")) >= 0)
        cfg->mtp_num_hidden_layers = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "num_nextn_predict_layers")) >= 0)
        cfg->mtp_num_hidden_layers = json_int(doc, idx);
}

static void parse_qwen35(ModelConfig *cfg, const JsonDoc *doc, int text_idx) {
    int idx;

    if ((idx = json_get(doc, text_idx, "num_experts")) >= 0)
        cfg->num_experts = json_int(doc, idx);
    if ((idx = json_get(doc, text_idx, "shared_expert_intermediate_size")) >= 0)
        cfg->shared_expert_intermediate_size = json_int(doc, idx);

    cfg->full_attention_interval = 4;
    if ((idx = json_get(doc, text_idx, "full_attention_interval")) >= 0)
        cfg->full_attention_interval = json_int(doc, idx);

    int lt_idx = json_get(doc, text_idx, "layer_types");
    if (lt_idx >= 0 && cfg->num_hidden_layers > 0)
        cfg->layer_types = parse_qwen_layer_types(doc, lt_idx, cfg->num_hidden_layers);

    if ((idx = json_get(doc, text_idx, "attn_output_gate")) >= 0)
        cfg->attn_output_gate = json_bool(doc, idx);

    if ((idx = json_get(doc, text_idx, "linear_conv_kernel_dim")) >= 0)
        cfg->linear_attn.linear_conv_kernel_dim = json_int(doc, idx);
    if ((idx = json_get(doc, text_idx, "linear_key_head_dim")) >= 0)
        cfg->linear_attn.linear_key_head_dim = json_int(doc, idx);
    if ((idx = json_get(doc, text_idx, "linear_num_key_heads")) >= 0)
        cfg->linear_attn.linear_num_key_heads = json_int(doc, idx);
    if ((idx = json_get(doc, text_idx, "linear_num_value_heads")) >= 0)
        cfg->linear_attn.linear_num_value_heads = json_int(doc, idx);
    if ((idx = json_get(doc, text_idx, "linear_value_head_dim")) >= 0)
        cfg->linear_attn.linear_value_head_dim = json_int(doc, idx);

    int rope_idx = json_get(doc, text_idx, "rope_parameters");
    parse_rope(doc, rope_idx, &cfg->rope);

    snprintf(cfg->model_name, sizeof(cfg->model_name),
             "qwen3.5-%dL-%dE-%dH",
             cfg->num_hidden_layers, cfg->num_experts, cfg->hidden_size);
}

static void parse_deepseek_v4(ModelConfig *cfg, const JsonDoc *doc, int obj) {
    int idx;

    // V4 uses n_routed_experts, not num_experts
    if ((idx = json_get(doc, obj, "n_routed_experts")) >= 0)
        cfg->num_experts = json_int(doc, idx);

    // V4 shared expert uses same moe_intermediate_size
    cfg->shared_expert_intermediate_size = cfg->moe_intermediate_size;

    // V4 attention config
    if ((idx = json_get(doc, obj, "q_lora_rank")) >= 0)
        cfg->v4.q_lora_rank = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "o_lora_rank")) >= 0)
        cfg->v4.o_lora_rank = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "o_groups")) >= 0)
        cfg->v4.o_groups = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "qk_rope_head_dim")) >= 0)
        cfg->v4.qk_rope_head_dim = json_int(doc, idx);

    // Compression
    if ((idx = json_get(doc, obj, "compress_rope_theta")) >= 0)
        cfg->v4.compress_rope_theta = json_number(doc, idx);

    int cr_idx = json_get(doc, obj, "compress_ratios");
    if (cr_idx >= 0 && cfg->num_hidden_layers > 0)
        cfg->layer_types = parse_v4_layer_types(doc, cr_idx, cfg->num_hidden_layers,
                                                 &cfg->v4.compress_ratios);

    // Indexer
    if ((idx = json_get(doc, obj, "index_head_dim")) >= 0)
        cfg->v4.index_head_dim = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "index_n_heads")) >= 0)
        cfg->v4.index_n_heads = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "index_topk")) >= 0)
        cfg->v4.index_topk = json_int(doc, idx);

    // Hyper-Connections
    cfg->v4.hc_mult = 4;
    cfg->v4.hc_eps = 1e-6f;
    cfg->v4.hc_sinkhorn_iters = 20;
    if ((idx = json_get(doc, obj, "hc_mult")) >= 0)
        cfg->v4.hc_mult = json_int(doc, idx);
    if ((idx = json_get(doc, obj, "hc_eps")) >= 0)
        cfg->v4.hc_eps = (float)json_number(doc, idx);
    if ((idx = json_get(doc, obj, "hc_sinkhorn_iters")) >= 0)
        cfg->v4.hc_sinkhorn_iters = json_int(doc, idx);

    // MoE specifics
    cfg->v4.num_hash_layers = 3;
    if ((idx = json_get(doc, obj, "num_hash_layers")) >= 0)
        cfg->v4.num_hash_layers = json_int(doc, idx);

    cfg->v4.window_size = 128;

    cfg->v4.route_scale = 1.0;
    if ((idx = json_get(doc, obj, "routed_scaling_factor")) >= 0)
        cfg->v4.route_scale = json_number(doc, idx);

    // RoPE — V4 uses rope_theta at root level
    cfg->rope.rope_theta = 10000.0;
    if ((idx = json_get(doc, obj, "rope_theta")) >= 0)
        cfg->rope.rope_theta = json_number(doc, idx);
    cfg->rope.partial_rotary_factor = 1.0f;

    snprintf(cfg->model_name, sizeof(cfg->model_name),
             "deepseek-v4-%dL-%dE-%dH",
             cfg->num_hidden_layers, cfg->num_experts, cfg->hidden_size);
}

bool config_load(ModelConfig *cfg, const char *path) {
    memset(cfg, 0, sizeof(*cfg));

    size_t len;
    char *json = read_file(path, &len);
    if (!json) {
        LOG_ERROR("config: failed to read %s", path);
        return false;
    }

    // V4 configs are large (per-layer quantization overrides)
    int max_tokens = 16384;
    JsonToken *tokens = malloc((size_t)max_tokens * sizeof(JsonToken));
    if (!tokens) {
        LOG_ERROR("config: failed to allocate JSON tokens");
        free(json);
        return false;
    }

    JsonDoc doc;
    if (!json_parse(&doc, json, len, tokens, max_tokens)) {
        LOG_ERROR("config: failed to parse JSON in %s", path);
        free(tokens);
        free(json);
        return false;
    }

    if (doc.tokens[0].type != JSON_OBJECT) {
        LOG_ERROR("config: root is not an object");
        free(tokens);
        free(json);
        return false;
    }

    cfg->arch = detect_arch(&doc);

    // Find the config object — text_config if nested, root otherwise
    int text_idx = json_get(&doc, 0, "text_config");
    if (text_idx < 0)
        text_idx = 0;

    parse_common(cfg, &doc, text_idx);

    if (cfg->arch == ARCH_DEEPSEEK_V4) {
        // V4 fields may be at root or text_config — check both
        int v4_obj = (text_idx != 0) ? text_idx : 0;
        parse_deepseek_v4(cfg, &doc, v4_obj);
    } else {
        parse_qwen35(cfg, &doc, text_idx);
    }

    free(tokens);
    free(json);

    LOG_INFO("config: loaded %s (arch=%s)", cfg->model_name,
             cfg->arch == ARCH_DEEPSEEK_V4 ? "deepseek_v4" : "qwen3.5");
    return true;
}

void config_free(ModelConfig *cfg) {
    free(cfg->layer_types);
    cfg->layer_types = NULL;
    free(cfg->v4.compress_ratios);
    cfg->v4.compress_ratios = NULL;
}

void config_print(const ModelConfig *cfg) {
    LOG_INFO("config: model=%s, arch=%s", cfg->model_name,
             cfg->arch == ARCH_DEEPSEEK_V4 ? "deepseek_v4" : "qwen3.5");
    LOG_INFO("config: hidden_size=%d, layers=%d, heads=%d, kv_heads=%d",
             cfg->hidden_size, cfg->num_hidden_layers,
             cfg->num_attention_heads, cfg->num_key_value_heads);
    LOG_INFO("config: head_dim=%d, vocab=%d, max_pos=%d",
             cfg->head_dim, cfg->vocab_size, cfg->max_position_embeddings);
    LOG_INFO("config: experts=%d, active=%d, moe_dim=%d",
             cfg->num_experts, cfg->num_experts_per_tok,
             cfg->moe_intermediate_size);
    LOG_INFO("config: rope_theta=%.0f", cfg->rope.rope_theta);

    if (cfg->arch == ARCH_DEEPSEEK_V4) {
        int sw = 0, csa = 0, hca = 0;
        if (cfg->layer_types) {
            for (int i = 0; i < cfg->num_hidden_layers; i++) {
                switch (cfg->layer_types[i]) {
                    case LAYER_V4_SLIDING_WINDOW: sw++; break;
                    case LAYER_V4_CSA: csa++; break;
                    case LAYER_V4_HCA: hca++; break;
                    default: break;
                }
            }
        }
        LOG_INFO("config: v4 layers: %d sliding + %d CSA + %d HCA", sw, csa, hca);
        LOG_INFO("config: v4 q_lora=%d, o_lora=%d, o_groups=%d, hc_mult=%d",
                 cfg->v4.q_lora_rank, cfg->v4.o_lora_rank,
                 cfg->v4.o_groups, cfg->v4.hc_mult);
        LOG_INFO("config: v4 index: heads=%d, dim=%d, topk=%d",
                 cfg->v4.index_n_heads, cfg->v4.index_head_dim, cfg->v4.index_topk);
    } else {
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
}
