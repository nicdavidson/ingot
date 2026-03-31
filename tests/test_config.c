#define _POSIX_C_SOURCE 200809L

#include "config/config.h"
#include "util/log.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Write a test config to a temp file, parse it, verify fields
static void test_397b_config(void) {
    const char *json =
        "{\n"
        "  \"model_type\": \"qwen3_5_moe\",\n"
        "  \"text_config\": {\n"
        "    \"hidden_size\": 4096,\n"
        "    \"num_hidden_layers\": 60,\n"
        "    \"num_attention_heads\": 32,\n"
        "    \"num_key_value_heads\": 2,\n"
        "    \"head_dim\": 256,\n"
        "    \"vocab_size\": 248320,\n"
        "    \"max_position_embeddings\": 262144,\n"
        "    \"num_experts\": 512,\n"
        "    \"num_experts_per_tok\": 10,\n"
        "    \"moe_intermediate_size\": 1024,\n"
        "    \"shared_expert_intermediate_size\": 1024,\n"
        "    \"rms_norm_eps\": 1e-06,\n"
        "    \"eos_token_id\": 248044,\n"
        "    \"full_attention_interval\": 4,\n"
        "    \"attn_output_gate\": true,\n"
        "    \"linear_conv_kernel_dim\": 4,\n"
        "    \"linear_key_head_dim\": 128,\n"
        "    \"linear_num_key_heads\": 16,\n"
        "    \"linear_num_value_heads\": 64,\n"
        "    \"linear_value_head_dim\": 128,\n"
        "    \"mtp_num_hidden_layers\": 1,\n"
        "    \"layer_types\": [\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\",\n"
        "      \"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\"\n"
        "    ],\n"
        "    \"rope_parameters\": {\n"
        "      \"rope_theta\": 10000000,\n"
        "      \"partial_rotary_factor\": 0.25,\n"
        "      \"mrope_interleaved\": true,\n"
        "      \"mrope_section\": [11, 11, 10]\n"
        "    }\n"
        "  }\n"
        "}\n";

    // Write to temp file
    const char *path = "/tmp/ingot_test_config.json";
    FILE *f = fopen(path, "w");
    assert(f);
    fputs(json, f);
    fclose(f);

    ModelConfig cfg;
    bool ok = config_load(&cfg, path);
    assert(ok);

    // Verify all fields
    assert(cfg.hidden_size == 4096);
    assert(cfg.num_hidden_layers == 60);
    assert(cfg.num_attention_heads == 32);
    assert(cfg.num_key_value_heads == 2);
    assert(cfg.head_dim == 256);
    assert(cfg.vocab_size == 248320);
    assert(cfg.max_position_embeddings == 262144);
    assert(cfg.num_experts == 512);
    assert(cfg.num_experts_per_tok == 10);
    assert(cfg.moe_intermediate_size == 1024);
    assert(cfg.shared_expert_intermediate_size == 1024);
    assert(cfg.eos_token_id == 248044);
    assert(cfg.full_attention_interval == 4);
    assert(cfg.attn_output_gate == true);
    assert(cfg.mtp_num_hidden_layers == 1);

    // Linear attention config
    assert(cfg.linear_attn.linear_conv_kernel_dim == 4);
    assert(cfg.linear_attn.linear_key_head_dim == 128);
    assert(cfg.linear_attn.linear_num_key_heads == 16);
    assert(cfg.linear_attn.linear_num_value_heads == 64);
    assert(cfg.linear_attn.linear_value_head_dim == 128);

    // RoPE config
    assert(cfg.rope.rope_theta == 10000000.0);
    assert(cfg.rope.partial_rotary_factor == 0.25f);
    assert(cfg.rope.mrope_interleaved == true);
    assert(cfg.rope.mrope_section[0] == 11);
    assert(cfg.rope.mrope_section[1] == 11);
    assert(cfg.rope.mrope_section[2] == 10);

    // Layer types
    assert(cfg.layer_types != NULL);
    int linear = 0, full = 0;
    for (int i = 0; i < 60; i++) {
        if (cfg.layer_types[i] == LAYER_LINEAR_ATTENTION) linear++;
        else full++;
    }
    assert(linear == 45);
    assert(full == 15);

    config_print(&cfg);
    config_free(&cfg);

    remove(path);
    printf("test_397b_config: PASSED\n");
}

int main(void) {
    log_init();
    test_397b_config();
    printf("\nAll config tests passed.\n");
    return 0;
}
