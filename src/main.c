#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/config.h"
#include "inference/inference.h"
#include "model/model.h"
#include "server/server.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"
#include "util/timer.h"

typedef enum {
    CMD_NONE,
    CMD_SERVE,
    CMD_GENERATE,
    CMD_CHAT,
    CMD_CONVERT,
    CMD_TOKENIZE,
} Command;

typedef struct {
    Command  cmd;
    char    *model_path;
    char    *prompt;
    char    *input_path;
    char    *output_path;
    char    *text;
    int      port;
    int      max_tokens;
    float    temperature;
    float    top_p;
    int      top_k;
} Args;

static void gen_print_token(int32_t id, const char *text, void *ud) {
    (void)id; (void)ud;
    printf("%s", text);
    fflush(stdout);
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "ingot — MoE inference engine for Apple Silicon\n"
        "\n"
        "Usage:\n"
        "  %s serve    --model <path> [--port 8090]\n"
        "  %s generate --model <path> --prompt <text> [--tokens 200]\n"
        "  %s chat     --model <path>\n"
        "  %s convert  --input <path> --output <path>\n"
        "  %s tokenize --model <path> --text <text>\n"
        "\n"
        "Options:\n"
        "  --model <path>     Path to model directory\n"
        "  --port <port>      Server port (default: 8090)\n"
        "  --prompt <text>    Prompt text for generation\n"
        "  --tokens <n>       Max tokens to generate (default: 200)\n"
        "  --temperature <f>  Sampling temperature (default: 0.7)\n"
        "  --top-p <f>        Top-p sampling (default: 0.9)\n"
        "  --top-k <n>        Top-k sampling (default: 40)\n"
        "  --input <path>     Input path (convert)\n"
        "  --output <path>    Output path (convert)\n"
        "  --text <text>      Text to tokenize\n"
        "\n",
        prog, prog, prog, prog, prog);
}

static Command parse_command(const char *s) {
    if (strcmp(s, "serve") == 0)    return CMD_SERVE;
    if (strcmp(s, "generate") == 0) return CMD_GENERATE;
    if (strcmp(s, "chat") == 0)     return CMD_CHAT;
    if (strcmp(s, "convert") == 0)  return CMD_CONVERT;
    if (strcmp(s, "tokenize") == 0) return CMD_TOKENIZE;
    return CMD_NONE;
}

static Args parse_args(int argc, char **argv) {
    Args args = {
        .cmd         = CMD_NONE,
        .port        = 8090,
        .max_tokens  = 200,
        .temperature = 0.7f,
        .top_p       = 0.9f,
        .top_k       = 40,
    };

    if (argc < 2) return args;

    args.cmd = parse_command(argv[1]);
    if (args.cmd == CMD_NONE) return args;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            args.port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            args.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            args.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            args.top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            args.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args.input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            args.text = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
        }
    }

    return args;
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

    if (args.cmd == CMD_NONE) {
        print_usage(argv[0]);
        return 1;
    }

    log_init();

    switch (args.cmd) {
        case CMD_SERVE: {
            if (!args.model_path) {
                LOG_ERROR("serve requires --model");
                return 1;
            }
            LOG_INFO("loading model: %s", args.model_path);
            Model *model = model_load(args.model_path);
            if (!model) {
                LOG_ERROR("failed to load model");
                return 1;
            }
            int ret = server_run(model, args.port);
            model_free(model);
            return ret;
        }

        case CMD_GENERATE: {
            if (!args.model_path || !args.prompt) {
                LOG_ERROR("generate requires --model and --prompt");
                return 1;
            }
            Model *gen_model = model_load(args.model_path);
            if (!gen_model) {
                LOG_ERROR("failed to load model");
                return 1;
            }

            // Tokenize prompt
            const Tokenizer *gen_tok = model_tokenizer(gen_model);
            int32_t prompt_tokens[8192];
            int num_prompt = tokenizer_encode(gen_tok, args.prompt,
                                              strlen(args.prompt),
                                              prompt_tokens, 8192);
            LOG_INFO("prompt: %d tokens", num_prompt);

            // Generate with print callback
            InferenceContext *gen_ctx = inference_create(gen_model);
            printf("\n");

            inference_generate(gen_ctx, prompt_tokens, num_prompt,
                              args.max_tokens,
                              args.temperature, args.top_p, args.top_k,
                              gen_print_token, NULL);

            printf("\n");
            inference_free(gen_ctx);
            model_free(gen_model);
            break;
        }

        case CMD_CHAT:
            if (!args.model_path) {
                LOG_ERROR("chat requires --model");
                return 1;
            }
            LOG_INFO("chat with: %s", args.model_path);
            // TODO: chat(args.model_path);
            break;

        case CMD_CONVERT:
            if (!args.input_path || !args.output_path) {
                LOG_ERROR("convert requires --input and --output");
                return 1;
            }
            LOG_INFO("converting: %s -> %s", args.input_path, args.output_path);
            // TODO: convert(args.input_path, args.output_path);
            break;

        case CMD_TOKENIZE: {
            if (!args.model_path || !args.text) {
                LOG_ERROR("tokenize requires --model and --text");
                return 1;
            }

            uint64_t t0 = timer_now_ns();
            Tokenizer *tok = tokenizer_load(args.model_path);
            if (!tok) {
                LOG_ERROR("failed to load tokenizer from %s", args.model_path);
                return 1;
            }
            uint64_t t1 = timer_now_ns();
            LOG_INFO("tokenizer loaded in %.1f ms (vocab=%d)",
                     timer_elapsed_ms(t0, t1), tokenizer_vocab_size(tok));

            // Encode
            size_t text_len = strlen(args.text);
            int32_t tokens[8192];
            t0 = timer_now_ns();
            int n = tokenizer_encode(tok, args.text, text_len, tokens, 8192);
            t1 = timer_now_ns();

            printf("Input:  \"%s\" (%zu bytes)\n", args.text, text_len);
            printf("Tokens: %d (encoded in %.3f ms)\n\n", n,
                   timer_elapsed_ms(t0, t1));

            // Print token IDs and their decoded text
            printf("ID        Decoded\n");
            printf("--------- -------\n");
            for (int i = 0; i < n; i++) {
                const char *decoded = tokenizer_decode(tok, tokens[i]);
                printf("%-9d \"%s\"\n", tokens[i], decoded);
            }

            // Decode back to string
            char decoded_buf[32768];
            tokenizer_decode_batch(tok, tokens, n, decoded_buf, sizeof(decoded_buf));
            printf("\nRound-trip: \"%s\"\n", decoded_buf);

            // Check round-trip
            if (strcmp(args.text, decoded_buf) == 0) {
                printf("Round-trip: OK (exact match)\n");
            } else {
                printf("Round-trip: MISMATCH\n");
                printf("  Expected: %zu bytes\n", text_len);
                printf("  Got:      %zu bytes\n", strlen(decoded_buf));
            }

            tokenizer_free(tok);
            break;
        }

        case CMD_NONE:
            break;
    }

    return 0;
}
