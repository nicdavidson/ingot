#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/log.h"

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
        case CMD_SERVE:
            if (!args.model_path) {
                LOG_ERROR("serve requires --model");
                return 1;
            }
            LOG_INFO("serving model: %s on port %d", args.model_path, args.port);
            // TODO: server_run(args.model_path, args.port);
            break;

        case CMD_GENERATE:
            if (!args.model_path || !args.prompt) {
                LOG_ERROR("generate requires --model and --prompt");
                return 1;
            }
            LOG_INFO("generating from: %s", args.model_path);
            // TODO: generate(args.model_path, args.prompt, args.max_tokens);
            break;

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

        case CMD_TOKENIZE:
            if (!args.model_path || !args.text) {
                LOG_ERROR("tokenize requires --model and --text");
                return 1;
            }
            LOG_INFO("tokenizing with: %s", args.model_path);
            // TODO: tokenize(args.model_path, args.text);
            break;

        case CMD_NONE:
            break;
    }

    return 0;
}
