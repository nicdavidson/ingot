#define _POSIX_C_SOURCE 200809L

#include "server/routes.h"
#include "server/json_write.h"
#include "server/sse.h"
#include "inference/inference.h"
#include "chat/template.h"
#include "chat/tool_parser.h"
#include "tokenizer/tokenizer.h"
#include "util/json_parse.h"
#include "util/log.h"
#include "util/timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

// Send a simple HTTP response
static void send_response(int fd, int status, const char *status_text,
                          const char *content_type,
                          const char *body, size_t body_len) {
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: keep-alive\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    ssize_t r = write(fd, header, (size_t)hlen);
    (void)r;
    if (body && body_len > 0) { r = write(fd, body, body_len); (void)r; }
}

static void send_json(int fd, int status, const char *status_text,
                      const char *json) {
    send_response(fd, status, status_text, "application/json",
                  json, strlen(json));
}

// --- Route handlers ---

static void handle_health(int fd) {
    char buf[256];
    JsonWriter w;
    jw_init(&w, buf, sizeof(buf));
    jw_object_start(&w);
    jw_key(&w, "status"); jw_string(&w, "ok");
    jw_key(&w, "version"); jw_string(&w, "0.1.0");
    jw_object_end(&w);
    buf[jw_length(&w)] = '\0';
    send_json(fd, 200, "OK", buf);
}

static void handle_models(int fd, Model *model) {
    const ModelConfig *cfg = model_config(model);
    char buf[1024];
    JsonWriter w;
    jw_init(&w, buf, sizeof(buf));
    jw_object_start(&w);
    jw_key(&w, "object"); jw_string(&w, "list");
    jw_key(&w, "data");
    jw_array_start(&w);
    jw_object_start(&w);
    jw_key(&w, "id"); jw_string(&w, cfg->model_name);
    jw_key(&w, "object"); jw_string(&w, "model");
    jw_key(&w, "owned_by"); jw_string(&w, "ingot");
    jw_object_end(&w);
    jw_array_end(&w);
    jw_object_end(&w);
    buf[jw_length(&w)] = '\0';
    send_json(fd, 200, "OK", buf);
}

// Streaming callback context
typedef struct {
    int         fd;
    const char *model_name;
    char       *chunk_id;
    int         token_count;
    char        content_buf[65536];
    int         content_len;
    bool        stream;
} StreamCtx;

static void stream_callback(int32_t token_id, const char *text, void *userdata) {
    (void)token_id;
    StreamCtx *sctx = userdata;

    // Accumulate content for non-streaming mode
    size_t tlen = strlen(text);
    if (sctx->content_len + (int)tlen < (int)sizeof(sctx->content_buf) - 1) {
        memcpy(sctx->content_buf + sctx->content_len, text, tlen);
        sctx->content_len += (int)tlen;
        sctx->content_buf[sctx->content_len] = '\0';
    }

    if (sctx->stream) {
        // Send SSE chunk
        char chunk[4096];
        int clen = sse_format_chunk(chunk, sizeof(chunk),
                                    sctx->model_name, text, NULL,
                                    sctx->chunk_id, 0);
        sse_write_event(sctx->fd, chunk, (size_t)clen);
    }

    sctx->token_count++;
}

static void handle_chat_completions(int fd, const HttpRequest *req, Model *model) {
    if (!req->body) {
        send_json(fd, 400, "Bad Request", "{\"error\":\"missing request body\"}");
        return;
    }

    // Parse request JSON
    JsonToken tokens[4096];
    JsonDoc doc;
    if (!json_parse(&doc, req->body, req->body_len, tokens, 4096)) {
        send_json(fd, 400, "Bad Request", "{\"error\":\"invalid JSON\"}");
        return;
    }

    // Extract parameters
    bool stream = false;
    int max_tokens_val = 2048;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k_val = 40;

    int idx;
    if ((idx = json_get(&doc, 0, "stream")) >= 0)
        stream = json_bool(&doc, idx);
    if ((idx = json_get(&doc, 0, "max_tokens")) >= 0)
        max_tokens_val = json_int(&doc, idx);
    if ((idx = json_get(&doc, 0, "temperature")) >= 0)
        temperature = (float)json_number(&doc, idx);
    if ((idx = json_get(&doc, 0, "top_p")) >= 0)
        top_p = (float)json_number(&doc, idx);
    if ((idx = json_get(&doc, 0, "top_k")) >= 0)
        top_k_val = json_int(&doc, idx);

    // Parse messages array
    int msgs_idx = json_get(&doc, 0, "messages");
    if (msgs_idx < 0) {
        send_json(fd, 400, "Bad Request", "{\"error\":\"missing messages\"}");
        return;
    }

    int num_messages = json_array_len(&doc, msgs_idx);
    ChatMessage *messages = calloc((size_t)num_messages, sizeof(ChatMessage));

    for (int i = 0; i < num_messages; i++) {
        int msg_idx = json_array_get(&doc, msgs_idx, i);
        if (msg_idx < 0) continue;

        int role_idx = json_get(&doc, msg_idx, "role");
        if (role_idx >= 0) {
            char role_str[32];
            json_string(&doc, role_idx, role_str, sizeof(role_str));
            messages[i].role = template_parse_role(role_str);
        }

        int content_idx = json_get(&doc, msg_idx, "content");
        if (content_idx >= 0 && doc.tokens[content_idx].type == JSON_STRING) {
            char *content = malloc(doc.tokens[content_idx].len + 1);
            json_string(&doc, content_idx, content, doc.tokens[content_idx].len + 1);
            messages[i].content = content;
        }
    }

    // Parse tools if present
    int tools_idx = json_get(&doc, 0, "tools");
    ToolDef *tools = NULL;
    int num_tools = 0;
    if (tools_idx >= 0) {
        num_tools = json_array_len(&doc, tools_idx);
        tools = calloc((size_t)num_tools, sizeof(ToolDef));
        for (int i = 0; i < num_tools; i++) {
            int tool_idx = json_array_get(&doc, tools_idx, i);
            if (tool_idx >= 0) {
                // Store the raw JSON substring for the tool definition
                tools[i].json = strndup(doc.tokens[tool_idx].start,
                                        doc.tokens[tool_idx].len);
            }
        }
    }

    // Apply chat template
    int prompt_size = template_apply(messages, num_messages,
                                     tools, num_tools,
                                     true, true, NULL, 0);
    char *prompt = malloc((size_t)prompt_size + 1);
    template_apply(messages, num_messages, tools, num_tools,
                   true, true, prompt, (size_t)prompt_size + 1);

    // Tokenize prompt
    const Tokenizer *tok = model_tokenizer(model);
    int32_t *prompt_tokens = malloc(sizeof(int32_t) * (size_t)(prompt_size + 100));
    int num_prompt_tokens = tokenizer_encode(tok, prompt, (size_t)prompt_size,
                                             prompt_tokens, prompt_size + 100);

    LOG_INFO("api: chat completions (messages=%d, prompt_tokens=%d, max=%d, stream=%d)",
             num_messages, num_prompt_tokens, max_tokens_val, stream);

    // Generate a request ID
    char chunk_id[64];
    snprintf(chunk_id, sizeof(chunk_id), "chatcmpl-%lld", (long long)time(NULL));

    const ModelConfig *cfg = model_config(model);

    // Create inference context
    InferenceContext *ictx = inference_create(model);

    StreamCtx sctx = {
        .fd = fd,
        .model_name = cfg->model_name,
        .chunk_id = chunk_id,
        .stream = stream,
    };

    if (stream) {
        // Send initial role chunk
        sse_write_headers(fd);
        char role_chunk[1024];
        int rclen = sse_format_chunk(role_chunk, sizeof(role_chunk),
                                     cfg->model_name, NULL, NULL, chunk_id, 0);
        sse_write_event(fd, role_chunk, (size_t)rclen);
    }

    // Run inference
    int generated = inference_generate(ictx, prompt_tokens, num_prompt_tokens,
                                       max_tokens_val,
                                       temperature, top_p, top_k_val,
                                       stream_callback, &sctx);

    if (stream) {
        // Send final chunk with finish_reason
        char final_chunk[1024];
        int fclen = sse_format_chunk(final_chunk, sizeof(final_chunk),
                                     cfg->model_name, NULL, "stop", chunk_id, 0);
        sse_write_event(fd, final_chunk, (size_t)fclen);
        sse_write_done(fd);
    } else {
        // Non-streaming: build complete response
        // Strip think tags
        char clean_content[65536];
        int clean_len = tool_parser_strip_think(
            sctx.content_buf, (size_t)sctx.content_len,
            clean_content, sizeof(clean_content));

        // Check for tool calls
        ParsedToolCall tool_calls[16];
        int num_tool_calls = tool_parser_parse(clean_content, (size_t)clean_len,
                                               tool_calls, 16);

        char response[131072];
        JsonWriter w;
        jw_init(&w, response, sizeof(response));
        jw_object_start(&w);

        jw_key(&w, "id"); jw_string(&w, chunk_id);
        jw_key(&w, "object"); jw_string(&w, "chat.completion");
        jw_key(&w, "created"); jw_int64(&w, (long long)time(NULL));
        jw_key(&w, "model"); jw_string(&w, cfg->model_name);

        jw_key(&w, "choices");
        jw_array_start(&w);
        jw_object_start(&w);
        jw_key(&w, "index"); jw_int(&w, 0);

        jw_key(&w, "message");
        jw_object_start(&w);
        jw_key(&w, "role"); jw_string(&w, "assistant");

        if (num_tool_calls > 0) {
            jw_key(&w, "content"); jw_null(&w);
            jw_key(&w, "tool_calls");
            jw_array_start(&w);
            for (int i = 0; i < num_tool_calls; i++) {
                jw_object_start(&w);
                char call_id[32];
                snprintf(call_id, sizeof(call_id), "call_%d", i + 1);
                jw_key(&w, "id"); jw_string(&w, call_id);
                jw_key(&w, "type"); jw_string(&w, "function");
                jw_key(&w, "function");
                jw_object_start(&w);
                jw_key(&w, "name"); jw_string(&w, tool_calls[i].function_name);
                jw_key(&w, "arguments"); jw_string(&w, tool_calls[i].arguments_json);
                jw_object_end(&w);
                jw_object_end(&w);
            }
            jw_array_end(&w);
        } else {
            jw_key(&w, "content"); jw_string(&w, clean_content);
        }
        jw_object_end(&w); // message

        jw_key(&w, "finish_reason");
        jw_string(&w, num_tool_calls > 0 ? "tool_calls" : "stop");
        jw_object_end(&w); // choice
        jw_array_end(&w); // choices

        jw_key(&w, "usage");
        jw_object_start(&w);
        jw_key(&w, "prompt_tokens"); jw_int(&w, num_prompt_tokens);
        jw_key(&w, "completion_tokens"); jw_int(&w, generated);
        jw_key(&w, "total_tokens"); jw_int(&w, num_prompt_tokens + generated);
        jw_object_end(&w);

        jw_object_end(&w);
        response[jw_length(&w)] = '\0';

        send_json(fd, 200, "OK", response);
    }

    // Cleanup
    inference_free(ictx);
    free(prompt_tokens);
    free(prompt);
    for (int i = 0; i < num_messages; i++) {
        free((void *)messages[i].content);
    }
    free(messages);
    for (int i = 0; i < num_tools; i++) {
        free((void *)tools[i].json);
    }
    free(tools);
}

static void handle_options(int fd) {
    const char *resp =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "\r\n";
    ssize_t r = write(fd, resp, strlen(resp));
    (void)r;
}

void route_handle(int fd, const HttpRequest *req, Model *model) {
    LOG_INFO("server: %s %s", req->method, req->path);

    // CORS preflight
    if (strcmp(req->method, "OPTIONS") == 0) {
        handle_options(fd);
        return;
    }

    if (strcmp(req->path, "/health") == 0 ||
        strcmp(req->path, "/v1/health") == 0) {
        handle_health(fd);
    } else if (strcmp(req->path, "/v1/models") == 0) {
        handle_models(fd, model);
    } else if (strcmp(req->path, "/v1/chat/completions") == 0) {
        if (strcmp(req->method, "POST") != 0) {
            send_json(fd, 405, "Method Not Allowed",
                      "{\"error\":\"use POST\"}");
            return;
        }
        handle_chat_completions(fd, req, model);
    } else {
        send_json(fd, 404, "Not Found",
                  "{\"error\":\"not found\"}");
    }
}
