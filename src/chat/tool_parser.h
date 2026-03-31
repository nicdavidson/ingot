#ifndef INGOT_TOOL_PARSER_H
#define INGOT_TOOL_PARSER_H

#include <stdbool.h>
#include <stddef.h>

// Parsed tool call from model output
typedef struct {
    char function_name[256];
    char arguments_json[4096]; // reconstructed JSON from <parameter> blocks
} ParsedToolCall;

// Parse model output for <tool_call> blocks.
// Returns number of tool calls found. Writes into calls array.
int tool_parser_parse(const char *text, size_t text_len,
                      ParsedToolCall *calls, int max_calls);

// Check if text contains any <tool_call> block.
bool tool_parser_has_tool_calls(const char *text, size_t text_len);

// Strip <think>...</think> blocks from text.
// Writes result into buf. Returns number of bytes written.
int tool_parser_strip_think(const char *text, size_t text_len,
                            char *buf, size_t buf_size);

#endif
