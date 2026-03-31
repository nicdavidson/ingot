#ifndef INGOT_TEMPLATE_H
#define INGOT_TEMPLATE_H

#include <stdbool.h>
#include <stddef.h>

// Message roles
typedef enum {
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL,
} ChatRole;

// A single tool call from an assistant message
typedef struct {
    const char *id;        // "call_1", etc.
    const char *name;      // function name
    const char *arguments; // JSON string of arguments
} ToolCall;

// A single chat message
typedef struct {
    ChatRole    role;
    const char *content;
    const char *tool_call_id;  // for tool responses

    // Tool calls (assistant messages only)
    ToolCall   *tool_calls;
    int         num_tool_calls;
} ChatMessage;

// Tool definition (for system prompt injection)
typedef struct {
    const char *json; // full JSON tool definition
} ToolDef;

// Format a conversation into the Qwen ChatML prompt string.
// Writes into buf, returns number of bytes written (not including NUL).
// If buf is NULL, returns required size.
int template_apply(const ChatMessage *messages, int num_messages,
                   const ToolDef *tools, int num_tools,
                   bool add_generation_prompt,
                   bool enable_thinking,
                   char *buf, size_t buf_size);

// Parse role string to enum. Returns -1 on unknown.
ChatRole template_parse_role(const char *role);

#endif
