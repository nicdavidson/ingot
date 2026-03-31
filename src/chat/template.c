#include "chat/template.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Buffered writer — writes to buf if available, always tracks total length
typedef struct {
    char  *buf;
    size_t cap;
    size_t len;
} Writer;

static void w_write(Writer *w, const char *s, size_t n) {
    if (w->buf && w->len + n < w->cap) {
        memcpy(w->buf + w->len, s, n);
    }
    w->len += n;
}

static void w_str(Writer *w, const char *s) {
    w_write(w, s, strlen(s));
}

ChatRole template_parse_role(const char *role) {
    if (strcmp(role, "system") == 0)    return ROLE_SYSTEM;
    if (strcmp(role, "user") == 0)      return ROLE_USER;
    if (strcmp(role, "assistant") == 0) return ROLE_ASSISTANT;
    if (strcmp(role, "tool") == 0)      return ROLE_TOOL;
    return (ChatRole)-1;
}

static void write_tool_system(Writer *w, const ToolDef *tools, int num_tools,
                              const char *system_content) {
    w_str(w, "<|im_start|>system\n");
    w_str(w, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (int i = 0; i < num_tools; i++) {
        w_str(w, "\n");
        w_str(w, tools[i].json);
    }
    w_str(w, "\n</tools>");
    w_str(w, "\n\nIf you choose to call a function ONLY reply in the "
             "following format with NO suffix:\n\n"
             "<tool_call>\n"
             "<function=example_function_name>\n"
             "<parameter=example_parameter_1>\n"
             "value_1\n"
             "</parameter>\n"
             "<parameter=example_parameter_2>\n"
             "This is the value for the second parameter\n"
             "that can span\n"
             "multiple lines\n"
             "</parameter>\n"
             "</function>\n"
             "</tool_call>\n\n"
             "<IMPORTANT>\n"
             "Reminder:\n"
             "- Function calls MUST follow the specified format: "
             "an inner <function=...></function> block must be nested "
             "within <tool_call></tool_call> XML tags\n"
             "- Required parameters MUST be specified\n"
             "- You may provide optional reasoning for your function call "
             "in natural language BEFORE the function call, but NOT after\n"
             "- If there is no function call available, answer the question "
             "like normal with your current knowledge and do not tell the "
             "user about function calls\n"
             "</IMPORTANT>");
    if (system_content && system_content[0]) {
        w_str(w, "\n\n");
        w_str(w, system_content);
    }
    w_str(w, "<|im_end|>\n");
}

static void write_tool_calls(Writer *w, const ToolCall *calls, int n,
                             bool has_content) {
    for (int i = 0; i < n; i++) {
        if (i == 0 && has_content)
            w_str(w, "\n\n<tool_call>\n<function=");
        else if (i == 0)
            w_str(w, "<tool_call>\n<function=");
        else
            w_str(w, "\n<tool_call>\n<function=");

        w_str(w, calls[i].name);
        w_str(w, ">\n");

        // Parse arguments JSON and emit as <parameter=...> blocks
        // For now, emit the raw JSON as a single parameter block
        // A proper implementation would iterate the JSON keys
        if (calls[i].arguments && calls[i].arguments[0]) {
            w_str(w, "<parameter=arguments>\n");
            w_str(w, calls[i].arguments);
            w_str(w, "\n</parameter>\n");
        }

        w_str(w, "</function>\n</tool_call>");
    }
}

int template_apply(const ChatMessage *messages, int num_messages,
                   const ToolDef *tools, int num_tools,
                   bool add_generation_prompt,
                   bool enable_thinking,
                   char *buf, size_t buf_size) {
    Writer w = { .buf = buf, .cap = buf_size, .len = 0 };

    bool has_tools = (tools && num_tools > 0);
    bool system_handled = false;

    // Handle system message + tools
    if (has_tools) {
        const char *sys_content = NULL;
        if (num_messages > 0 && messages[0].role == ROLE_SYSTEM) {
            sys_content = messages[0].content;
            system_handled = true;
        }
        write_tool_system(&w, tools, num_tools, sys_content);
    } else if (num_messages > 0 && messages[0].role == ROLE_SYSTEM) {
        w_str(&w, "<|im_start|>system\n");
        if (messages[0].content) w_str(&w, messages[0].content);
        w_str(&w, "<|im_end|>\n");
        system_handled = true;
    }

    // Process remaining messages
    bool prev_was_tool = false;
    for (int i = (system_handled ? 1 : 0); i < num_messages; i++) {
        const ChatMessage *msg = &messages[i];

        switch (msg->role) {
            case ROLE_USER:
                w_str(&w, "<|im_start|>user\n");
                if (msg->content) w_str(&w, msg->content);
                w_str(&w, "<|im_end|>\n");
                prev_was_tool = false;
                break;

            case ROLE_ASSISTANT:
                w_str(&w, "<|im_start|>assistant\n");
                if (msg->content) w_str(&w, msg->content);
                if (msg->tool_calls && msg->num_tool_calls > 0) {
                    write_tool_calls(&w, msg->tool_calls, msg->num_tool_calls,
                                     msg->content && msg->content[0]);
                }
                w_str(&w, "<|im_end|>\n");
                prev_was_tool = false;
                break;

            case ROLE_TOOL:
                if (!prev_was_tool) {
                    w_str(&w, "<|im_start|>user");
                }
                w_str(&w, "\n<tool_response>\n");
                if (msg->content) w_str(&w, msg->content);
                w_str(&w, "\n</tool_response>");

                // Check if next message is also a tool
                if (i + 1 >= num_messages || messages[i + 1].role != ROLE_TOOL) {
                    w_str(&w, "<|im_end|>\n");
                }
                prev_was_tool = true;
                break;

            case ROLE_SYSTEM:
                // System after first position — just emit it
                w_str(&w, "<|im_start|>system\n");
                if (msg->content) w_str(&w, msg->content);
                w_str(&w, "<|im_end|>\n");
                prev_was_tool = false;
                break;
        }
    }

    // Generation prompt
    if (add_generation_prompt) {
        w_str(&w, "<|im_start|>assistant\n");
        if (!enable_thinking) {
            w_str(&w, "<think>\n\n</think>\n\n");
        } else {
            w_str(&w, "<think>\n");
        }
    }

    // NUL-terminate
    if (w.buf && w.len < w.cap) w.buf[w.len] = '\0';

    return (int)w.len;
}
