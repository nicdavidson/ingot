#include "chat/tool_parser.h"

#include <string.h>
#include <stdio.h>

// Find substring in bounded text
static const char *find_str(const char *haystack, size_t hay_len,
                            const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen > hay_len) return NULL;
    for (size_t i = 0; i <= hay_len - nlen; i++) {
        if (memcmp(haystack + i, needle, nlen) == 0)
            return haystack + i;
    }
    return NULL;
}

// Parse a single <tool_call>...</tool_call> block
static bool parse_one_call(const char *start, const char *end,
                           ParsedToolCall *call) {
    memset(call, 0, sizeof(*call));

    // Find <function=NAME>
    const char *func_start = find_str(start, (size_t)(end - start), "<function=");
    if (!func_start) return false;
    func_start += 10; // skip "<function="

    const char *func_end = memchr(func_start, '>', (size_t)(end - func_start));
    if (!func_end) return false;

    size_t name_len = (size_t)(func_end - func_start);
    if (name_len >= sizeof(call->function_name)) return false;
    memcpy(call->function_name, func_start, name_len);
    call->function_name[name_len] = '\0';

    // Find </function>
    const char *func_body_end = find_str(func_end, (size_t)(end - func_end),
                                         "</function>");
    if (!func_body_end) func_body_end = end;

    // Parse <parameter=NAME>VALUE</parameter> blocks into JSON
    int json_pos = 0;
    call->arguments_json[json_pos++] = '{';

    const char *pos = func_end + 1;
    bool first_param = true;
    while (pos < func_body_end) {
        const char *param_start = find_str(pos, (size_t)(func_body_end - pos),
                                           "<parameter=");
        if (!param_start) break;
        param_start += 11; // skip "<parameter="

        const char *param_name_end = memchr(param_start, '>',
                                            (size_t)(func_body_end - param_start));
        if (!param_name_end) break;

        const char *param_val_start = param_name_end + 1;
        // Skip leading newline after >
        if (param_val_start < func_body_end && *param_val_start == '\n')
            param_val_start++;

        const char *param_end = find_str(param_val_start,
                                         (size_t)(func_body_end - param_val_start),
                                         "\n</parameter>");
        if (!param_end)
            param_end = find_str(param_val_start,
                                 (size_t)(func_body_end - param_val_start),
                                 "</parameter>");
        if (!param_end) break;

        size_t pname_len = (size_t)(param_name_end - param_start);
        size_t pval_len = (size_t)(param_end - param_val_start);

        if (!first_param) {
            call->arguments_json[json_pos++] = ',';
        }
        first_param = false;

        // Write "name": "value" (with JSON escaping for the value)
        json_pos += snprintf(call->arguments_json + json_pos,
                             sizeof(call->arguments_json) - (size_t)json_pos,
                             "\"%.*s\":", (int)pname_len, param_start);

        // Simple JSON string escaping
        call->arguments_json[json_pos++] = '"';
        for (size_t i = 0; i < pval_len && (size_t)json_pos + 4 < sizeof(call->arguments_json); i++) {
            char c = param_val_start[i];
            switch (c) {
                case '"':  call->arguments_json[json_pos++] = '\\';
                           call->arguments_json[json_pos++] = '"'; break;
                case '\\': call->arguments_json[json_pos++] = '\\';
                           call->arguments_json[json_pos++] = '\\'; break;
                case '\n': call->arguments_json[json_pos++] = '\\';
                           call->arguments_json[json_pos++] = 'n'; break;
                case '\r': call->arguments_json[json_pos++] = '\\';
                           call->arguments_json[json_pos++] = 'r'; break;
                case '\t': call->arguments_json[json_pos++] = '\\';
                           call->arguments_json[json_pos++] = 't'; break;
                default:   call->arguments_json[json_pos++] = c; break;
            }
        }
        call->arguments_json[json_pos++] = '"';

        pos = param_end + strlen("</parameter>");
    }

    call->arguments_json[json_pos++] = '}';
    call->arguments_json[json_pos] = '\0';

    return true;
}

int tool_parser_parse(const char *text, size_t text_len,
                      ParsedToolCall *calls, int max_calls) {
    int count = 0;
    const char *pos = text;
    size_t remaining = text_len;

    while (count < max_calls && remaining > 0) {
        const char *tc_start = find_str(pos, remaining, "<tool_call>");
        if (!tc_start) break;

        const char *tc_end = find_str(tc_start + 11,
                                      remaining - (size_t)(tc_start - pos) - 11,
                                      "</tool_call>");
        if (!tc_end) break;

        if (parse_one_call(tc_start + 11, tc_end, &calls[count])) {
            count++;
        }

        pos = tc_end + 12;
        remaining = text_len - (size_t)(pos - text);
    }

    return count;
}

bool tool_parser_has_tool_calls(const char *text, size_t text_len) {
    return find_str(text, text_len, "<tool_call>") != NULL;
}

int tool_parser_strip_think(const char *text, size_t text_len,
                            char *buf, size_t buf_size) {
    int written = 0;
    const char *pos = text;
    size_t remaining = text_len;

    while (remaining > 0) {
        const char *think_start = find_str(pos, remaining, "<think>");
        if (!think_start) {
            // Copy remainder
            size_t to_copy = remaining;
            if ((size_t)written + to_copy >= buf_size)
                to_copy = buf_size - (size_t)written - 1;
            memcpy(buf + written, pos, to_copy);
            written += (int)to_copy;
            break;
        }

        // Copy text before <think>
        size_t before = (size_t)(think_start - pos);
        if (before > 0 && (size_t)written + before < buf_size) {
            memcpy(buf + written, pos, before);
            written += (int)before;
        }

        // Find </think> and skip everything in between
        const char *think_end = find_str(think_start + 7,
                                         remaining - (size_t)(think_start - pos) - 7,
                                         "</think>");
        if (think_end) {
            pos = think_end + 8;
            // Skip a leading newline after </think> if present
            if (pos < text + text_len && *pos == '\n') pos++;
            remaining = text_len - (size_t)(pos - text);
        } else {
            // Unclosed <think> — skip to end
            break;
        }
    }

    if ((size_t)written < buf_size) buf[written] = '\0';
    return written;
}
