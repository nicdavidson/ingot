#define _POSIX_C_SOURCE 200809L

#include "chat/template.h"
#include "chat/tool_parser.h"
#include "util/log.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_simple_chat(void) {
    ChatMessage msgs[] = {
        { .role = ROLE_SYSTEM, .content = "You are a helpful assistant." },
        { .role = ROLE_USER, .content = "Hello!" },
    };

    char buf[4096];
    int len = template_apply(msgs, 2, NULL, 0, true, true, buf, sizeof(buf));
    assert(len > 0);

    // Verify structure
    assert(strstr(buf, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"));
    assert(strstr(buf, "<|im_start|>user\nHello!<|im_end|>\n"));
    assert(strstr(buf, "<|im_start|>assistant\n<think>\n"));

    printf("test_simple_chat: PASSED\n");
}

static void test_thinking_disabled(void) {
    ChatMessage msgs[] = {
        { .role = ROLE_USER, .content = "Hi" },
    };

    char buf[4096];
    template_apply(msgs, 1, NULL, 0, true, false, buf, sizeof(buf));

    // When thinking is disabled, should emit empty think block
    assert(strstr(buf, "<think>\n\n</think>\n\n"));

    printf("test_thinking_disabled: PASSED\n");
}

static void test_tool_injection(void) {
    ChatMessage msgs[] = {
        { .role = ROLE_SYSTEM, .content = "Be helpful." },
        { .role = ROLE_USER, .content = "Read /etc/hosts" },
    };
    ToolDef tools[] = {
        { .json = "{\"type\":\"function\",\"function\":{\"name\":\"read_file\"}}" },
    };

    char buf[8192];
    template_apply(msgs, 2, tools, 1, true, true, buf, sizeof(buf));

    // Tools should be injected into system message
    assert(strstr(buf, "# Tools\n\nYou have access to"));
    assert(strstr(buf, "<tools>\n{\"type\":\"function\""));
    assert(strstr(buf, "</tools>"));
    // System content should follow
    assert(strstr(buf, "Be helpful."));
    // User message should follow
    assert(strstr(buf, "<|im_start|>user\nRead /etc/hosts<|im_end|>"));

    printf("test_tool_injection: PASSED\n");
}

static void test_tool_response(void) {
    ChatMessage msgs[] = {
        { .role = ROLE_USER, .content = "Read the file" },
        { .role = ROLE_ASSISTANT, .content = "Let me read that." },
        { .role = ROLE_TOOL, .content = "file contents here", .tool_call_id = "call_1" },
    };

    char buf[4096];
    template_apply(msgs, 3, NULL, 0, true, true, buf, sizeof(buf));

    assert(strstr(buf, "<|im_start|>user\n<tool_response>\nfile contents here\n</tool_response><|im_end|>"));

    printf("test_tool_response: PASSED\n");
}

static void test_tool_parser(void) {
    const char *output =
        "I'll read that file for you.\n\n"
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\n"
        "/etc/hosts\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";

    ParsedToolCall calls[4];
    int n = tool_parser_parse(output, strlen(output), calls, 4);
    assert(n == 1);
    assert(strcmp(calls[0].function_name, "read_file") == 0);
    assert(strstr(calls[0].arguments_json, "\"path\""));
    assert(strstr(calls[0].arguments_json, "/etc/hosts"));

    printf("test_tool_parser: PASSED\n");
}

static void test_multi_tool_calls(void) {
    const char *output =
        "<tool_call>\n<function=read_file>\n"
        "<parameter=path>\n/etc/hosts\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>\n/tmp/out.txt\n</parameter>\n"
        "<parameter=content>\nhello world\n</parameter>\n"
        "</function>\n</tool_call>";

    ParsedToolCall calls[4];
    int n = tool_parser_parse(output, strlen(output), calls, 4);
    assert(n == 2);
    assert(strcmp(calls[0].function_name, "read_file") == 0);
    assert(strcmp(calls[1].function_name, "write_file") == 0);

    printf("test_multi_tool_calls: PASSED\n");
}

static void test_strip_think(void) {
    const char *text = "<think>\nLet me reason...\n</think>\nHere is the answer.";
    char buf[256];
    int len = tool_parser_strip_think(text, strlen(text), buf, sizeof(buf));
    assert(len > 0);
    assert(strcmp(buf, "Here is the answer.") == 0);

    printf("test_strip_think: PASSED\n");
}

static void test_has_tool_calls(void) {
    assert(tool_parser_has_tool_calls("<tool_call>x</tool_call>", 24));
    assert(!tool_parser_has_tool_calls("no tools here", 13));

    printf("test_has_tool_calls: PASSED\n");
}

static void test_size_query(void) {
    ChatMessage msgs[] = {
        { .role = ROLE_USER, .content = "Hi" },
    };
    // NULL buf should return required size
    int needed = template_apply(msgs, 1, NULL, 0, true, true, NULL, 0);
    assert(needed > 0);

    char *buf = malloc((size_t)needed + 1);
    int actual = template_apply(msgs, 1, NULL, 0, true, true, buf, (size_t)needed + 1);
    assert(actual == needed);
    free(buf);

    printf("test_size_query: PASSED\n");
}

int main(void) {
    log_init();
    test_simple_chat();
    test_thinking_disabled();
    test_tool_injection();
    test_tool_response();
    test_tool_parser();
    test_multi_tool_calls();
    test_strip_think();
    test_has_tool_calls();
    test_size_query();
    printf("\nAll template tests passed.\n");
    return 0;
}
