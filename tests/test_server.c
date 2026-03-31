#define _POSIX_C_SOURCE 200809L

#include "server/request_parse.h"
#include "server/json_write.h"
#include "server/sse.h"
#include "util/log.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_parse_get(void) {
    const char *raw = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    HttpRequest req;
    int consumed = http_parse_request(&req, raw, strlen(raw));
    assert(consumed > 0);
    assert(strcmp(req.method, "GET") == 0);
    assert(strcmp(req.path, "/health") == 0);
    assert(req.body == NULL);
    http_request_free(&req);
    printf("test_parse_get: PASSED\n");
}

static void test_parse_post(void) {
    const char *raw =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 29\r\n"
        "\r\n"
        "{\"messages\":[],\"stream\":true}";
    HttpRequest req;
    int consumed = http_parse_request(&req, raw, strlen(raw));
    assert(consumed > 0);
    assert(strcmp(req.method, "POST") == 0);
    assert(strcmp(req.path, "/v1/chat/completions") == 0);
    assert(req.body != NULL);
    assert(req.body_len == 29);
    assert(strstr(req.body, "\"stream\":true") != NULL);
    http_request_free(&req);
    printf("test_parse_post: PASSED\n");
}

static void test_parse_incomplete(void) {
    const char *raw = "GET /health HTTP/1.1\r\n";
    HttpRequest req;
    int consumed = http_parse_request(&req, raw, strlen(raw));
    assert(consumed == 0); // incomplete, no \r\n\r\n yet
    printf("test_parse_incomplete: PASSED\n");
}

static void test_json_writer(void) {
    char buf[512];
    JsonWriter w;
    jw_init(&w, buf, sizeof(buf));

    jw_object_start(&w);
    jw_key(&w, "model"); jw_string(&w, "test");
    jw_key(&w, "count"); jw_int(&w, 42);
    jw_key(&w, "active"); jw_bool(&w, true);
    jw_key(&w, "items");
    jw_array_start(&w);
    jw_string(&w, "a");
    jw_string(&w, "b");
    jw_array_end(&w);
    jw_object_end(&w);

    buf[jw_length(&w)] = '\0';

    assert(strstr(buf, "\"model\":\"test\""));
    assert(strstr(buf, "\"count\":42"));
    assert(strstr(buf, "\"active\":true"));
    assert(strstr(buf, "\"items\":[\"a\",\"b\"]"));

    printf("test_json_writer: PASSED\n");
}

static void test_json_escape(void) {
    char buf[256];
    JsonWriter w;
    jw_init(&w, buf, sizeof(buf));

    jw_string(&w, "hello\nworld\t\"quoted\"");
    buf[jw_length(&w)] = '\0';

    assert(strstr(buf, "\\n"));
    assert(strstr(buf, "\\t"));
    assert(strstr(buf, "\\\"quoted\\\""));

    printf("test_json_escape: PASSED\n");
}

static void test_sse_chunk(void) {
    char buf[1024];
    int len = sse_format_chunk(buf, sizeof(buf),
                               "test-model", "Hello", NULL,
                               "chatcmpl-123", 0);
    assert(len > 0);
    assert(strstr(buf, "\"model\":\"test-model\""));
    assert(strstr(buf, "\"content\":\"Hello\""));
    assert(strstr(buf, "\"finish_reason\":null"));
    assert(strstr(buf, "chat.completion.chunk"));

    printf("test_sse_chunk: PASSED\n");
}

static void test_sse_chunk_finish(void) {
    char buf[1024];
    int len = sse_format_chunk(buf, sizeof(buf),
                               "test-model", NULL, "stop",
                               "chatcmpl-123", 0);
    assert(len > 0);
    assert(strstr(buf, "\"finish_reason\":\"stop\""));

    printf("test_sse_chunk_finish: PASSED\n");
}

int main(void) {
    log_init();
    test_parse_get();
    test_parse_post();
    test_parse_incomplete();
    test_json_writer();
    test_json_escape();
    test_sse_chunk();
    test_sse_chunk_finish();
    printf("\nAll server tests passed.\n");
    return 0;
}
