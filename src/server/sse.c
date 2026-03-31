#define _POSIX_C_SOURCE 200809L

#include "server/sse.h"
#include "server/json_write.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

// Suppress unused write return warnings — server code ignores partial writes
// on streaming connections (broken pipe is handled via SIGPIPE ignore).
static void send_bytes(int fd, const void *buf, size_t len) {
    ssize_t r = write(fd, buf, len);
    (void)r;
}

void sse_write_headers(int fd) {
    const char *headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    send_bytes(fd, headers, strlen(headers));
}

void sse_write_event(int fd, const char *data, size_t len) {
    send_bytes(fd, "data: ", 6);
    send_bytes(fd, data, len);
    send_bytes(fd, "\n\n", 2);
}

void sse_write_done(int fd) {
    send_bytes(fd, "data: [DONE]\n\n", 14);
}

int sse_format_chunk(char *buf, size_t buf_size,
                     const char *model_name,
                     const char *content,
                     const char *finish_reason,
                     const char *chunk_id,
                     int index) {
    JsonWriter w;
    jw_init(&w, buf, buf_size);

    jw_object_start(&w);

    jw_key(&w, "id");
    jw_string(&w, chunk_id);

    jw_key(&w, "object");
    jw_string(&w, "chat.completion.chunk");

    jw_key(&w, "created");
    jw_int64(&w, (long long)time(NULL));

    jw_key(&w, "model");
    jw_string(&w, model_name);

    jw_key(&w, "choices");
    jw_array_start(&w);
    jw_object_start(&w);

    jw_key(&w, "index");
    jw_int(&w, index);

    jw_key(&w, "delta");
    jw_object_start(&w);
    if (content) {
        jw_key(&w, "content");
        jw_string(&w, content);
    }
    jw_object_end(&w);

    if (finish_reason) {
        jw_key(&w, "finish_reason");
        jw_string(&w, finish_reason);
    } else {
        jw_key(&w, "finish_reason");
        jw_null(&w);
    }

    jw_object_end(&w);
    jw_array_end(&w);

    jw_object_end(&w);

    if (jw_length(&w) < buf_size) buf[jw_length(&w)] = '\0';
    return (int)jw_length(&w);
}
