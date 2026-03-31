#ifndef INGOT_REQUEST_PARSE_H
#define INGOT_REQUEST_PARSE_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    char    method[8];      // GET, POST, etc.
    char    path[256];      // /v1/chat/completions
    char   *body;           // request body (malloc'd, caller frees)
    size_t  body_len;
    size_t  content_length;
    bool    keep_alive;
    char    content_type[128];
} HttpRequest;

// Parse an HTTP/1.1 request from raw data.
// Returns bytes consumed on success, 0 if incomplete, -1 on error.
// Body is allocated separately if Content-Length is present.
int http_parse_request(HttpRequest *req, const char *data, size_t len);

// Free request body.
void http_request_free(HttpRequest *req);

#endif
