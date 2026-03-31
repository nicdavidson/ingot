#include "server/request_parse.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Find \r\n\r\n header terminator
static const char *find_header_end(const char *data, size_t len) {
    for (size_t i = 0; i + 3 < len; i++) {
        if (data[i] == '\r' && data[i+1] == '\n' &&
            data[i+2] == '\r' && data[i+3] == '\n') {
            return data + i + 4;
        }
    }
    return NULL;
}

// Extract a header value (case-insensitive key match)
static bool get_header(const char *headers, size_t hlen,
                       const char *key, char *out, size_t out_size) {
    size_t klen = strlen(key);
    const char *pos = headers;
    const char *end = headers + hlen;

    while (pos < end) {
        const char *eol = memchr(pos, '\n', (size_t)(end - pos));
        if (!eol) break;

        // Case-insensitive key match
        if ((size_t)(eol - pos) > klen + 1) {
            bool match = true;
            for (size_t i = 0; i < klen; i++) {
                if (tolower((unsigned char)pos[i]) != tolower((unsigned char)key[i])) {
                    match = false;
                    break;
                }
            }
            if (match && pos[klen] == ':') {
                const char *val = pos + klen + 1;
                while (val < eol && *val == ' ') val++;
                size_t vlen = (size_t)(eol - val);
                if (vlen > 0 && val[vlen-1] == '\r') vlen--;
                if (vlen >= out_size) vlen = out_size - 1;
                memcpy(out, val, vlen);
                out[vlen] = '\0';
                return true;
            }
        }

        pos = eol + 1;
    }
    return false;
}

int http_parse_request(HttpRequest *req, const char *data, size_t len) {
    memset(req, 0, sizeof(*req));

    const char *body_start = find_header_end(data, len);
    if (!body_start) return 0; // incomplete

    size_t header_len = (size_t)(body_start - data);

    // Parse request line: METHOD PATH HTTP/1.1\r\n
    const char *eol = memchr(data, '\r', len);
    if (!eol) return -1;

    // Method
    const char *space1 = memchr(data, ' ', (size_t)(eol - data));
    if (!space1) return -1;
    size_t mlen = (size_t)(space1 - data);
    if (mlen >= sizeof(req->method)) return -1;
    memcpy(req->method, data, mlen);
    req->method[mlen] = '\0';

    // Path
    const char *path_start = space1 + 1;
    const char *space2 = memchr(path_start, ' ', (size_t)(eol - path_start));
    if (!space2) return -1;
    size_t plen = (size_t)(space2 - path_start);
    if (plen >= sizeof(req->path)) plen = sizeof(req->path) - 1;
    memcpy(req->path, path_start, plen);
    req->path[plen] = '\0';

    // Headers
    char cl_str[32] = {0};
    if (get_header(data, header_len, "Content-Length", cl_str, sizeof(cl_str))) {
        req->content_length = (size_t)atol(cl_str);
    }

    get_header(data, header_len, "Content-Type",
               req->content_type, sizeof(req->content_type));

    char conn[32] = {0};
    if (get_header(data, header_len, "Connection", conn, sizeof(conn))) {
        req->keep_alive = (strstr(conn, "close") == NULL);
    } else {
        req->keep_alive = true; // HTTP/1.1 default
    }

    // Body
    size_t available_body = len - header_len;
    if (req->content_length > 0) {
        if (available_body < req->content_length) return 0; // incomplete

        req->body = malloc(req->content_length + 1);
        memcpy(req->body, body_start, req->content_length);
        req->body[req->content_length] = '\0';
        req->body_len = req->content_length;
        return (int)(header_len + req->content_length);
    }

    return (int)header_len;
}

void http_request_free(HttpRequest *req) {
    free(req->body);
    req->body = NULL;
}
