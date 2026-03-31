#include "server/json_write.h"

#include <stdio.h>
#include <string.h>

static void jw_put(JsonWriter *w, const char *s, size_t n) {
    if (w->len + n < w->cap) {
        memcpy(w->buf + w->len, s, n);
    }
    w->len += n;
}

static void jw_putc(JsonWriter *w, char c) {
    if (w->len < w->cap) w->buf[w->len] = c;
    w->len++;
}

static void jw_comma(JsonWriter *w) {
    if (w->depth >= 0 && w->needs_comma[w->depth]) {
        jw_putc(w, ',');
    }
    if (w->depth >= 0) w->needs_comma[w->depth] = true;
}

void jw_init(JsonWriter *w, char *buf, size_t cap) {
    w->buf = buf;
    w->cap = cap;
    w->len = 0;
    w->depth = -1;
    memset(w->needs_comma, 0, sizeof(w->needs_comma));
}

void jw_object_start(JsonWriter *w) {
    jw_comma(w);
    jw_putc(w, '{');
    w->depth++;
    w->needs_comma[w->depth] = false;
}

void jw_object_end(JsonWriter *w) {
    jw_putc(w, '}');
    w->depth--;
}

void jw_array_start(JsonWriter *w) {
    jw_comma(w);
    jw_putc(w, '[');
    w->depth++;
    w->needs_comma[w->depth] = false;
}

void jw_array_end(JsonWriter *w) {
    jw_putc(w, ']');
    w->depth--;
}

void jw_key(JsonWriter *w, const char *key) {
    jw_comma(w);
    jw_putc(w, '"');
    jw_put(w, key, strlen(key));
    jw_putc(w, '"');
    jw_putc(w, ':');
    w->needs_comma[w->depth] = false; // value follows, not comma
}

void jw_string(JsonWriter *w, const char *val) {
    jw_string_len(w, val, strlen(val));
}

void jw_string_len(JsonWriter *w, const char *val, size_t len) {
    jw_comma(w);
    jw_putc(w, '"');
    for (size_t i = 0; i < len; i++) {
        switch (val[i]) {
            case '"':  jw_put(w, "\\\"", 2); break;
            case '\\': jw_put(w, "\\\\", 2); break;
            case '\n': jw_put(w, "\\n", 2);  break;
            case '\r': jw_put(w, "\\r", 2);  break;
            case '\t': jw_put(w, "\\t", 2);  break;
            default:
                if ((unsigned char)val[i] < 0x20) {
                    char esc[8];
                    int n = snprintf(esc, sizeof(esc), "\\u%04x", (unsigned char)val[i]);
                    jw_put(w, esc, (size_t)n);
                } else {
                    jw_putc(w, val[i]);
                }
                break;
        }
    }
    jw_putc(w, '"');
}

void jw_int(JsonWriter *w, int val) {
    jw_comma(w);
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%d", val);
    jw_put(w, buf, (size_t)n);
}

void jw_int64(JsonWriter *w, long long val) {
    jw_comma(w);
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%lld", val);
    jw_put(w, buf, (size_t)n);
}

void jw_double(JsonWriter *w, double val) {
    jw_comma(w);
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "%g", val);
    jw_put(w, buf, (size_t)n);
}

void jw_bool(JsonWriter *w, bool val) {
    jw_comma(w);
    if (val) jw_put(w, "true", 4);
    else     jw_put(w, "false", 5);
}

void jw_null(JsonWriter *w) {
    jw_comma(w);
    jw_put(w, "null", 4);
}

void jw_raw(JsonWriter *w, const char *json) {
    jw_comma(w);
    jw_put(w, json, strlen(json));
}

size_t jw_length(const JsonWriter *w) {
    return w->len;
}
