#ifndef INGOT_JSON_WRITE_H
#define INGOT_JSON_WRITE_H

#include <stddef.h>
#include <stdbool.h>

// Streaming JSON builder — writes directly to a buffer.
// No allocations, no tree building. Just appends.

typedef struct {
    char   *buf;
    size_t  cap;
    size_t  len;
    int     depth;
    bool    needs_comma[32]; // per nesting level
} JsonWriter;

void jw_init(JsonWriter *w, char *buf, size_t cap);

void jw_object_start(JsonWriter *w);
void jw_object_end(JsonWriter *w);
void jw_array_start(JsonWriter *w);
void jw_array_end(JsonWriter *w);

void jw_key(JsonWriter *w, const char *key);

void jw_string(JsonWriter *w, const char *val);
void jw_string_len(JsonWriter *w, const char *val, size_t len);
void jw_int(JsonWriter *w, int val);
void jw_int64(JsonWriter *w, long long val);
void jw_double(JsonWriter *w, double val);
void jw_bool(JsonWriter *w, bool val);
void jw_null(JsonWriter *w);

// Write raw JSON (already formatted)
void jw_raw(JsonWriter *w, const char *json);

// Get current output length
size_t jw_length(const JsonWriter *w);

#endif
