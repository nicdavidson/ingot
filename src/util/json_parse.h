#ifndef INGOT_JSON_PARSE_H
#define INGOT_JSON_PARSE_H

#include <stdbool.h>
#include <stddef.h>

// Minimal recursive-descent JSON parser.
// Parses into a flat token array (no allocations beyond the token buffer).
// Designed for config files and API request bodies — not for streaming.

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT,
} JsonType;

typedef struct {
    JsonType type;
    const char *start;    // Pointer into original JSON string
    size_t     len;       // Length of the raw token
    int        children;  // Number of direct children (arrays/objects)
    int        next;      // Index of next sibling token (-1 if none)
} JsonToken;

typedef struct {
    const char  *json;
    size_t       json_len;
    JsonToken   *tokens;
    int          num_tokens;
    int          max_tokens;
} JsonDoc;

// Parse JSON string into token array.
// Returns true on success. tokens must be pre-allocated.
bool json_parse(JsonDoc *doc, const char *json, size_t len,
                JsonToken *tokens, int max_tokens);

// Find a key in an object token. Returns token index or -1.
int json_get(const JsonDoc *doc, int obj_idx, const char *key);

// Extract a string value (copies into buf, NUL-terminates). Returns false if not a string.
bool json_string(const JsonDoc *doc, int idx, char *buf, size_t buf_size);

// Extract a number value. Returns 0.0 if not a number.
double json_number(const JsonDoc *doc, int idx);

// Extract an integer value.
int json_int(const JsonDoc *doc, int idx);

// Extract a bool value.
bool json_bool(const JsonDoc *doc, int idx);

// Get array length.
int json_array_len(const JsonDoc *doc, int idx);

// Get array element by index. Returns token index or -1.
int json_array_get(const JsonDoc *doc, int arr_idx, int element);

#endif
