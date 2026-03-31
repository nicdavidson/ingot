#include "util/json_parse.h"

#include <string.h>
#include <stdlib.h>

// --- Internal parser state ---

typedef struct {
    const char *s;
    size_t      len;
    size_t      pos;
    JsonToken  *tokens;
    int         ntok;
    int         max;
} Parser;

static void skip_ws(Parser *p) {
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            p->pos++;
        else
            break;
    }
}

static int alloc_token(Parser *p) {
    if (p->ntok >= p->max) return -1;
    int idx = p->ntok++;
    p->tokens[idx] = (JsonToken){ .next = -1 };
    return idx;
}

static bool parse_value(Parser *p);

static bool parse_string_raw(Parser *p) {
    if (p->pos >= p->len || p->s[p->pos] != '"') return false;
    p->pos++; // opening quote
    size_t start = p->pos;
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        if (c == '\\') {
            p->pos += 2; // skip escaped char
            continue;
        }
        if (c == '"') {
            int idx = alloc_token(p);
            if (idx < 0) return false;
            p->tokens[idx].type = JSON_STRING;
            p->tokens[idx].start = p->s + start;
            p->tokens[idx].len = p->pos - start;
            p->pos++; // closing quote
            return true;
        }
        p->pos++;
    }
    return false; // unterminated string
}

static bool parse_number(Parser *p) {
    size_t start = p->pos;
    if (p->s[p->pos] == '-') p->pos++;
    while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') p->pos++;
    if (p->pos < p->len && p->s[p->pos] == '.') {
        p->pos++;
        while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') p->pos++;
    }
    if (p->pos < p->len && (p->s[p->pos] == 'e' || p->s[p->pos] == 'E')) {
        p->pos++;
        if (p->pos < p->len && (p->s[p->pos] == '+' || p->s[p->pos] == '-')) p->pos++;
        while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') p->pos++;
    }
    if (p->pos == start) return false;
    int idx = alloc_token(p);
    if (idx < 0) return false;
    p->tokens[idx].type = JSON_NUMBER;
    p->tokens[idx].start = p->s + start;
    p->tokens[idx].len = p->pos - start;
    return true;
}

static bool parse_literal(Parser *p, const char *lit, size_t llen, JsonType type) {
    if (p->pos + llen > p->len) return false;
    if (memcmp(p->s + p->pos, lit, llen) != 0) return false;
    int idx = alloc_token(p);
    if (idx < 0) return false;
    p->tokens[idx].type = type;
    p->tokens[idx].start = p->s + p->pos;
    p->tokens[idx].len = llen;
    p->pos += llen;
    return true;
}

static bool parse_array(Parser *p) {
    p->pos++; // skip '['
    int arr_idx = alloc_token(p);
    if (arr_idx < 0) return false;
    p->tokens[arr_idx].type = JSON_ARRAY;
    p->tokens[arr_idx].start = p->s + p->pos - 1;
    p->tokens[arr_idx].children = 0;

    skip_ws(p);
    if (p->pos < p->len && p->s[p->pos] == ']') {
        p->pos++;
        p->tokens[arr_idx].len = (size_t)(p->s + p->pos - p->tokens[arr_idx].start);
        return true;
    }

    int prev_child = -1;
    for (;;) {
        skip_ws(p);
        int child_idx = p->ntok;
        if (!parse_value(p)) return false;
        p->tokens[arr_idx].children++;
        if (prev_child >= 0) p->tokens[prev_child].next = child_idx;
        prev_child = child_idx;

        skip_ws(p);
        if (p->pos >= p->len) return false;
        if (p->s[p->pos] == ']') {
            p->pos++;
            break;
        }
        if (p->s[p->pos] != ',') return false;
        p->pos++;
    }
    p->tokens[arr_idx].len = (size_t)(p->s + p->pos - p->tokens[arr_idx].start);
    return true;
}

static bool parse_object(Parser *p) {
    p->pos++; // skip '{'
    int obj_idx = alloc_token(p);
    if (obj_idx < 0) return false;
    p->tokens[obj_idx].type = JSON_OBJECT;
    p->tokens[obj_idx].start = p->s + p->pos - 1;
    p->tokens[obj_idx].children = 0;

    skip_ws(p);
    if (p->pos < p->len && p->s[p->pos] == '}') {
        p->pos++;
        p->tokens[obj_idx].len = (size_t)(p->s + p->pos - p->tokens[obj_idx].start);
        return true;
    }

    int prev_key = -1;
    for (;;) {
        skip_ws(p);
        // Key
        int key_idx = p->ntok;
        if (!parse_string_raw(p)) return false;

        skip_ws(p);
        if (p->pos >= p->len || p->s[p->pos] != ':') return false;
        p->pos++;

        // Value
        skip_ws(p);
        if (!parse_value(p)) return false;

        p->tokens[obj_idx].children++;
        if (prev_key >= 0) p->tokens[prev_key].next = key_idx;
        prev_key = key_idx;

        skip_ws(p);
        if (p->pos >= p->len) return false;
        if (p->s[p->pos] == '}') {
            p->pos++;
            break;
        }
        if (p->s[p->pos] != ',') return false;
        p->pos++;
    }
    p->tokens[obj_idx].len = (size_t)(p->s + p->pos - p->tokens[obj_idx].start);
    return true;
}

static bool parse_value(Parser *p) {
    skip_ws(p);
    if (p->pos >= p->len) return false;

    char c = p->s[p->pos];
    switch (c) {
        case '"': return parse_string_raw(p);
        case '{': return parse_object(p);
        case '[': return parse_array(p);
        case 't': return parse_literal(p, "true", 4, JSON_BOOL);
        case 'f': return parse_literal(p, "false", 5, JSON_BOOL);
        case 'n': return parse_literal(p, "null", 4, JSON_NULL);
        default:
            if (c == '-' || (c >= '0' && c <= '9'))
                return parse_number(p);
            return false;
    }
}

// --- Public API ---

bool json_parse(JsonDoc *doc, const char *json, size_t len,
                JsonToken *tokens, int max_tokens) {
    Parser p = {
        .s      = json,
        .len    = len,
        .pos    = 0,
        .tokens = tokens,
        .ntok   = 0,
        .max    = max_tokens,
    };

    if (!parse_value(&p)) return false;

    doc->json       = json;
    doc->json_len   = len;
    doc->tokens     = tokens;
    doc->num_tokens = p.ntok;
    doc->max_tokens = max_tokens;
    return true;
}

int json_get(const JsonDoc *doc, int obj_idx, const char *key) {
    if (obj_idx < 0 || obj_idx >= doc->num_tokens) return -1;
    if (doc->tokens[obj_idx].type != JSON_OBJECT) return -1;

    size_t klen = strlen(key);
    int idx = obj_idx + 1; // first key is right after the object token
    for (int i = 0; i < doc->tokens[obj_idx].children; i++) {
        if (idx < 0 || idx >= doc->num_tokens) return -1;
        JsonToken *kt = &doc->tokens[idx];
        if (kt->type == JSON_STRING && kt->len == klen &&
            memcmp(kt->start, key, klen) == 0) {
            return idx + 1; // value is right after the key
        }
        // Skip to next key: key token's next points to next key
        idx = kt->next;
    }
    return -1;
}

bool json_string(const JsonDoc *doc, int idx, char *buf, size_t buf_size) {
    if (idx < 0 || idx >= doc->num_tokens) return false;
    if (doc->tokens[idx].type != JSON_STRING) return false;

    const char *src = doc->tokens[idx].start;
    size_t slen = doc->tokens[idx].len;
    size_t out = 0;

    for (size_t i = 0; i < slen && out + 1 < buf_size; i++) {
        if (src[i] == '\\' && i + 1 < slen) {
            i++;
            switch (src[i]) {
                case '"':  buf[out++] = '"'; break;
                case '\\': buf[out++] = '\\'; break;
                case '/':  buf[out++] = '/'; break;
                case 'n':  buf[out++] = '\n'; break;
                case 'r':  buf[out++] = '\r'; break;
                case 't':  buf[out++] = '\t'; break;
                case 'b':  buf[out++] = '\b'; break;
                case 'f':  buf[out++] = '\f'; break;
                default:   buf[out++] = src[i]; break;
            }
        } else {
            buf[out++] = src[i];
        }
    }
    buf[out] = '\0';
    return true;
}

double json_number(const JsonDoc *doc, int idx) {
    if (idx < 0 || idx >= doc->num_tokens) return 0.0;
    if (doc->tokens[idx].type != JSON_NUMBER) return 0.0;
    return strtod(doc->tokens[idx].start, NULL);
}

int json_int(const JsonDoc *doc, int idx) {
    return (int)json_number(doc, idx);
}

bool json_bool(const JsonDoc *doc, int idx) {
    if (idx < 0 || idx >= doc->num_tokens) return false;
    if (doc->tokens[idx].type != JSON_BOOL) return false;
    return doc->tokens[idx].start[0] == 't';
}

int json_array_len(const JsonDoc *doc, int idx) {
    if (idx < 0 || idx >= doc->num_tokens) return 0;
    if (doc->tokens[idx].type != JSON_ARRAY) return 0;
    return doc->tokens[idx].children;
}

int json_array_get(const JsonDoc *doc, int arr_idx, int element) {
    if (arr_idx < 0 || arr_idx >= doc->num_tokens) return -1;
    if (doc->tokens[arr_idx].type != JSON_ARRAY) return -1;
    if (element >= doc->tokens[arr_idx].children) return -1;

    int idx = arr_idx + 1; // first element is right after array token
    for (int i = 0; i < element; i++) {
        if (idx < 0 || idx >= doc->num_tokens) return -1;
        idx = doc->tokens[idx].next;
    }
    return idx;
}
