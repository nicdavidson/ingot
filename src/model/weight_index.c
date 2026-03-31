#define _POSIX_C_SOURCE 200809L

#include "model/weight_index.h"
#include "util/json_parse.h"
#include "util/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[rd] = '\0';
    *out_len = rd;
    return buf;
}

bool weight_index_load(WeightIndex *idx, const char *path) {
    memset(idx, 0, sizeof(*idx));

    size_t len;
    char *json = read_file(path, &len);
    if (!json) {
        LOG_ERROR("weight_index: failed to read %s", path);
        return false;
    }

    int max_tokens = 65536;
    JsonToken *tokens = calloc((size_t)max_tokens, sizeof(JsonToken));
    if (!tokens) { free(json); return false; }

    JsonDoc doc;
    if (!json_parse(&doc, json, len, tokens, max_tokens)) {
        LOG_ERROR("weight_index: failed to parse %s", path);
        free(tokens);
        free(json);
        return false;
    }

    if (doc.tokens[0].type != JSON_OBJECT) {
        free(tokens);
        free(json);
        return false;
    }

    // Count entries
    int num = doc.tokens[0].children;
    idx->entries = calloc((size_t)num, sizeof(WeightEntry));
    idx->capacity = num;
    idx->count = 0;

    int key_idx = 1;
    for (int i = 0; i < num; i++) {
        if (key_idx < 0 || key_idx >= doc.num_tokens) break;

        WeightEntry *e = &idx->entries[idx->count];

        // Key = weight name
        json_string(&doc, key_idx, e->name, sizeof(e->name));

        int val_idx = key_idx + 1;
        if (val_idx >= doc.num_tokens) break;

        // Value = object with offset, size, dtype, shape, etc.
        int v;
        if ((v = json_get(&doc, val_idx, "offset")) >= 0)
            e->offset = (size_t)json_number(&doc, v);
        if ((v = json_get(&doc, val_idx, "size")) >= 0)
            e->size = (size_t)json_number(&doc, v);
        if ((v = json_get(&doc, val_idx, "dtype")) >= 0)
            json_string(&doc, v, e->dtype, sizeof(e->dtype));
        if ((v = json_get(&doc, val_idx, "num_experts")) >= 0)
            e->num_experts = json_int(&doc, v);
        if ((v = json_get(&doc, val_idx, "expert_stride")) >= 0)
            e->expert_stride = (size_t)json_number(&doc, v);

        int shape_idx = json_get(&doc, val_idx, "shape");
        if (shape_idx >= 0) {
            int shape_len = json_array_len(&doc, shape_idx);
            e->ndim = shape_len > 4 ? 4 : shape_len;
            for (int s = 0; s < e->ndim; s++) {
                int el = json_array_get(&doc, shape_idx, s);
                if (el >= 0) e->shape[s] = json_int(&doc, el);
            }
        }

        idx->count++;
        key_idx = doc.tokens[key_idx].next;
        if (key_idx < 0) break;
    }

    free(tokens);
    free(json);

    LOG_INFO("weight_index: loaded %d weight entries", idx->count);
    return true;
}

const WeightEntry *weight_index_find(const WeightIndex *idx, const char *name) {
    for (int i = 0; i < idx->count; i++) {
        if (strcmp(idx->entries[i].name, name) == 0)
            return &idx->entries[i];
    }
    return NULL;
}

void weight_index_free(WeightIndex *idx) {
    free(idx->entries);
    memset(idx, 0, sizeof(*idx));
}
