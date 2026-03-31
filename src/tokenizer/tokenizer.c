#define _POSIX_C_SOURCE 200809L

#include "tokenizer/tokenizer.h"
#include "tokenizer/byte_decode.h"
#include "util/json_parse.h"
#include "util/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Hash map for string→int lookups ---

typedef struct {
    char *key;
    int   value;
} HashEntry;

typedef struct {
    HashEntry *entries;
    int        capacity;
    int        count;
} HashMap;

static uint32_t hash_str(const char *s) {
    uint32_t h = 5381;
    while (*s) h = ((h << 5) + h) + (uint8_t)*s++;
    return h;
}

static void hashmap_init(HashMap *m, int capacity) {
    m->capacity = capacity;
    m->count = 0;
    m->entries = calloc((size_t)capacity, sizeof(HashEntry));
}

static void hashmap_free(HashMap *m) {
    for (int i = 0; i < m->capacity; i++)
        free(m->entries[i].key);
    free(m->entries);
}

static void hashmap_put(HashMap *m, const char *key, int value) {
    uint32_t idx = hash_str(key) % (uint32_t)m->capacity;
    for (int i = 0; i < m->capacity; i++) {
        uint32_t slot = (idx + (uint32_t)i) % (uint32_t)m->capacity;
        if (!m->entries[slot].key) {
            m->entries[slot].key = strdup(key);
            m->entries[slot].value = value;
            m->count++;
            return;
        }
        if (strcmp(m->entries[slot].key, key) == 0) {
            m->entries[slot].value = value;
            return;
        }
    }
}

static int hashmap_get(const HashMap *m, const char *key, int default_val) {
    uint32_t idx = hash_str(key) % (uint32_t)m->capacity;
    for (int i = 0; i < m->capacity; i++) {
        uint32_t slot = (idx + (uint32_t)i) % (uint32_t)m->capacity;
        if (!m->entries[slot].key) return default_val;
        if (strcmp(m->entries[slot].key, key) == 0) return m->entries[slot].value;
    }
    return default_val;
}

// --- Merge pair hash for O(1) merge lookup ---

typedef struct {
    int first;
    int second;
    int rank;    // merge priority (lower = merge first)
} MergePair;

typedef struct {
    MergePair *entries;
    int        capacity;
    int        count;
} MergeMap;

static uint32_t hash_pair(int a, int b) {
    return (uint32_t)a * 2654435761U + (uint32_t)b;
}

static void mergemap_init(MergeMap *m, int capacity) {
    m->capacity = capacity;
    m->count = 0;
    m->entries = calloc((size_t)capacity, sizeof(MergePair));
    for (int i = 0; i < capacity; i++) m->entries[i].rank = -1;
}

static void mergemap_free(MergeMap *m) {
    free(m->entries);
}

static void mergemap_put(MergeMap *m, int first, int second, int rank) {
    uint32_t idx = hash_pair(first, second) % (uint32_t)m->capacity;
    for (int i = 0; i < m->capacity; i++) {
        uint32_t slot = (idx + (uint32_t)i) % (uint32_t)m->capacity;
        if (m->entries[slot].rank < 0) {
            m->entries[slot] = (MergePair){ first, second, rank };
            m->count++;
            return;
        }
        if (m->entries[slot].first == first && m->entries[slot].second == second) {
            m->entries[slot].rank = rank;
            return;
        }
    }
}

static int mergemap_get(const MergeMap *m, int first, int second) {
    uint32_t idx = hash_pair(first, second) % (uint32_t)m->capacity;
    for (int i = 0; i < m->capacity; i++) {
        uint32_t slot = (idx + (uint32_t)i) % (uint32_t)m->capacity;
        if (m->entries[slot].rank < 0) return -1;
        if (m->entries[slot].first == first && m->entries[slot].second == second)
            return m->entries[slot].rank;
    }
    return -1;
}

// --- Tokenizer struct ---

struct Tokenizer {
    char   **id_to_token;  // array indexed by token ID
    int      vocab_size;
    HashMap  token_to_id;  // string → ID
    MergeMap merges;       // (first_id, second_id) → rank

    // Special token IDs
    int eos_id;
    int im_start_id;
    int im_end_id;

    // Decode buffer
    char decode_buf[1024];
};

// --- File loading helpers ---

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

// Load vocab.json: simple {"token": id, ...} format
static bool load_vocab(Tokenizer *tok, const char *path) {
    size_t len;
    char *json = read_file(path, &len);
    if (!json) {
        LOG_ERROR("tokenizer: failed to read %s", path);
        return false;
    }

    // We need a LOT of tokens for 248K entries — each key-value pair
    // is 2 tokens plus the object token itself. Use heap allocation.
    int max_tokens = 600000;
    JsonToken *tokens = calloc((size_t)max_tokens, sizeof(JsonToken));
    if (!tokens) {
        LOG_ERROR("tokenizer: failed to allocate JSON token buffer");
        free(json);
        return false;
    }

    JsonDoc doc;
    if (!json_parse(&doc, json, len, tokens, max_tokens)) {
        LOG_ERROR("tokenizer: failed to parse %s", path);
        free(tokens);
        free(json);
        return false;
    }

    if (doc.tokens[0].type != JSON_OBJECT) {
        LOG_ERROR("tokenizer: vocab.json root is not an object");
        free(tokens);
        free(json);
        return false;
    }

    // Find max token ID to size our array
    int max_id = 0;
    int key_idx = 1; // first key after object token
    for (int i = 0; i < doc.tokens[0].children; i++) {
        int val_idx = key_idx + 1;
        int id = json_int(&doc, val_idx);
        if (id > max_id) max_id = id;
        key_idx = doc.tokens[key_idx].next;
        if (key_idx < 0) break;
    }

    tok->vocab_size = max_id + 1;
    tok->id_to_token = calloc((size_t)tok->vocab_size, sizeof(char *));
    hashmap_init(&tok->token_to_id, tok->vocab_size * 2); // 50% load factor

    // Second pass: populate
    key_idx = 1;
    for (int i = 0; i < doc.tokens[0].children; i++) {
        int val_idx = key_idx + 1;

        char token_str[1024];
        if (!json_string(&doc, key_idx, token_str, sizeof(token_str))) {
            key_idx = doc.tokens[key_idx].next;
            if (key_idx < 0) break;
            continue;
        }

        int id = json_int(&doc, val_idx);
        if (id >= 0 && id < tok->vocab_size) {
            tok->id_to_token[id] = strdup(token_str);
            hashmap_put(&tok->token_to_id, token_str, id);
        }

        key_idx = doc.tokens[key_idx].next;
        if (key_idx < 0) break;
    }

    free(tokens);
    free(json);

    LOG_INFO("tokenizer: loaded vocab with %d entries (max_id=%d)",
             tok->token_to_id.count, max_id);
    return true;
}

// Load merges.txt: one merge per line, "first second" format
static bool load_merges(Tokenizer *tok, const char *path) {
    size_t len;
    char *data = read_file(path, &len);
    if (!data) {
        LOG_ERROR("tokenizer: failed to read %s", path);
        return false;
    }

    // Count lines for capacity
    int num_lines = 0;
    for (size_t i = 0; i < len; i++)
        if (data[i] == '\n') num_lines++;

    mergemap_init(&tok->merges, num_lines * 2);

    char *line = data;
    int rank = 0;
    while (line < data + len) {
        char *eol = strchr(line, '\n');
        if (!eol) eol = data + len;
        size_t line_len = (size_t)(eol - line);

        // Skip empty lines and BPE header
        if (line_len > 0 && line[0] != '#') {
            // Find space separator
            char *space = memchr(line, ' ', line_len);
            if (space) {
                size_t first_len = (size_t)(space - line);
                size_t second_len = line_len - first_len - 1;

                char first[512], second[512];
                if (first_len < sizeof(first) && second_len < sizeof(second)) {
                    memcpy(first, line, first_len);
                    first[first_len] = '\0';
                    memcpy(second, space + 1, second_len);
                    second[second_len] = '\0';

                    // Strip trailing \r
                    if (second_len > 0 && second[second_len - 1] == '\r')
                        second[--second_len] = '\0';

                    int first_id = hashmap_get(&tok->token_to_id, first, -1);
                    int second_id = hashmap_get(&tok->token_to_id, second, -1);

                    if (first_id >= 0 && second_id >= 0) {
                        // The merged token is first+second concatenated
                        char merged[1024];
                        snprintf(merged, sizeof(merged), "%s%s", first, second);
                        int merged_id = hashmap_get(&tok->token_to_id, merged, -1);
                        (void)merged_id; // We just need the pair → rank mapping

                        mergemap_put(&tok->merges, first_id, second_id, rank);
                        rank++;
                    }
                }
            }
        }

        line = eol + 1;
    }

    free(data);
    LOG_INFO("tokenizer: loaded %d merges", rank);
    return true;
}

// Load added_tokens from tokenizer_config.json
// These are special tokens (eos, im_start, etc.) not in vocab.json
static bool load_added_tokens(Tokenizer *tok, const char *path) {
    size_t len;
    char *json = read_file(path, &len);
    if (!json) return false; // optional file

    int max_tokens = 32768;
    JsonToken *tokens = calloc((size_t)max_tokens, sizeof(JsonToken));
    if (!tokens) { free(json); return false; }
    JsonDoc doc;
    if (!json_parse(&doc, json, len, tokens, max_tokens)) {
        LOG_WARN("tokenizer: failed to parse %s (may be too large)", path);
        free(tokens);
        free(json);
        return false;
    }

    // Try "added_tokens_decoder" (tokenizer_config.json format),
    // otherwise treat root object as the token map (added_tokens.json format)
    int atd_idx = json_get(&doc, 0, "added_tokens_decoder");
    if (atd_idx < 0) {
        // Root object is the map
        if (doc.tokens[0].type == JSON_OBJECT) {
            atd_idx = 0;
        } else {
            free(tokens);
            free(json);
            return false;
        }
    }

    int added = 0;
    int key_idx = atd_idx + 1;
    for (int i = 0; i < doc.tokens[atd_idx].children; i++) {
        if (key_idx < 0 || key_idx >= doc.num_tokens) break;

        // Key is the token ID as a string
        char id_str[32];
        json_string(&doc, key_idx, id_str, sizeof(id_str));
        int id = atoi(id_str);

        // Value is an object with "content" field
        int val_idx = key_idx + 1;
        int content_idx = json_get(&doc, val_idx, "content");
        if (content_idx >= 0) {
            char content[256];
            json_string(&doc, content_idx, content, sizeof(content));

            // Add to vocab if not already present
            if (id >= 0 && hashmap_get(&tok->token_to_id, content, -1) < 0) {
                // Extend id_to_token array if needed
                if (id >= tok->vocab_size) {
                    int new_size = id + 1;
                    char **new_arr = realloc(tok->id_to_token,
                                            (size_t)new_size * sizeof(char *));
                    if (new_arr) {
                        memset(new_arr + tok->vocab_size, 0,
                               (size_t)(new_size - tok->vocab_size) * sizeof(char *));
                        tok->id_to_token = new_arr;
                        tok->vocab_size = new_size;
                    }
                }
                if (id < tok->vocab_size) {
                    free(tok->id_to_token[id]);
                    tok->id_to_token[id] = strdup(content);
                    hashmap_put(&tok->token_to_id, content, id);
                    added++;
                }
            }
        }

        key_idx = doc.tokens[key_idx].next;
        if (key_idx < 0) break;
    }

    free(tokens);
    free(json);

    if (added > 0)
        LOG_INFO("tokenizer: loaded %d added tokens", added);
    return true;
}

// --- Public API ---

Tokenizer *tokenizer_load(const char *model_dir) {
    byte_decode_init();

    Tokenizer *tok = calloc(1, sizeof(Tokenizer));
    if (!tok) return NULL;

    char path[1024];

    // Load vocab.json
    snprintf(path, sizeof(path), "%s/vocab.json", model_dir);
    if (!load_vocab(tok, path)) {
        tokenizer_free(tok);
        return NULL;
    }

    // Load merges.txt
    snprintf(path, sizeof(path), "%s/merges.txt", model_dir);
    if (!load_merges(tok, path)) {
        tokenizer_free(tok);
        return NULL;
    }

    // Load added tokens — try added_tokens.json first (extracted),
    // then tokenizer_config.json (HuggingFace format)
    snprintf(path, sizeof(path), "%s/added_tokens.json", model_dir);
    if (!load_added_tokens(tok, path)) {
        snprintf(path, sizeof(path), "%s/tokenizer_config.json", model_dir);
        load_added_tokens(tok, path);
    }

    // Resolve special token IDs
    tok->eos_id = hashmap_get(&tok->token_to_id, "<|endoftext|>", -1);
    tok->im_start_id = hashmap_get(&tok->token_to_id, "<|im_start|>", -1);
    tok->im_end_id = hashmap_get(&tok->token_to_id, "<|im_end|>", -1);

    LOG_INFO("tokenizer: eos=%d, im_start=%d, im_end=%d",
             tok->eos_id, tok->im_start_id, tok->im_end_id);

    return tok;
}

void tokenizer_free(Tokenizer *tok) {
    if (!tok) return;
    if (tok->id_to_token) {
        for (int i = 0; i < tok->vocab_size; i++)
            free(tok->id_to_token[i]);
        free(tok->id_to_token);
    }
    hashmap_free(&tok->token_to_id);
    mergemap_free(&tok->merges);
    free(tok);
}

// BPE encoding: convert text to byte-encoded tokens, then apply merges
int tokenizer_encode(const Tokenizer *tok, const char *text, size_t text_len,
                     int32_t *out, int max_tokens) {
    if (!text || text_len == 0) return 0;

    // Step 1: Convert each byte to its GPT-2 unicode character,
    //         then look up the single-char token ID
    int *ids = malloc(text_len * sizeof(int));
    if (!ids) return 0;
    int n = 0;

    for (size_t i = 0; i < text_len; i++) {
        // Get the GPT-2 byte-encoded character for this byte
        char encoded[8];
        uint8_t byte = (uint8_t)text[i];
        int elen = byte_encode_string(&byte, 1, encoded, sizeof(encoded));
        encoded[elen] = '\0';

        int id = hashmap_get(&tok->token_to_id, encoded, -1);
        if (id >= 0) {
            ids[n++] = id;
        }
    }

    // Step 2: Iteratively apply the highest-priority merge
    while (n > 1) {
        // Find the merge pair with lowest rank (= highest priority)
        int best_rank = __INT_MAX__;
        int best_pos = -1;

        for (int i = 0; i < n - 1; i++) {
            int rank = mergemap_get(&tok->merges, ids[i], ids[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_pos = i;
            }
        }

        if (best_pos < 0) break; // no more merges applicable

        // Get the merged token string
        const char *first = tok->id_to_token[ids[best_pos]];
        const char *second = tok->id_to_token[ids[best_pos + 1]];
        char merged[1024];
        snprintf(merged, sizeof(merged), "%s%s", first, second);
        int merged_id = hashmap_get(&tok->token_to_id, merged, -1);

        if (merged_id < 0) break; // shouldn't happen

        // Replace pair with merged token
        ids[best_pos] = merged_id;
        memmove(&ids[best_pos + 1], &ids[best_pos + 2],
                (size_t)(n - best_pos - 2) * sizeof(int));
        n--;
    }

    // Step 3: Copy to output
    int count = (n > max_tokens && out) ? max_tokens : n;
    if (out) {
        for (int i = 0; i < count; i++)
            out[i] = (int32_t)ids[i];
    }

    free(ids);
    return count;
}

const char *tokenizer_decode(const Tokenizer *tok, int32_t token_id) {
    Tokenizer *mtok = (Tokenizer *)tok; // cast away const for decode_buf

    if (token_id < 0 || token_id >= tok->vocab_size || !tok->id_to_token[token_id]) {
        mtok->decode_buf[0] = '\0';
        return mtok->decode_buf;
    }

    const char *token_str = tok->id_to_token[token_id];
    size_t token_len = strlen(token_str);

    // Check if it's a special token (starts with < and ends with >)
    if (token_len >= 2 && token_str[0] == '<' &&
        token_str[token_len - 1] == '>') {
        // Return special tokens as-is
        snprintf(mtok->decode_buf, sizeof(mtok->decode_buf), "%s", token_str);
        return mtok->decode_buf;
    }

    // Byte-decode the GPT-2 encoded token to raw UTF-8
    uint8_t raw[512];
    int raw_len = byte_decode_token(token_str, token_len, raw, sizeof(raw));
    if (raw_len > 0 && (size_t)raw_len < sizeof(mtok->decode_buf)) {
        memcpy(mtok->decode_buf, raw, (size_t)raw_len);
        mtok->decode_buf[raw_len] = '\0';
    } else {
        mtok->decode_buf[0] = '\0';
    }

    return mtok->decode_buf;
}

int tokenizer_decode_batch(const Tokenizer *tok, const int32_t *tokens,
                           int num_tokens, char *buf, size_t buf_size) {
    int written = 0;
    for (int i = 0; i < num_tokens; i++) {
        const char *s = tokenizer_decode(tok, tokens[i]);
        size_t slen = strlen(s);
        if ((size_t)written + slen + 1 > buf_size) break;
        memcpy(buf + written, s, slen);
        written += (int)slen;
    }
    if ((size_t)written < buf_size) buf[written] = '\0';
    return written;
}

int tokenizer_vocab_size(const Tokenizer *tok) {
    return tok->vocab_size;
}

int tokenizer_eos_id(const Tokenizer *tok) { return tok->eos_id; }
int tokenizer_im_start_id(const Tokenizer *tok) { return tok->im_start_id; }
int tokenizer_im_end_id(const Tokenizer *tok) { return tok->im_end_id; }
