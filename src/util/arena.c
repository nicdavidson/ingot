#include "util/arena.h"
#include "util/log.h"

#include <stdlib.h>
#include <string.h>

Arena arena_create(size_t size) {
    uint8_t *buf = malloc(size);
    if (!buf) {
        LOG_ERROR("arena_create: failed to allocate %zu bytes", size);
        return (Arena){0};
    }
    return (Arena){ .buf = buf, .size = size, .used = 0 };
}

void *arena_alloc(Arena *a, size_t bytes) {
    // Align to 16 bytes
    size_t aligned = (bytes + 15) & ~(size_t)15;
    if (a->used + aligned > a->size) {
        LOG_ERROR("arena_alloc: out of memory (need %zu, have %zu free)",
                  aligned, a->size - a->used);
        return NULL;
    }
    void *ptr = a->buf + a->used;
    a->used += aligned;
    return ptr;
}

void *arena_alloc_zero(Arena *a, size_t bytes) {
    void *ptr = arena_alloc(a, bytes);
    if (ptr) memset(ptr, 0, bytes);
    return ptr;
}

void arena_reset(Arena *a) {
    a->used = 0;
}

void arena_destroy(Arena *a) {
    free(a->buf);
    *a = (Arena){0};
}
