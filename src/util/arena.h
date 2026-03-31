#ifndef INGOT_ARENA_H
#define INGOT_ARENA_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *buf;
    size_t   size;
    size_t   used;
} Arena;

Arena  arena_create(size_t size);
void  *arena_alloc(Arena *a, size_t bytes);
void  *arena_alloc_zero(Arena *a, size_t bytes);
void   arena_reset(Arena *a);
void   arena_destroy(Arena *a);

#endif
