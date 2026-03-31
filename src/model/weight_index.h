#ifndef INGOT_WEIGHT_INDEX_H
#define INGOT_WEIGHT_INDEX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Weight entry — location of a tensor in the mmap'd files
typedef struct {
    char     name[128];
    size_t   offset;    // byte offset within the file
    size_t   size;      // byte size of the tensor data
    char     dtype[8];  // "U32", "BF16", "F32", etc.
    int      shape[4];
    int      ndim;
    // Expert-specific
    int      num_experts;
    size_t   expert_stride;
} WeightEntry;

typedef struct {
    WeightEntry *entries;
    int          count;
    int          capacity;
} WeightIndex;

// Load weight_index.json. Returns true on success.
bool weight_index_load(WeightIndex *idx, const char *path);

// Find a weight by name. Returns NULL if not found.
const WeightEntry *weight_index_find(const WeightIndex *idx, const char *name);

// Free the index.
void weight_index_free(WeightIndex *idx);

#endif
