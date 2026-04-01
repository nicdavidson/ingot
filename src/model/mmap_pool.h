#ifndef INGOT_MMAP_POOL_H
#define INGOT_MMAP_POOL_H

#include <stdbool.h>
#include <stddef.h>

typedef struct MmapPool MmapPool;

// Create a pool for managing mmap'd weight files.
MmapPool *mmap_pool_create(int max_files);

// Map a file into the pool. Returns pointer to mapped region.
// Uses MAP_SHARED for page cache benefits.
void *mmap_pool_add(MmapPool *pool, const char *path, size_t *out_size);

// Issue madvise(MADV_WILLNEED) for a region (expert prefetch hint).
void mmap_pool_prefetch(void *addr, size_t len);

// Get the file descriptor for a mapped file by index.
// Returns -1 if index is out of range.
int mmap_pool_get_fd(MmapPool *pool, int index);

// Get the number of files in the pool.
int mmap_pool_count(MmapPool *pool);

// Free pool and unmap all files.
void mmap_pool_free(MmapPool *pool);

#endif
