#define _GNU_SOURCE

#include "model/mmap_pool.h"
#include "util/log.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    void   *addr;
    size_t  size;
    int     fd;
    char    path[512];
} MappedFile;

struct MmapPool {
    MappedFile *files;
    int         count;
    int         capacity;
};

MmapPool *mmap_pool_create(int max_files) {
    MmapPool *pool = calloc(1, sizeof(MmapPool));
    if (!pool) return NULL;
    pool->files = calloc((size_t)max_files, sizeof(MappedFile));
    pool->capacity = max_files;
    return pool;
}

void *mmap_pool_add(MmapPool *pool, const char *path, size_t *out_size) {
    if (pool->count >= pool->capacity) {
        LOG_ERROR("mmap_pool: at capacity (%d files)", pool->capacity);
        return NULL;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("mmap_pool: failed to open %s", path);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        LOG_ERROR("mmap_pool: failed to stat %s", path);
        close(fd);
        return NULL;
    }

    size_t size = (size_t)st.st_size;

    // MAP_SHARED: pages come from page cache, not charged to Jetsam footprint.
    // This is critical for expert weights — lets the OS manage eviction.
    void *addr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        LOG_ERROR("mmap_pool: mmap failed for %s (%zu bytes)", path, size);
        close(fd);
        return NULL;
    }

    // Default readahead behavior — per-expert MADV_WILLNEED provides
    // targeted prefetch hints when experts are selected by the gate

    MappedFile *mf = &pool->files[pool->count++];
    mf->addr = addr;
    mf->size = size;
    mf->fd = fd;
    snprintf(mf->path, sizeof(mf->path), "%s", path);

    if (out_size) *out_size = size;

    LOG_DEBUG("mmap_pool: mapped %s (%zu MB)", path, size / (1024 * 1024));
    return addr;
}

void mmap_pool_prefetch(void *addr, size_t len) {
    // Hint to the OS that we'll need these pages soon
    madvise(addr, len, MADV_WILLNEED);
}

int mmap_pool_get_fd(MmapPool *pool, int index) {
    if (!pool || index < 0 || index >= pool->count) return -1;
    return pool->files[index].fd;
}

int mmap_pool_count(MmapPool *pool) {
    if (!pool) return 0;
    return pool->count;
}

void mmap_pool_free(MmapPool *pool) {
    if (!pool) return;
    for (int i = 0; i < pool->count; i++) {
        munmap(pool->files[i].addr, pool->files[i].size);
        close(pool->files[i].fd);
    }
    free(pool->files);
    free(pool);
}
