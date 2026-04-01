#import <dispatch/dispatch.h>

#include "model/expert_io.h"
#include "util/log.h"

#include <stdlib.h>
#include <unistd.h>

struct ExpertIO {
    dispatch_group_t   group;
    dispatch_queue_t   queue;
    int                max_concurrent;
};

ExpertIO *expert_io_create(int max_concurrent) {
    ExpertIO *io = calloc(1, sizeof(ExpertIO));
    if (!io) return NULL;

    io->group = dispatch_group_create();
    // Concurrent queue — GCD will dispatch pread() calls in parallel,
    // letting the SSD controller serve multiple outstanding reads.
    io->queue = dispatch_queue_create("ingot.expert_io",
                                       DISPATCH_QUEUE_CONCURRENT);
    io->max_concurrent = max_concurrent;

    LOG_INFO("expert_io: created (max_concurrent=%d)", max_concurrent);
    return io;
}

void expert_io_fetch(ExpertIO *io, int fd,
                     size_t *offsets, size_t *sizes,
                     void **destinations, int count) {
    for (int i = 0; i < count; i++) {
        size_t offset = offsets[i];
        size_t size   = sizes[i];
        void  *dest   = destinations[i];

        dispatch_group_async(io->group, io->queue, ^{
            size_t remaining = size;
            size_t off = offset;
            char *buf = (char *)dest;

            while (remaining > 0) {
                ssize_t n = pread(fd, buf, remaining, (off_t)off);
                if (n <= 0) {
                    LOG_ERROR("expert_io: pread failed (offset=%zu, remaining=%zu)",
                              off, remaining);
                    break;
                }
                buf += n;
                off += (size_t)n;
                remaining -= (size_t)n;
            }
        });
    }
}

void expert_io_wait(ExpertIO *io) {
    dispatch_group_wait(io->group, DISPATCH_TIME_FOREVER);
}

void expert_io_free(ExpertIO *io) {
    if (!io) return;
    // ARC handles dispatch object release
    free(io);
}
