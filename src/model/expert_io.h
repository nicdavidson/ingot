#ifndef INGOT_EXPERT_IO_H
#define INGOT_EXPERT_IO_H

#include <stddef.h>

typedef struct ExpertIO ExpertIO;

// Create expert I/O subsystem for parallel pread-based expert loading.
// max_concurrent: max experts to read in parallel (typically num_experts_per_tok)
ExpertIO *expert_io_create(int max_concurrent);

// Issue parallel pread() for multiple experts. Non-blocking — starts reads immediately.
// fd: file descriptor for this layer's expert file
// offsets[count]: byte offset of each expert within the file
// sizes[count]: byte size of each expert
// destinations[count]: pre-allocated buffers to read into (e.g. Metal unified memory)
void expert_io_fetch(ExpertIO *io, int fd,
                     size_t *offsets, size_t *sizes,
                     void **destinations, int count);

// Block until all in-flight reads complete.
void expert_io_wait(ExpertIO *io);

// Free the expert I/O subsystem.
void expert_io_free(ExpertIO *io);

#endif
