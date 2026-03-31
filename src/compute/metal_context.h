#ifndef INGOT_METAL_CONTEXT_H
#define INGOT_METAL_CONTEXT_H

#include <stdbool.h>
#include <stddef.h>

// Opaque Metal context — wraps MTLDevice, command queue, compiled pipelines.
// Only available on macOS (PLATFORM_MACOS).
typedef struct MetalContext MetalContext;

// Initialize Metal: create device, command queue, compile all shaders.
// Returns NULL on failure (e.g., no Metal GPU available).
MetalContext *metal_init(void);

// Free Metal context and all resources.
void metal_free(MetalContext *ctx);

// Wrap an existing memory region (e.g., mmap'd weights) as a Metal buffer.
// Uses newBufferWithBytesNoCopy for zero-copy GPU access.
// Returns an opaque buffer handle, or NULL on failure.
void *metal_wrap_buffer(MetalContext *ctx, void *data, size_t size);

// Create a new GPU buffer of given size.
void *metal_alloc_buffer(MetalContext *ctx, size_t size);

// Free a Metal buffer.
void metal_free_buffer(void *buffer);

// Synchronize — wait for all enqueued GPU work to complete.
void metal_sync(MetalContext *ctx);

#endif
