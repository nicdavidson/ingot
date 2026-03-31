#ifndef INGOT_SSE_H
#define INGOT_SSE_H

#include <stddef.h>

// Write SSE headers to a socket.
void sse_write_headers(int fd);

// Write an SSE data event. Handles multi-line data.
void sse_write_event(int fd, const char *data, size_t len);

// Write the SSE [DONE] event.
void sse_write_done(int fd);

// Format a chat completion chunk as SSE JSON.
// Returns bytes written into buf.
int sse_format_chunk(char *buf, size_t buf_size,
                     const char *model_name,
                     const char *content,
                     const char *finish_reason,
                     const char *chunk_id,
                     int index);

#endif
