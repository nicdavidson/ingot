#ifndef INGOT_BYTE_DECODE_H
#define INGOT_BYTE_DECODE_H

#include <stddef.h>
#include <stdint.h>

// GPT-2 byte-level BPE uses a mapping from bytes (0-255) to Unicode
// characters. This avoids control characters in the vocab.
// For example: space (0x20) → Ġ (U+0120), newline (0x0A) → Ċ (U+010A)

// Initialize the byte ↔ unicode mapping tables.
void byte_decode_init(void);

// Convert a BPE token string (which uses GPT-2 byte encoding) to raw bytes.
// Returns number of bytes written to out.
int byte_decode_token(const char *token, size_t token_len,
                      uint8_t *out, size_t out_size);

// Convert raw bytes to a GPT-2 byte-encoded string.
// Returns number of chars written to out (not including NUL).
int byte_encode_string(const uint8_t *bytes, size_t len,
                       char *out, size_t out_size);

#endif
