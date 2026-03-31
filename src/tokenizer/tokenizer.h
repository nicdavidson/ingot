#ifndef INGOT_TOKENIZER_H
#define INGOT_TOKENIZER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct Tokenizer Tokenizer;

// Load tokenizer from HuggingFace model directory (vocab.json + merges.txt).
// Returns NULL on failure.
Tokenizer *tokenizer_load(const char *model_dir);

// Free tokenizer.
void tokenizer_free(Tokenizer *tok);

// Encode text to token IDs. Returns number of tokens written.
// If out is NULL, just returns the count.
int tokenizer_encode(const Tokenizer *tok, const char *text, size_t text_len,
                     int32_t *out, int max_tokens);

// Decode a single token ID to UTF-8 string.
// Returns pointer to internal buffer (valid until next decode call).
const char *tokenizer_decode(const Tokenizer *tok, int32_t token_id);

// Decode a sequence of token IDs to UTF-8 string.
// Writes into buf, NUL-terminates. Returns number of bytes written.
int tokenizer_decode_batch(const Tokenizer *tok, const int32_t *tokens,
                           int num_tokens, char *buf, size_t buf_size);

// Get vocab size.
int tokenizer_vocab_size(const Tokenizer *tok);

// Special token IDs
int tokenizer_eos_id(const Tokenizer *tok);
int tokenizer_im_start_id(const Tokenizer *tok);
int tokenizer_im_end_id(const Tokenizer *tok);

#endif
