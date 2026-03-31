#include "tokenizer/byte_decode.h"

#include <string.h>

// GPT-2 maps bytes 0-255 to unicode code points to avoid control chars.
// Printable ASCII bytes (33-126, 161-172, 174-255) map to themselves.
// The remaining 68 bytes map to U+0100 through U+0143 (Ā-ń range).
//
// This means:
//   byte 0x20 (space) → U+0120 (Ġ)
//   byte 0x0A (newline) → U+010A (Ċ)
//   byte 0x00 → U+0100 (Ā)
//   etc.

// unicode_to_byte[codepoint] = byte value (for the 256 mapped codepoints)
static uint8_t unicode_to_byte[512];
// byte_to_unicode[byte] = unicode codepoint
static uint32_t byte_to_unicode[256];
static int initialized = 0;

void byte_decode_init(void) {
    if (initialized) return;

    memset(unicode_to_byte, 0, sizeof(unicode_to_byte));
    memset(byte_to_unicode, 0, sizeof(byte_to_unicode));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        // Printable ranges that map to themselves
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            byte_to_unicode[b] = (uint32_t)b;
            unicode_to_byte[b] = (uint8_t)b;
        } else {
            // Non-printable bytes get mapped to U+0100 + offset
            uint32_t cp = 256 + (uint32_t)n;
            byte_to_unicode[b] = cp;
            if (cp < 512) unicode_to_byte[cp] = (uint8_t)b;
            n++;
        }
    }

    initialized = 1;
}

// Read one UTF-8 character from src, return its codepoint and advance *pos.
static uint32_t read_utf8(const char *src, size_t len, size_t *pos) {
    uint8_t c = (uint8_t)src[*pos];
    uint32_t cp;
    int extra;

    if (c < 0x80) {
        cp = c; extra = 0;
    } else if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F; extra = 1;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F; extra = 2;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07; extra = 3;
    } else {
        (*pos)++;
        return 0xFFFD; // replacement char
    }

    (*pos)++;
    for (int i = 0; i < extra && *pos < len; i++) {
        cp = (cp << 6) | ((uint8_t)src[*pos] & 0x3F);
        (*pos)++;
    }
    return cp;
}

int byte_decode_token(const char *token, size_t token_len,
                      uint8_t *out, size_t out_size) {
    if (!initialized) byte_decode_init();

    int written = 0;
    size_t pos = 0;

    while (pos < token_len && (size_t)written < out_size) {
        uint32_t cp = read_utf8(token, token_len, &pos);

        // Check if this codepoint is in our mapping
        if (cp < 512 && (unicode_to_byte[cp] != 0 || cp == byte_to_unicode[0])) {
            out[written++] = unicode_to_byte[cp];
        } else if (cp < 256) {
            // Direct mapping for printable chars
            out[written++] = (uint8_t)cp;
        }
        // else skip unmapped codepoints
    }

    return written;
}

// Write a unicode codepoint as UTF-8 into buf. Returns bytes written.
static int write_utf8(uint32_t cp, char *buf, size_t remaining) {
    if (cp < 0x80 && remaining >= 1) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp < 0x800 && remaining >= 2) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000 && remaining >= 3) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else if (remaining >= 4) {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

int byte_encode_string(const uint8_t *bytes, size_t len,
                       char *out, size_t out_size) {
    if (!initialized) byte_decode_init();

    int written = 0;
    for (size_t i = 0; i < len; i++) {
        uint32_t cp = byte_to_unicode[bytes[i]];
        int n = write_utf8(cp, out + written, out_size - (size_t)written);
        if (n == 0) break;
        written += n;
    }
    if ((size_t)written < out_size) out[written] = '\0';
    return written;
}
