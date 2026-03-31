#define _POSIX_C_SOURCE 200809L

#include "tokenizer/byte_decode.h"
#include "util/log.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static void test_byte_roundtrip(void) {
    byte_decode_init();

    // Test: every byte value should round-trip through encode → decode
    for (int b = 0; b < 256; b++) {
        uint8_t byte = (uint8_t)b;
        char encoded[8];
        int elen = byte_encode_string(&byte, 1, encoded, sizeof(encoded));
        assert(elen > 0);

        uint8_t decoded;
        int dlen = byte_decode_token(encoded, (size_t)elen, &decoded, 1);
        assert(dlen == 1);
        assert(decoded == byte);
    }
    printf("test_byte_roundtrip: PASSED\n");
}

static void test_space_encoding(void) {
    byte_decode_init();

    // Space (0x20) should encode to Ġ (U+0120 → UTF-8: 0xC4 0xA0)
    uint8_t space = 0x20;
    char encoded[8];
    int elen = byte_encode_string(&space, 1, encoded, sizeof(encoded));
    assert(elen == 2);
    assert((uint8_t)encoded[0] == 0xC4);
    assert((uint8_t)encoded[1] == 0xA0);

    // Decode "Ġ" back to space
    uint8_t decoded;
    int dlen = byte_decode_token(encoded, (size_t)elen, &decoded, 1);
    assert(dlen == 1);
    assert(decoded == 0x20);

    printf("test_space_encoding: PASSED\n");
}

static void test_newline_encoding(void) {
    byte_decode_init();

    // Newline (0x0A) should encode to Ċ (U+010A → UTF-8: 0xC4 0x8A)
    uint8_t nl = 0x0A;
    char encoded[8];
    int elen = byte_encode_string(&nl, 1, encoded, sizeof(encoded));
    assert(elen == 2);
    assert((uint8_t)encoded[0] == 0xC4);
    assert((uint8_t)encoded[1] == 0x8A);

    uint8_t decoded;
    int dlen = byte_decode_token(encoded, (size_t)elen, &decoded, 1);
    assert(dlen == 1);
    assert(decoded == 0x0A);

    printf("test_newline_encoding: PASSED\n");
}

static void test_printable_passthrough(void) {
    byte_decode_init();

    // Printable ASCII (like 'A' = 0x41) should encode to itself
    uint8_t a = 'A';
    char encoded[8];
    int elen = byte_encode_string(&a, 1, encoded, sizeof(encoded));
    assert(elen == 1);
    assert(encoded[0] == 'A');

    uint8_t decoded;
    int dlen = byte_decode_token("A", 1, &decoded, 1);
    assert(dlen == 1);
    assert(decoded == 'A');

    printf("test_printable_passthrough: PASSED\n");
}

static void test_multichar_decode(void) {
    byte_decode_init();

    // "Hello" should decode to "Hello" (all printable)
    const char *token = "Hello";
    uint8_t decoded[16];
    int dlen = byte_decode_token(token, 5, decoded, sizeof(decoded));
    assert(dlen == 5);
    assert(memcmp(decoded, "Hello", 5) == 0);

    // "Ġthe" should decode to " the" (space + "the")
    // Ġ is U+0120, UTF-8: C4 A0
    const char gthe[] = { (char)0xC4, (char)0xA0, 't', 'h', 'e', '\0' };
    dlen = byte_decode_token(gthe, 5, decoded, sizeof(decoded));
    assert(dlen == 4);
    assert(decoded[0] == ' ');
    assert(decoded[1] == 't');
    assert(decoded[2] == 'h');
    assert(decoded[3] == 'e');

    printf("test_multichar_decode: PASSED\n");
}

int main(void) {
    log_init();
    test_byte_roundtrip();
    test_space_encoding();
    test_newline_encoding();
    test_printable_passthrough();
    test_multichar_decode();
    printf("\nAll tokenizer tests passed.\n");
    return 0;
}
