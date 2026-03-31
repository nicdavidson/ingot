#ifndef INGOT_DEQUANT_H
#define INGOT_DEQUANT_H

#include <stddef.h>
#include <stdint.h>

// Dequantize a 4-bit quantized matrix-vector product.
// The Qwen 3.5 4-bit format:
//   weight: U32 packed (8 x 4-bit values per U32)
//   scales: BF16 per group
//   biases: BF16 per group (zero-point, not additive bias)
//
// Layout for a [M, K] matrix:
//   weight shape: [M, K/8]  (K values packed 8 per U32)
//   scales shape: [M, G]    (G = K/group_size, group_size typically 64)
//   biases shape: [M, G]
//
// Computes out[M] = dequant(weight) @ x[K]

void dequant_matmul_q4(
    float       *out,          // [M]
    const void  *weight,       // [M, K/8] U32
    const void  *scales,       // [M, G] BF16
    const void  *biases,       // [M, G] BF16
    const float *x,            // [K]
    int          M,
    int          K,
    int          group_size    // typically 64
);

// Convert BF16 to float
float bf16_to_f32(uint16_t bf16);

// Dequantize a single row and write to float buffer
void dequant_row_q4(
    float       *out,         // [K]
    const void  *weight,      // [K/8] U32
    const void  *scales,      // [G] BF16
    const void  *biases,      // [G] BF16
    int          K,
    int          group_size
);

#endif
