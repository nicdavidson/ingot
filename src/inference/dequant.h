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

// --- 6-bit quantization (MLX contiguous bit packing) ---
// Weight layout: values packed contiguously at 6 bits each across U32 boundaries.
// Per group of `group_size` values: group_size*6/32 U32s.
// Total weight shape: [M, K*6/32] U32.
// Scales/biases: same as Q4 — [M, K/group_size] BF16.

void dequant_matmul_q6(
    float       *out,          // [M]
    const void  *weight,       // [M, K*6/32] U32
    const void  *scales,       // [M, G] BF16
    const void  *biases,       // [M, G] BF16
    const float *x,            // [K]
    int          M,
    int          K,
    int          group_size
);

void dequant_row_q6(
    float       *out,         // [K]
    const void  *weight,      // [K*6/32] U32
    const void  *scales,      // [G] BF16
    const void  *biases,      // [G] BF16
    int          K,
    int          group_size
);

// --- 2-bit quantization ---
// Weight layout: 16 values per U32 (2 bits each, clean division).
// Weight shape: [M, K/16] U32.
// Scales/biases: [M, K/group_size] BF16.

void dequant_matmul_q2(
    float       *out,          // [M]
    const void  *weight,       // [M, K/16] U32
    const void  *scales,       // [M, G] BF16
    const void  *biases,       // [M, G] BF16
    const float *x,            // [K]
    int          M,
    int          K,
    int          group_size
);

void dequant_row_q2(
    float       *out,         // [K]
    const void  *weight,      // [K/16] U32
    const void  *scales,      // [G] BF16
    const void  *biases,      // [G] BF16
    int          K,
    int          group_size
);

// --- FP8 quantize + dequantize (in-place, simulates QAT precision) ---
// V4-Flash was trained with FP8 activations on KV non-rope dims, with E8M0
// power-of-2 scales per block_size group. Skipping this fake-quant produces
// gradual distribution drift over many layers.
//
// E4M3 FP8: sign(1) + exp(4) + mantissa(3), bias=7, max=448.
// E8M0 scale: power-of-2 only, so scale = 2^ceil(log2(amax / 448)).

void fp8_act_quant_inplace(float *x, int len, int block_size);

// --- MXFP4 (Microscaling FP4) — used by DeepSeek V4 routed experts ---
// Weights are 4-bit FP4 (E2M1) packed 8 per U32, identical layout to Q4.
// Scales are E8M0 stored as U8: scale = 2^(byte - 127). Group size = 32.
// No biases.
//
// FP4 codes: 0..7 = +0,+0.5,+1,+1.5,+2,+3,+4,+6;  8..15 = negatives.

void dequant_matmul_mxfp4(
    float       *out,          // [M]
    const void  *weight,       // [M, K/8] U32
    const void  *scales,       // [M, G] U8 (E8M0 exponents); G = K/group_size
    const float *x,            // [K]
    int          M,
    int          K,
    int          group_size    // 32 for V4
);

#endif
