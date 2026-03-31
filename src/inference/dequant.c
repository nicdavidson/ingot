#include "inference/dequant.h"

#include <string.h>

float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = (uint32_t)bf16 << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

void dequant_row_q4(
    float       *out,
    const void  *weight_raw,
    const void  *scales_raw,
    const void  *biases_raw,
    int          K,
    int          group_size)
{
    const uint32_t *weight = weight_raw;
    const uint16_t *scales = scales_raw;
    const uint16_t *biases = biases_raw;

    int num_groups = K / group_size;

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(scales[g]);
        float zero  = bf16_to_f32(biases[g]);

        int base_k = g * group_size;

        // Each U32 contains 8 x 4-bit values
        int u32s_per_group = group_size / 8;
        int u32_offset = g * u32s_per_group;

        for (int u = 0; u < u32s_per_group; u++) {
            uint32_t packed = weight[u32_offset + u];
            for (int b = 0; b < 8; b++) {
                int val = (int)((packed >> (b * 4)) & 0xF);
                int k = base_k + u * 8 + b;
                if (k < K) {
                    out[k] = ((float)val - zero) * scale;
                }
            }
        }
    }
}

void dequant_matmul_q4(
    float       *out,
    const void  *weight_raw,
    const void  *scales_raw,
    const void  *biases_raw,
    const float *x,
    int          M,
    int          K,
    int          group_size)
{
    const uint32_t *weight = weight_raw;
    const uint16_t *scales = scales_raw;
    const uint16_t *biases = biases_raw;

    int K_packed = K / 8;          // U32s per row
    int num_groups = K / group_size;

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;

        const uint32_t *row_w = weight + m * K_packed;
        const uint16_t *row_s = scales + m * num_groups;
        const uint16_t *row_b = biases + m * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(row_s[g]);
            float zero  = bf16_to_f32(row_b[g]);

            int base_k = g * group_size;
            int u32s_per_group = group_size / 8;
            int u32_base = g * u32s_per_group;

            for (int u = 0; u < u32s_per_group; u++) {
                uint32_t packed = row_w[u32_base + u];
                int k_start = base_k + u * 8;

                // Unroll 8 values from one U32
                for (int b = 0; b < 8; b++) {
                    int val = (int)((packed >> (b * 4)) & 0xF);
                    float dequant = ((float)val - zero) * scale;
                    sum += dequant * x[k_start + b];
                }
            }
        }

        out[m] = sum;
    }
}
