#include "inference/dequant.h"

#include <math.h>
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
    int u32s_per_group = group_size / 8;

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;

        const uint32_t *row_w = weight + (size_t)m * K_packed;
        const uint16_t *row_s = scales + (size_t)m * num_groups;
        const uint16_t *row_b = biases + (size_t)m * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(row_s[g]);
            float zero  = bf16_to_f32(row_b[g]);

            int base_k = g * group_size;
            int u32_base = g * u32s_per_group;

            for (int u = 0; u < u32s_per_group; u++) {
                uint32_t packed = row_w[u32_base + u];
                int k_start = base_k + u * 8;

                // FMA-restructured dequant: fma(nibble, scale*x, -zero*scale*x)
                // Saturates FMA pipeline instead of creating dependency chain
                // Original: sum += ((val - zero) * scale) * x[k]
                // Restructured: sum += fma(val, scale*x[k], -zero*scale*x[k])
                float sx0 = scale * x[k_start];
                float sx1 = scale * x[k_start + 1];
                float sx2 = scale * x[k_start + 2];
                float sx3 = scale * x[k_start + 3];
                float sx4 = scale * x[k_start + 4];
                float sx5 = scale * x[k_start + 5];
                float sx6 = scale * x[k_start + 6];
                float sx7 = scale * x[k_start + 7];

                float nz = -zero;
                sum += fmaf((float)((packed)       & 0xF), sx0, nz * sx0);
                sum += fmaf((float)((packed >> 4)  & 0xF), sx1, nz * sx1);
                sum += fmaf((float)((packed >> 8)  & 0xF), sx2, nz * sx2);
                sum += fmaf((float)((packed >> 12) & 0xF), sx3, nz * sx3);
                sum += fmaf((float)((packed >> 16) & 0xF), sx4, nz * sx4);
                sum += fmaf((float)((packed >> 20) & 0xF), sx5, nz * sx5);
                sum += fmaf((float)((packed >> 24) & 0xF), sx6, nz * sx6);
                sum += fmaf((float)((packed >> 28) & 0xF), sx7, nz * sx7);
            }
        }

        out[m] = sum;
    }
}
