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
        float bias  = bf16_to_f32(biases[g]);

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
                    out[k] = (float)val * scale + bias;
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
    const uint16_t *biases = biases_raw;  // may be NULL (V4 packed experts have no biases)

    int K_packed = K / 8;          // U32s per row
    int num_groups = K / group_size;
    int u32s_per_group = group_size / 8;

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;

        const uint32_t *row_w = weight + (size_t)m * K_packed;
        const uint16_t *row_s = scales + (size_t)m * num_groups;
        const uint16_t *row_b = biases ? biases + (size_t)m * num_groups : NULL;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(row_s[g]);
            float bias  = row_b ? bf16_to_f32(row_b[g]) : 0.0f;

            int base_k = g * group_size;
            int u32_base = g * u32s_per_group;

            for (int u = 0; u < u32s_per_group; u++) {
                uint32_t packed = row_w[u32_base + u];
                int k_start = base_k + u * 8;

                // FMA-restructured affine dequant: fma(nibble, scale*x, bias*x)
                // Correct formula: dequant = nibble * scale + bias
                // FMA form: sum += (nibble * scale + bias) * x = fma(nibble, scale*x, bias*x)
                const float *xp = x + k_start;
                sum += fmaf((float)((packed)       & 0xF), scale * xp[0], bias * xp[0]);
                sum += fmaf((float)((packed >> 4)  & 0xF), scale * xp[1], bias * xp[1]);
                sum += fmaf((float)((packed >> 8)  & 0xF), scale * xp[2], bias * xp[2]);
                sum += fmaf((float)((packed >> 12) & 0xF), scale * xp[3], bias * xp[3]);
                sum += fmaf((float)((packed >> 16) & 0xF), scale * xp[4], bias * xp[4]);
                sum += fmaf((float)((packed >> 20) & 0xF), scale * xp[5], bias * xp[5]);
                sum += fmaf((float)((packed >> 24) & 0xF), scale * xp[6], bias * xp[6]);
                sum += fmaf((float)((packed >> 28) & 0xF), scale * xp[7], bias * xp[7]);
            }
        }

        out[m] = sum;
    }
}


// --- 6-bit dequantization (MLX contiguous bit packing) ---
//
// Values are packed contiguously at 6 bits each. A value may straddle two U32s.
// Within each group of group_size values:
//   group_size * 6 / 32 U32s (e.g., 128*6/32 = 24 U32s for gs=128)
//
// Extraction uses a 64-bit window to avoid branch on straddling:
//   combined = (uint64_t)w[idx] | ((uint64_t)w[idx+1] << 32)
//   val = (combined >> bit_offset) & 0x3F

void dequant_row_q6(
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
    int u32s_per_group = group_size * 6 / 32;

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(scales[g]);
        float bias  = bf16_to_f32(biases[g]);

        int base_k = g * group_size;
        const uint32_t *gw = weight + g * u32s_per_group;

        for (int i = 0; i < group_size; i++) {
            int bit_pos = i * 6;
            int u32_idx = bit_pos / 32;
            int bit_off = bit_pos % 32;

            // 64-bit window extraction — handles straddling without branching
            uint64_t combined = (uint64_t)gw[u32_idx];
            if (bit_off + 6 > 32 && u32_idx + 1 < u32s_per_group)
                combined |= (uint64_t)gw[u32_idx + 1] << 32;
            int val = (int)((combined >> bit_off) & 0x3F);

            out[base_k + i] = (float)val * scale + bias;
        }
    }
}

void dequant_matmul_q6(
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

    int K_packed = K * 6 / 32;    // U32s per row
    int num_groups = K / group_size;
    int u32s_per_group = group_size * 6 / 32;

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;

        const uint32_t *row_w = weight + (size_t)m * K_packed;
        const uint16_t *row_s = scales + (size_t)m * num_groups;
        const uint16_t *row_b = biases + (size_t)m * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(row_s[g]);
            float bias  = bf16_to_f32(row_b[g]);

            int base_k = g * group_size;
            const uint32_t *gw = row_w + g * u32s_per_group;

            // Pre-compute bias contribution: sum(bias * x[k]) for this group
            float bias_sum = 0.0f;
            for (int i = 0; i < group_size; i++) {
                bias_sum += x[base_k + i];
            }
            sum += bias * bias_sum;

            // Extract and FMA each 6-bit value
            for (int i = 0; i < group_size; i++) {
                int bit_pos = i * 6;
                int u32_idx = bit_pos / 32;
                int bit_off = bit_pos % 32;

                uint64_t combined = (uint64_t)gw[u32_idx];
                if (bit_off + 6 > 32)
                    combined |= (uint64_t)gw[u32_idx + 1] << 32;
                int val = (int)((combined >> bit_off) & 0x3F);

                sum += (float)val * scale * x[base_k + i];
            }
        }

        out[m] = sum;
    }
}


// --- 2-bit dequantization ---
//
// 16 values per U32 (2 bits each, clean division).
// Within each group of group_size values: group_size/16 U32s.

void dequant_row_q2(
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
    int u32s_per_group = group_size / 16;

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(scales[g]);
        float bias  = bf16_to_f32(biases[g]);

        int base_k = g * group_size;
        int u32_offset = g * u32s_per_group;

        for (int u = 0; u < u32s_per_group; u++) {
            uint32_t packed = weight[u32_offset + u];
            for (int b = 0; b < 16; b++) {
                int val = (int)((packed >> (b * 2)) & 0x3);
                int k = base_k + u * 16 + b;
                if (k < K) {
                    out[k] = (float)val * scale + bias;
                }
            }
        }
    }
}

void dequant_matmul_q2(
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

    int K_packed = K / 16;         // U32s per row
    int num_groups = K / group_size;
    int u32s_per_group = group_size / 16;

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;

        const uint32_t *row_w = weight + (size_t)m * K_packed;
        const uint16_t *row_s = scales + (size_t)m * num_groups;
        const uint16_t *row_b = biases + (size_t)m * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(row_s[g]);
            float bias  = bf16_to_f32(row_b[g]);

            int base_k = g * group_size;
            int u32_base = g * u32s_per_group;

            for (int u = 0; u < u32s_per_group; u++) {
                uint32_t packed = row_w[u32_base + u];
                int k_start = base_k + u * 16;

                // Unrolled 2-bit extraction with FMA
                const float *xp = x + k_start;
                sum += fmaf((float)((packed)       & 0x3), scale * xp[0],  bias * xp[0]);
                sum += fmaf((float)((packed >> 2)  & 0x3), scale * xp[1],  bias * xp[1]);
                sum += fmaf((float)((packed >> 4)  & 0x3), scale * xp[2],  bias * xp[2]);
                sum += fmaf((float)((packed >> 6)  & 0x3), scale * xp[3],  bias * xp[3]);
                sum += fmaf((float)((packed >> 8)  & 0x3), scale * xp[4],  bias * xp[4]);
                sum += fmaf((float)((packed >> 10) & 0x3), scale * xp[5],  bias * xp[5]);
                sum += fmaf((float)((packed >> 12) & 0x3), scale * xp[6],  bias * xp[6]);
                sum += fmaf((float)((packed >> 14) & 0x3), scale * xp[7],  bias * xp[7]);
                sum += fmaf((float)((packed >> 16) & 0x3), scale * xp[8],  bias * xp[8]);
                sum += fmaf((float)((packed >> 18) & 0x3), scale * xp[9],  bias * xp[9]);
                sum += fmaf((float)((packed >> 20) & 0x3), scale * xp[10], bias * xp[10]);
                sum += fmaf((float)((packed >> 22) & 0x3), scale * xp[11], bias * xp[11]);
                sum += fmaf((float)((packed >> 24) & 0x3), scale * xp[12], bias * xp[12]);
                sum += fmaf((float)((packed >> 26) & 0x3), scale * xp[13], bias * xp[13]);
                sum += fmaf((float)((packed >> 28) & 0x3), scale * xp[14], bias * xp[14]);
                sum += fmaf((float)((packed >> 30) & 0x3), scale * xp[15], bias * xp[15]);
            }
        }

        out[m] = sum;
    }
}

// MXFP4 (FP4 E2M1 + E8M0 group exponent) used by V4 routed experts.
//
// FP4 E2M1 codes (sign | exp(2) | mantissa(1)):
//   0:+0   1:+0.5  2:+1   3:+1.5  4:+2  5:+3  6:+4  7:+6
//   8:-0   9:-0.5  10:-1  11:-1.5 12:-2 13:-3 14:-4 15:-6
// Group exponent stored as U8: dequant = fp4_value * 2^(byte - 127).
static const float MXFP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

void dequant_matmul_mxfp4(
    float       *out,
    const void  *weight_raw,
    const void  *scales_raw,
    const float *x,
    int          M,
    int          K,
    int          group_size)
{
    const uint32_t *weight = weight_raw;
    const uint8_t  *scales = scales_raw;

    int K_packed = K / 8;
    int num_groups = K / group_size;
    int u32s_per_group = group_size / 8;

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        const uint32_t *row_w = weight + (size_t)m * K_packed;
        const uint8_t  *row_s = scales + (size_t)m * num_groups;

        for (int g = 0; g < num_groups; g++) {
            // E8M0: scale = 2^(byte - 127). ldexpf is fast and exact.
            float scale = ldexpf(1.0f, (int)row_s[g] - 127);
            int base_k = g * group_size;
            int u32_base = g * u32s_per_group;

            for (int u = 0; u < u32s_per_group; u++) {
                uint32_t packed = row_w[u32_base + u];
                const float *xp = x + base_k + u * 8;
                sum += MXFP4_LUT[(packed)       & 0xF] * scale * xp[0];
                sum += MXFP4_LUT[(packed >> 4)  & 0xF] * scale * xp[1];
                sum += MXFP4_LUT[(packed >> 8)  & 0xF] * scale * xp[2];
                sum += MXFP4_LUT[(packed >> 12) & 0xF] * scale * xp[3];
                sum += MXFP4_LUT[(packed >> 16) & 0xF] * scale * xp[4];
                sum += MXFP4_LUT[(packed >> 20) & 0xF] * scale * xp[5];
                sum += MXFP4_LUT[(packed >> 24) & 0xF] * scale * xp[6];
                sum += MXFP4_LUT[(packed >> 28) & 0xF] * scale * xp[7];
            }
        }

        out[m] = sum;
    }
}
