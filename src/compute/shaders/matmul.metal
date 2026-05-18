#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Quantized 4-bit matrix-vector multiply with FMA-restructured dequant
//
// Computes out[m] = dequant(A[m,:]) · x  for one row per threadgroup.
// Format: 8 nibbles packed per uint32, BF16 scale+bias per group of 64.
//
// Threadgroup layout: 256 threads = 8 SIMD groups × 32 lanes
// Each threadgroup processes ONE output row.
// Input vector x is cached in threadgroup shared memory when it fits (K<=4096).
// For larger K, reads directly from device memory (unified memory makes this fast).
// ============================================================================

// BF16 → float conversion
inline float bf16_to_float(ushort val) {
    return as_type<float>(uint(val) << 16);
}

// Shared memory budget: 8192 halfs = 16KB (same budget, 2x elements via half precision)
constant uint SHARED_DIM = 8192;

kernel void matmul_q4_fma(
    device const uint32_t *weights  [[buffer(0)]],  // [M, K/8] packed nibbles
    device const ushort   *scales   [[buffer(1)]],  // [M, num_groups] BF16
    device const ushort   *biases   [[buffer(2)]],  // [M, num_groups] BF16
    device const float    *x        [[buffer(3)]],  // [K]
    device       float    *out      [[buffer(4)]],  // [M]
    constant     uint     &M        [[buffer(5)]],
    constant     uint     &K        [[buffer(6)]],
    constant     uint     &group_size [[buffer(7)]],
    uint                   tg_id    [[threadgroup_position_in_grid]],
    uint                   tid      [[thread_index_in_threadgroup]],
    uint                   simd_lane [[thread_index_in_simdgroup]],
    uint                   simd_id  [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id;
    if (row >= M) return;

    // --- Cache input vector in shared memory if it fits ---
    threadgroup half x_shared[SHARED_DIM];
    bool use_shared = (K <= SHARED_DIM);

    if (use_shared) {
        uint k4 = K / 4;
        for (uint i = tid; i < k4; i += 256) {
            float4 val = *reinterpret_cast<device const float4 *>(x + i * 4);
            x_shared[i * 4]     = half(val.x);
            x_shared[i * 4 + 1] = half(val.y);
            x_shared[i * 4 + 2] = half(val.z);
            x_shared[i * 4 + 3] = half(val.w);
        }
        for (uint i = k4 * 4 + tid; i < K; i += 256) {
            x_shared[i] = half(x[i]);
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // --- Compute dot product ---
    uint K_packed = K / 8;
    uint num_groups = K / group_size;
    uint u32s_per_group = group_size / 8;

    device const uint32_t *row_w = weights + row * K_packed;
    device const ushort   *row_s = scales  + row * num_groups;
    device const ushort   *row_b = biases  + row * num_groups;

    float sum = 0.0f;

    for (uint u = tid; u < K_packed; u += 256) {
        uint g = u / u32s_per_group;
        float scale = bf16_to_float(row_s[g]);
        float bias  = bf16_to_float(row_b[g]);

        uint32_t packed = row_w[u];
        uint k_base = u * 8;

        // Read input values from shared memory or device memory
        float x0, x1, x2, x3, x4, x5, x6, x7;
        if (use_shared) {
            x0 = float(x_shared[k_base]);     x1 = float(x_shared[k_base + 1]);
            x2 = float(x_shared[k_base + 2]); x3 = float(x_shared[k_base + 3]);
            x4 = float(x_shared[k_base + 4]); x5 = float(x_shared[k_base + 5]);
            x6 = float(x_shared[k_base + 6]); x7 = float(x_shared[k_base + 7]);
        } else {
            x0 = x[k_base];     x1 = x[k_base + 1];
            x2 = x[k_base + 2]; x3 = x[k_base + 3];
            x4 = x[k_base + 4]; x5 = x[k_base + 5];
            x6 = x[k_base + 6]; x7 = x[k_base + 7];
        }

        float sx0 = scale * x0, bx0 = bias * x0;
        float sx1 = scale * x1, bx1 = bias * x1;
        float sx2 = scale * x2, bx2 = bias * x2;
        float sx3 = scale * x3, bx3 = bias * x3;
        float sx4 = scale * x4, bx4 = bias * x4;
        float sx5 = scale * x5, bx5 = bias * x5;
        float sx6 = scale * x6, bx6 = bias * x6;
        float sx7 = scale * x7, bx7 = bias * x7;

        sum += bx0 + bx1 + bx2 + bx3 + bx4 + bx5 + bx6 + bx7;
        sum = fma(float(packed        & 0xF), sx0, sum);
        sum = fma(float((packed >> 4)  & 0xF), sx1, sum);
        sum = fma(float((packed >> 8)  & 0xF), sx2, sum);
        sum = fma(float((packed >> 12) & 0xF), sx3, sum);
        sum = fma(float((packed >> 16) & 0xF), sx4, sum);
        sum = fma(float((packed >> 20) & 0xF), sx5, sum);
        sum = fma(float((packed >> 24) & 0xF), sx6, sum);
        sum = fma(float((packed >> 28) & 0xF), sx7, sum);
    }

    // --- SIMD reduction ---
    sum = simd_sum(sum);

    // --- Cross-simdgroup reduction ---
    threadgroup float partial_sums[8];
    if (simd_lane == 0) {
        partial_sums[simd_id] = sum;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < 8; s++) {
            total += partial_sums[s];
        }
        out[row] = total;
    }
}

// ============================================================================
// BF16 matrix-vector multiply
// ============================================================================

kernel void matmul_bf16(
    device const ushort *A       [[buffer(0)]],
    device const float  *x       [[buffer(1)]],
    device       float  *out     [[buffer(2)]],
    constant     uint   &M       [[buffer(3)]],
    constant     uint   &K       [[buffer(4)]],
    uint                 tg_id   [[threadgroup_position_in_grid]],
    uint                 tid     [[thread_index_in_threadgroup]],
    uint                 simd_lane [[thread_index_in_simdgroup]],
    uint                 simd_id [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id;
    if (row >= M) return;

    threadgroup half x_shared[SHARED_DIM];
    bool use_shared = (K <= SHARED_DIM);

    if (use_shared) {
        uint k4 = K / 4;
        for (uint i = tid; i < k4; i += 256) {
            float4 val = *reinterpret_cast<device const float4 *>(x + i * 4);
            x_shared[i * 4]     = val.x;
            x_shared[i * 4 + 1] = val.y;
            x_shared[i * 4 + 2] = val.z;
            x_shared[i * 4 + 3] = val.w;
        }
        for (uint i = k4 * 4 + tid; i < K; i += 256) {
            x_shared[i] = x[i];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    device const ushort *row_a = A + row * K;
    float sum = 0.0f;

    if (use_shared) {
        for (uint k = tid; k < K; k += 256) {
            sum += bf16_to_float(row_a[k]) * float(x_shared[k]);
        }
    } else {
        for (uint k = tid; k < K; k += 256) {
            sum += bf16_to_float(row_a[k]) * x[k];
        }
    }

    sum = simd_sum(sum);

    threadgroup float partial_sums[8];
    if (simd_lane == 0) {
        partial_sums[simd_id] = sum;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < 8; s++) {
            total += partial_sums[s];
        }
        out[row] = total;
    }
}
