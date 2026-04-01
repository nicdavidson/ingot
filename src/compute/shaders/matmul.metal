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
// Input vector x is cached in threadgroup shared memory (16KB budget).
// ============================================================================

// BF16 → float conversion
inline float bf16_to_float(ushort val) {
    return as_type<float>(uint(val) << 16);
}

// Shared memory: cache the input vector x (up to 4096 floats = 16KB)
constant uint MAX_SHARED_DIM = 4096;

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
    // Each threadgroup handles one output row
    uint row = tg_id;
    if (row >= M) return;

    // --- Cache input vector in shared memory ---
    threadgroup float x_shared[MAX_SHARED_DIM];
    // 256 threads cooperatively load x using float4 for coalesced access
    uint k4 = K / 4;
    for (uint i = tid; i < k4; i += 256) {
        // Vector load: 4 floats at once
        float4 val = *reinterpret_cast<device const float4 *>(x + i * 4);
        x_shared[i * 4]     = val.x;
        x_shared[i * 4 + 1] = val.y;
        x_shared[i * 4 + 2] = val.z;
        x_shared[i * 4 + 3] = val.w;
    }
    // Handle remainder
    for (uint i = k4 * 4 + tid; i < K; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // --- Compute dot product with strided assignment ---
    uint K_packed = K / 8;       // uint32s per row
    uint num_groups = K / group_size;
    uint u32s_per_group = group_size / 8;  // typically 8 for group_size=64

    device const uint32_t *row_w = weights + row * K_packed;
    device const ushort   *row_s = scales  + row * num_groups;
    device const ushort   *row_b = biases  + row * num_groups;

    float sum = 0.0f;

    // Each thread processes strided uint32 words across all groups
    // Total uint32s = K/8. 256 threads each handle K/8/256 words.
    for (uint u = tid; u < K_packed; u += 256) {
        // Which group does this uint32 belong to?
        uint g = u / u32s_per_group;
        float scale = bf16_to_float(row_s[g]);
        float bias  = bf16_to_float(row_b[g]);

        uint32_t packed = row_w[u];
        uint k_base = u * 8;

        // Pre-compute scale*x and bias*x for each position.
        // FMA: result += nibble * (scale*x) + (bias*x)
        // This collapses dequant + dot product into a single FMA per nibble.
        float sx0 = scale * x_shared[k_base],     bx0 = bias * x_shared[k_base];
        float sx1 = scale * x_shared[k_base + 1], bx1 = bias * x_shared[k_base + 1];
        float sx2 = scale * x_shared[k_base + 2], bx2 = bias * x_shared[k_base + 2];
        float sx3 = scale * x_shared[k_base + 3], bx3 = bias * x_shared[k_base + 3];
        float sx4 = scale * x_shared[k_base + 4], bx4 = bias * x_shared[k_base + 4];
        float sx5 = scale * x_shared[k_base + 5], bx5 = bias * x_shared[k_base + 5];
        float sx6 = scale * x_shared[k_base + 6], bx6 = bias * x_shared[k_base + 6];
        float sx7 = scale * x_shared[k_base + 7], bx7 = bias * x_shared[k_base + 7];

        // Accumulate bias*x terms first, then FMA nibble * sx into sum
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

    // --- SIMD reduction within each simdgroup ---
    sum = simd_sum(sum);

    // --- Cross-simdgroup reduction via shared memory ---
    threadgroup float partial_sums[8];  // one per simdgroup
    if (simd_lane == 0) {
        partial_sums[simd_id] = sum;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Thread 0 writes the final result
    if (tid == 0) {
        float total = 0.0f;
        // 256 threads / 32 = 8 simdgroups
        for (uint s = 0; s < 8; s++) {
            total += partial_sums[s];
        }
        out[row] = total;
    }
}

// ============================================================================
// BF16 matrix-vector multiply (for norm weights, embedding, etc.)
// Same threadgroup pattern as above but simpler dequant.
// ============================================================================

kernel void matmul_bf16(
    device const ushort *A       [[buffer(0)]],  // [M, K] BF16
    device const float  *x       [[buffer(1)]],  // [K]
    device       float  *out     [[buffer(2)]],  // [M]
    constant     uint   &M       [[buffer(3)]],
    constant     uint   &K       [[buffer(4)]],
    uint                 tg_id   [[threadgroup_position_in_grid]],
    uint                 tid     [[thread_index_in_threadgroup]],
    uint                 simd_lane [[thread_index_in_simdgroup]],
    uint                 simd_id [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id;
    if (row >= M) return;

    threadgroup float x_shared[MAX_SHARED_DIM];
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

    device const ushort *row_a = A + row * K;
    float sum = 0.0f;

    for (uint k = tid; k < K; k += 256) {
        sum += bf16_to_float(row_a[k]) * x_shared[k];
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
