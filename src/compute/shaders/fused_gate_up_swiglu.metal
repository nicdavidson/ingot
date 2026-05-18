#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Fused Gate+Up+SwiGLU kernel
//
// Reads input vector ONCE, computes both gate and up projections, then
// applies SiLU(gate) * up in a single kernel. Saves a full round-trip
// to memory vs. doing gate_proj, up_proj, and SiLU separately.
//
// gate_proj: [moe_dim, K] Q4  →  gate[moe_dim]
// up_proj:   [moe_dim, K] Q4  →  up[moe_dim]
// output:    SiLU(gate) * up   →  out[moe_dim]
//
// Each threadgroup processes one output element.
// 256 threads, shared memory for input vector, SIMD reduction.
// ============================================================================

inline float bf16_to_float(ushort val) {
    return as_type<float>(uint(val) << 16);
}

constant uint MAX_SHARED_DIM = 8192;

kernel void fused_gate_up_swiglu(
    // Gate projection weights
    device const uint32_t *gate_w   [[buffer(0)]],   // [moe_dim, K/8]
    device const ushort   *gate_s   [[buffer(1)]],   // [moe_dim, num_groups]
    device const ushort   *gate_b   [[buffer(2)]],   // [moe_dim, num_groups]
    // Up projection weights
    device const uint32_t *up_w     [[buffer(3)]],   // [moe_dim, K/8]
    device const ushort   *up_s     [[buffer(4)]],   // [moe_dim, num_groups]
    device const ushort   *up_b     [[buffer(5)]],   // [moe_dim, num_groups]
    // Input and output
    device const float    *x        [[buffer(6)]],   // [K]
    device       float    *out      [[buffer(7)]],   // [moe_dim]
    // Dimensions
    constant     uint     &moe_dim  [[buffer(8)]],
    constant     uint     &K        [[buffer(9)]],
    constant     uint     &group_size [[buffer(10)]],
    // Thread info
    uint                   tg_id    [[threadgroup_position_in_grid]],
    uint                   tid      [[thread_index_in_threadgroup]],
    uint                   simd_lane [[thread_index_in_simdgroup]],
    uint                   simd_id  [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id;
    if (row >= moe_dim) return;

    // Cache input in shared memory (loaded once, used twice for gate and up)
    threadgroup half x_shared[MAX_SHARED_DIM];
    bool use_shared = (K <= MAX_SHARED_DIM);
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
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint K_packed = K / 8;
    uint num_groups = K / group_size;
    uint u32s_per_group = group_size / 8;

    // Compute both gate and up dot products simultaneously
    device const uint32_t *grow_w = gate_w + row * K_packed;
    device const ushort   *grow_s = gate_s + row * num_groups;
    device const ushort   *grow_b = gate_b + row * num_groups;

    device const uint32_t *urow_w = up_w + row * K_packed;
    device const ushort   *urow_s = up_s + row * num_groups;
    device const ushort   *urow_b = up_b + row * num_groups;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (uint u = tid; u < K_packed; u += 256) {
        uint g = u / u32s_per_group;
        uint k_base = u * 8;

        float gscale = bf16_to_float(grow_s[g]);
        float gbias  = bf16_to_float(grow_b[g]);
        uint32_t gpacked = grow_w[u];

        float uscale = bf16_to_float(urow_s[g]);
        float ubias  = bf16_to_float(urow_b[g]);
        uint32_t upacked = urow_w[u];

        // Pre-compute scale*x and bias*x, accumulate bias terms,
        // then FMA nibble * sx into running sums
        for (uint b = 0; b < 8; b++) {
            float xv = use_shared ? float(x_shared[k_base + b]) : x[k_base + b];
            float gsx = gscale * xv, gbx = gbias * xv;
            float usx = uscale * xv, ubx = ubias * xv;
            gate_sum += gbx;
            gate_sum = fma(float((gpacked >> (b * 4)) & 0xF), gsx, gate_sum);
            up_sum += ubx;
            up_sum = fma(float((upacked >> (b * 4)) & 0xF), usx, up_sum);
        }
    }

    // SIMD reduction
    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    threadgroup float gate_partials[8];
    threadgroup float up_partials[8];
    if (simd_lane == 0) {
        gate_partials[simd_id] = gate_sum;
        up_partials[simd_id]   = up_sum;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
        for (uint s = 0; s < 8; s++) {
            g += gate_partials[s];
            u += up_partials[s];
        }
        // SiLU(gate) * up
        float silu = g / (1.0f + exp(-g));
        out[row] = silu * u;
    }
}
