#include <metal_stdlib>
using namespace metal;

// Numerically stable softmax
// out[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
// Used for attention scores and MoE gate routing.

kernel void softmax(
    device const float  *x      [[buffer(0)]],
    device       float  *out    [[buffer(1)]],
    constant     uint   &N      [[buffer(2)]],
    uint                 tid    [[thread_position_in_threadgroup]],
    uint                 tg_size [[threads_per_threadgroup]],
    threadgroup  float  *shared [[threadgroup(0)]])
{
    // Pass 1: find max
    float local_max = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        local_max = max(local_max, x[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    // Pass 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float e = exp(x[i] - max_val);
        out[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared[0];

    // Pass 3: normalize
    for (uint i = tid; i < N; i += tg_size) {
        out[i] /= sum;
    }
}
