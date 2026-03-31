#include <metal_stdlib>
using namespace metal;

// RMS Normalization: out[i] = (x[i] / rms) * weight[i]
// where rms = sqrt(mean(x^2) + eps)
//
// Qwen uses RMSNorm (no bias, no mean subtraction) with eps=1e-6.
// Two-pass: first compute sum of squares, then normalize.

// Pass 1: compute partial sum of squares per threadgroup
kernel void rmsnorm_sum_sq(
    device const float  *x       [[buffer(0)]],
    device       float  *partial [[buffer(1)]],
    constant     uint   &N       [[buffer(2)]],
    uint                 gid     [[thread_position_in_grid]],
    uint                 tid     [[thread_position_in_threadgroup]],
    uint                 tg_size [[threads_per_threadgroup]],
    uint                 tg_id   [[threadgroup_position_in_grid]],
    threadgroup  float  *shared  [[threadgroup(0)]])
{
    float sum = 0.0f;
    for (uint i = gid; i < N; i += tg_size) {
        float val = x[i];
        sum += val * val;
    }
    shared[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within threadgroup
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial[tg_id] = shared[0];
    }
}

// Single-threadgroup RMSNorm for typical hidden sizes (up to 4096)
kernel void rmsnorm(
    device const float  *x      [[buffer(0)]],
    device const float  *weight [[buffer(1)]],
    device       float  *out    [[buffer(2)]],
    constant     uint   &N      [[buffer(3)]],
    constant     float  &eps    [[buffer(4)]],
    uint                 gid    [[thread_position_in_grid]],
    uint                 tid    [[thread_position_in_threadgroup]],
    uint                 tg_size [[threads_per_threadgroup]],
    threadgroup  float  *shared [[threadgroup(0)]])
{
    // Compute partial sum of squares
    float sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = x[i];
        sum += val * val;
    }
    shared[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared[0] / float(N) + eps);

    // Normalize
    for (uint i = tid; i < N; i += tg_size) {
        out[i] = (x[i] / rms) * weight[i];
    }
}
