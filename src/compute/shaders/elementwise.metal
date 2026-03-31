#include <metal_stdlib>
using namespace metal;

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
// Used in MoE expert FFN (SwiGLU: silu(gate) * up)
kernel void silu(
    device const float *x   [[buffer(0)]],
    device       float *out [[buffer(1)]],
    constant     uint  &N   [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    float val = x[gid];
    out[gid] = val / (1.0f + exp(-val));
}

// GELU activation (approximate: tanh version)
kernel void gelu(
    device const float *x   [[buffer(0)]],
    device       float *out [[buffer(1)]],
    constant     uint  &N   [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    float val = x[gid];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (val + 0.044715f * val * val * val)));
    out[gid] = val * cdf;
}

// Element-wise addition: out = a + b
kernel void add(
    device const float *a   [[buffer(0)]],
    device const float *b   [[buffer(1)]],
    device       float *out [[buffer(2)]],
    constant     uint  &N   [[buffer(3)]],
    uint                gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    out[gid] = a[gid] + b[gid];
}

// Element-wise multiplication: out = a * b
kernel void mul(
    device const float *a   [[buffer(0)]],
    device const float *b   [[buffer(1)]],
    device       float *out [[buffer(2)]],
    constant     uint  &N   [[buffer(3)]],
    uint                gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    out[gid] = a[gid] * b[gid];
}

// Token embedding lookup: out = embedding_table[token_id]
kernel void embedding_lookup(
    device const half  *table [[buffer(0)]],
    device       float *out   [[buffer(1)]],
    constant     uint  &dim   [[buffer(2)]],
    constant     uint  &token [[buffer(3)]],
    uint                gid   [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    out[gid] = float(table[token * dim + gid]);
}
