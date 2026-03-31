#include <metal_stdlib>
using namespace metal;

// Rotary Position Embeddings (RoPE)
//
// Qwen 3.5 uses partial rotary: only partial_rotary_factor (0.25) of head_dim
// gets rotary applied. With head_dim=256, rotary_dim=64.
//
// For each pair (x[2i], x[2i+1]):
//   x_rot[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
//   x_rot[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)
// where theta = pos / (rope_theta^(2i/rotary_dim))

kernel void rope_apply(
    device       float  *x          [[buffer(0)]],
    constant     uint   &head_dim   [[buffer(1)]],
    constant     uint   &rotary_dim [[buffer(2)]],
    constant     uint   &position   [[buffer(3)]],
    constant     float  &theta_base [[buffer(4)]],
    constant     uint   &num_heads  [[buffer(5)]],
    uint                 gid        [[thread_position_in_grid]])
{
    uint head = gid / (rotary_dim / 2);
    uint pair = gid % (rotary_dim / 2);

    if (head >= num_heads) return;

    uint base_idx = head * head_dim + pair * 2;

    float freq = 1.0f / pow(theta_base, float(pair * 2) / float(rotary_dim));
    float angle = float(position) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float x0 = x[base_idx];
    float x1 = x[base_idx + 1];

    x[base_idx]     = x0 * cos_a - x1 * sin_a;
    x[base_idx + 1] = x0 * sin_a + x1 * cos_a;
}
