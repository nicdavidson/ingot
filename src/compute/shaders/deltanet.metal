#include <metal_stdlib>
using namespace metal;

// Gated DeltaNet (Linear Attention) — 75% of Qwen 3.5 layers.
//
// DeltaNet maintains a recurrent state matrix S ∈ R^(key_dim × value_dim)
// instead of a KV cache. For each token:
//
//   k = linear_key_proj(x)              [key_dim]
//   v = linear_value_proj(x)            [value_dim]
//   beta = sigmoid(gate_proj(x))        [1] — forget gate
//   q = linear_query_proj(x)            [key_dim]
//
//   S = beta * S + (1 - beta) * outer(k, v)   — state update (delta rule)
//   output = S^T @ q                           — output query
//
// This is O(1) per token (no growing KV cache!) but requires careful
// numerical precision in the state matrix accumulation.

// Recurrent state update: S = beta * S + (1 - beta) * outer(k, v)
kernel void deltanet_recurrent(
    device       float  *S          [[buffer(0)]],  // [key_dim, value_dim]
    device const float  *k          [[buffer(1)]],  // [key_dim]
    device const float  *v          [[buffer(2)]],  // [value_dim]
    constant     float  &beta       [[buffer(3)]],  // forget gate
    constant     uint   &key_dim    [[buffer(4)]],
    constant     uint   &value_dim  [[buffer(5)]],
    uint2                gid        [[thread_position_in_grid]])
{
    uint ki = gid.x;
    uint vi = gid.y;

    if (ki >= key_dim || vi >= value_dim) return;

    uint idx = ki * value_dim + vi;
    float s = S[idx];
    float update = k[ki] * v[vi];

    S[idx] = beta * s + (1.0f - beta) * update;
}

// Query the recurrent state: output = S^T @ q
kernel void deltanet_gate(
    device const float  *S          [[buffer(0)]],  // [key_dim, value_dim]
    device const float  *q          [[buffer(1)]],  // [key_dim]
    device       float  *out        [[buffer(2)]],  // [value_dim]
    constant     uint   &key_dim    [[buffer(3)]],
    constant     uint   &value_dim  [[buffer(4)]],
    uint                 gid        [[thread_position_in_grid]])
{
    if (gid >= value_dim) return;

    float sum = 0.0f;
    for (uint ki = 0; ki < key_dim; ki++) {
        sum += S[ki * value_dim + gid] * q[ki];
    }
    out[gid] = sum;
}
