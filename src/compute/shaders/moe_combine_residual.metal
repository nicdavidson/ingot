#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Fused MoE Combine + Shared Expert + Residual
//
// In one kernel pass:
//   out[i] = residual[i]
//          + shared_gate * shared_expert[i]
//          + sum_k(weight[k] * expert_out[k][i])
//
// Saves 3 separate memory round-trips (combine, add shared, add residual).
// Each thread handles one element of the hidden dimension.
// ============================================================================

kernel void moe_combine_residual(
    device const float  *residual       [[buffer(0)]],  // [hidden_dim]
    device const float  *shared_expert  [[buffer(1)]],  // [hidden_dim]
    device const float  *expert_outs    [[buffer(2)]],  // [K, hidden_dim]
    device const float  *expert_weights [[buffer(3)]],  // [K]
    device       float  *out            [[buffer(4)]],  // [hidden_dim]
    constant     float  &shared_gate    [[buffer(5)]],  // scalar sigmoid gate
    constant     uint   &K              [[buffer(6)]],  // num active experts
    constant     uint   &hidden_dim     [[buffer(7)]],
    uint                 gid            [[thread_position_in_grid]])
{
    if (gid >= hidden_dim) return;

    float val = residual[gid] + shared_gate * shared_expert[gid];

    for (uint k = 0; k < K; k++) {
        val += expert_weights[k] * expert_outs[k * hidden_dim + gid];
    }

    out[gid] = val;
}
