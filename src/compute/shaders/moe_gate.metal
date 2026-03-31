#include <metal_stdlib>
using namespace metal;

// MoE Gate: compute routing weights and select top-K experts.
//
// Flow:
// 1. gate_logits = hidden_state @ gate_weight  (matmul, done separately)
// 2. gate_probs = softmax(gate_logits)         (done separately)
// 3. top_k_indices, top_k_weights = top_k(gate_probs, K)
// 4. top_k_weights = top_k_weights / sum(top_k_weights)  (renormalize)

// Top-K selection from gate probabilities.
// Single-threaded per token (K is small: 8-10).
kernel void moe_gate_topk(
    device const float  *probs       [[buffer(0)]],  // [num_experts]
    device       uint   *indices     [[buffer(1)]],  // [K] output
    device       float  *weights     [[buffer(2)]],  // [K] output
    constant     uint   &num_experts [[buffer(3)]],
    constant     uint   &K           [[buffer(4)]],
    uint                 gid         [[thread_position_in_grid]])
{
    if (gid != 0) return; // single thread

    // Simple selection sort for top-K (K ≤ 10, not worth heapifying)
    for (uint k = 0; k < K; k++) {
        float best_val = -INFINITY;
        uint  best_idx = 0;

        for (uint e = 0; e < num_experts; e++) {
            float p = probs[e];

            // Skip already selected
            bool skip = false;
            for (uint j = 0; j < k; j++) {
                if (indices[j] == e) { skip = true; break; }
            }
            if (skip) continue;

            if (p > best_val) {
                best_val = p;
                best_idx = e;
            }
        }

        indices[k] = best_idx;
        weights[k] = best_val;
    }

    // Renormalize weights
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) sum += weights[k];
    if (sum > 0.0f) {
        for (uint k = 0; k < K; k++) weights[k] /= sum;
    }
}

// Combine expert outputs: out = sum_k(weight[k] * expert_out[k])
kernel void moe_combine(
    device const float  *expert_outs [[buffer(0)]],  // [K, hidden_dim]
    device const float  *weights     [[buffer(1)]],  // [K]
    device       float  *out         [[buffer(2)]],  // [hidden_dim]
    constant     uint   &K           [[buffer(3)]],
    constant     uint   &hidden_dim  [[buffer(4)]],
    uint                 gid         [[thread_position_in_grid]])
{
    if (gid >= hidden_dim) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += weights[k] * expert_outs[k * hidden_dim + gid];
    }
    out[gid] = sum;
}
