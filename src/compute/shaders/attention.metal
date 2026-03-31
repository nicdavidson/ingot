#include <metal_stdlib>
using namespace metal;

// Sliding Window Attention (SWA) — used in 25% of Qwen 3.5 layers.
// GQA with 32 Q heads and 2 KV heads (16:1 ratio).
//
// For single-token generation (decode step):
// - Q is [num_heads, head_dim] (current token only)
// - K, V are [seq_len, num_kv_heads, head_dim] (from KV cache)
// - Output is [num_heads, head_dim]

// Compute attention scores: scores[h, t] = Q[h] · K[t, kv_group]
kernel void attention_scores(
    device const float  *Q          [[buffer(0)]],
    device const float  *K          [[buffer(1)]],
    device       float  *scores     [[buffer(2)]],
    constant     uint   &num_heads  [[buffer(3)]],
    constant     uint   &num_kv_heads [[buffer(4)]],
    constant     uint   &head_dim   [[buffer(5)]],
    constant     uint   &seq_len    [[buffer(6)]],
    constant     float  &scale      [[buffer(7)]],
    uint2                gid        [[thread_position_in_grid]])
{
    uint h = gid.x; // query head
    uint t = gid.y; // time step

    if (h >= num_heads || t >= seq_len) return;

    uint kv_head = h / (num_heads / num_kv_heads); // GQA grouping

    device const float *q = Q + h * head_dim;
    device const float *k = K + t * num_kv_heads * head_dim + kv_head * head_dim;

    float dot = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        dot += q[i] * k[i];
    }

    scores[h * seq_len + t] = dot * scale;
}

// Compute weighted values: out[h] = sum_t(attn[h,t] * V[t, kv_group])
kernel void attention_values(
    device const float  *attn       [[buffer(0)]],  // [num_heads, seq_len] (after softmax)
    device const float  *V          [[buffer(1)]],
    device       float  *out        [[buffer(2)]],
    constant     uint   &num_heads  [[buffer(3)]],
    constant     uint   &num_kv_heads [[buffer(4)]],
    constant     uint   &head_dim   [[buffer(5)]],
    constant     uint   &seq_len    [[buffer(6)]],
    uint2                gid        [[thread_position_in_grid]])
{
    uint h = gid.x; // head
    uint d = gid.y; // dimension within head

    if (h >= num_heads || d >= head_dim) return;

    uint kv_head = h / (num_heads / num_kv_heads);

    float sum = 0.0f;
    for (uint t = 0; t < seq_len; t++) {
        float a = attn[h * seq_len + t];
        float v = V[t * num_kv_heads * head_dim + kv_head * head_dim + d];
        sum += a * v;
    }

    out[h * head_dim + d] = sum;
}
