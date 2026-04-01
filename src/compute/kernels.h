#ifndef INGOT_KERNELS_H
#define INGOT_KERNELS_H

#include "compute/metal_context.h"

#include <stdint.h>

// All kernel functions take a MetalContext and Metal buffer handles.
// Buffers are created via metal_wrap_buffer() or metal_alloc_buffer().

// Matrix-vector multiply: out[M] = A[M,K] @ x[K]
// Legacy wrappers (redirect to optimized versions)
void kernel_matmul_f16(MetalContext *ctx,
                       void *A, void *x, void *out,
                       uint32_t M, uint32_t K);

void kernel_matmul_q4(MetalContext *ctx,
                      void *A, void *x, void *out,
                      uint32_t M, uint32_t K);

// Optimized Q4 matmul with separate weight/scale/bias and FMA dequant
void kernel_matmul_q4_fma(MetalContext *ctx,
                          void *weights, void *scales, void *biases,
                          void *x, void *out,
                          uint32_t M, uint32_t K, uint32_t group_size);

// Q4 matmul using offsets within a shared buffer (for mmap'd weights)
void kernel_matmul_q4_fma_offsets(MetalContext *ctx,
                                   void *shared_buf,
                                   size_t w_offset, size_t s_offset, size_t b_offset,
                                   void *x, void *out,
                                   uint32_t M, uint32_t K, uint32_t group_size);

// BF16 matmul with threadgroup optimization
void kernel_matmul_bf16(MetalContext *ctx,
                        void *A, void *x, void *out,
                        uint32_t M, uint32_t K);

// RMS normalization: out[N] = rmsnorm(x[N], weight[N], eps)
void kernel_rmsnorm(MetalContext *ctx,
                    void *x, void *weight, void *out,
                    uint32_t N, float eps);

// Rotary position embedding (in-place)
void kernel_rope(MetalContext *ctx,
                 void *x,
                 uint32_t head_dim, uint32_t rotary_dim,
                 uint32_t position, float theta_base,
                 uint32_t num_heads);

// Softmax: out[N] = softmax(x[N])
void kernel_softmax(MetalContext *ctx,
                    void *x, void *out,
                    uint32_t N);

// Element-wise ops
void kernel_silu(MetalContext *ctx, void *x, void *out, uint32_t N);
void kernel_gelu(MetalContext *ctx, void *x, void *out, uint32_t N);
void kernel_add(MetalContext *ctx, void *a, void *b, void *out, uint32_t N);
void kernel_mul(MetalContext *ctx, void *a, void *b, void *out, uint32_t N);

// Embedding lookup
void kernel_embedding(MetalContext *ctx,
                      void *table, void *out,
                      uint32_t dim, uint32_t token_id);

// Attention
void kernel_attention_scores(MetalContext *ctx,
                             void *Q, void *K, void *scores,
                             uint32_t num_heads, uint32_t num_kv_heads,
                             uint32_t head_dim, uint32_t seq_len,
                             float scale);

void kernel_attention_values(MetalContext *ctx,
                             void *attn, void *V, void *out,
                             uint32_t num_heads, uint32_t num_kv_heads,
                             uint32_t head_dim, uint32_t seq_len);

// MoE gate
void kernel_moe_gate_topk(MetalContext *ctx,
                          void *probs, void *indices, void *weights,
                          uint32_t num_experts, uint32_t K);

void kernel_moe_combine(MetalContext *ctx,
                        void *expert_outs, void *weights, void *out,
                        uint32_t K, uint32_t hidden_dim);

// Fused gate+up projection + SwiGLU activation (single kernel, reads input once)
void kernel_fused_gate_up_swiglu(MetalContext *ctx,
                                  void *gate_w, void *gate_s, void *gate_b,
                                  void *up_w, void *up_s, void *up_b,
                                  void *x, void *out,
                                  uint32_t moe_dim, uint32_t K,
                                  uint32_t group_size);

// Fused MoE combine + shared expert + residual add
void kernel_moe_combine_residual(MetalContext *ctx,
                                  void *residual, void *shared_expert,
                                  void *expert_outs, void *expert_weights,
                                  void *out,
                                  float shared_gate,
                                  uint32_t K, uint32_t hidden_dim);

// DeltaNet
void kernel_deltanet_recurrent(MetalContext *ctx,
                               void *S, void *k, void *v,
                               float beta,
                               uint32_t key_dim, uint32_t value_dim);

void kernel_deltanet_gate(MetalContext *ctx,
                          void *S, void *q, void *out,
                          uint32_t key_dim, uint32_t value_dim);

#endif
