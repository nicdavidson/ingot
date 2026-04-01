#import <Metal/Metal.h>

#include "compute/kernels.h"
#include "util/log.h"

// Access internal pipelines from MetalContext
// (This file is part of the compute module and shares the struct definition)
enum {
    PIPE_MATMUL_F16,
    PIPE_MATMUL_Q4,
    PIPE_MATMUL_Q4_FMA,
    PIPE_MATMUL_BF16,
    PIPE_RMSNORM,
    PIPE_ROPE,
    PIPE_SOFTMAX,
    PIPE_SILU,
    PIPE_GELU,
    PIPE_ADD,
    PIPE_MUL,
    PIPE_EMBEDDING,
    PIPE_ATTENTION_SCORES,
    PIPE_ATTENTION_VALUES,
    PIPE_MOE_GATE_TOPK,
    PIPE_MOE_COMBINE,
    PIPE_MOE_COMBINE_RESIDUAL,
    PIPE_FUSED_GATE_UP_SWIGLU,
    PIPE_DELTANET_RECURRENT,
    PIPE_DELTANET_GATE,
    PIPE_COUNT,
};

struct MetalContext {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pipelines[PIPE_COUNT];
};

// Helper: dispatch a compute command
static void dispatch_1d(MetalContext *ctx, int pipe_idx,
                        void *bufs[], uint32_t params[], int nbufs, int nparams,
                        uint32_t grid_size) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[pipe_idx];
    if (!pipeline) {
        LOG_ERROR("kernel: pipeline %d not compiled", pipe_idx);
        return;
    }

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    [enc setComputePipelineState:pipeline];

    for (int i = 0; i < nbufs; i++) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)bufs[i];
        [enc setBuffer:buf offset:0 atIndex:(NSUInteger)i];
    }

    for (int i = 0; i < nparams; i++) {
        [enc setBytes:&params[i] length:sizeof(uint32_t) atIndex:(NSUInteger)(nbufs + i)];
    }

    NSUInteger tg_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)grid_size);
    [enc dispatchThreads:MTLSizeMake(grid_size, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

__attribute__((unused))
static void dispatch_1d_with_floats(MetalContext *ctx, int pipe_idx,
                                     void *bufs[], int nbufs,
                                     void *params_data, size_t params_size,
                                     int params_idx,
                                     uint32_t grid_size) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[pipe_idx];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    for (int i = 0; i < nbufs; i++) {
        [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:0 atIndex:(NSUInteger)i];
    }

    [enc setBytes:params_data length:params_size atIndex:(NSUInteger)params_idx];

    NSUInteger tg_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)grid_size);
    [enc dispatchThreads:MTLSizeMake(grid_size, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// --- Public kernel functions ---

void kernel_matmul_f16(MetalContext *ctx,
                       void *A, void *x, void *out,
                       uint32_t M, uint32_t K) {
    // Use BF16 kernel with threadgroup dispatch
    kernel_matmul_bf16(ctx, A, x, out, M, K);
}

void kernel_matmul_q4(MetalContext *ctx,
                      void *A, void *x, void *out,
                      uint32_t M, uint32_t K) {
    // Redirect to FMA kernel with default group_size=64
    kernel_matmul_q4_fma(ctx, A, NULL, NULL, x, out, M, K, 64);
}

// Optimized Q4 matmul: 256-thread threadgroups, shared memory, FMA dequant
void kernel_matmul_q4_fma(MetalContext *ctx,
                          void *weights, void *scales, void *biases,
                          void *x, void *out,
                          uint32_t M, uint32_t K, uint32_t group_size) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_MATMUL_Q4_FMA];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)weights offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)scales  offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)biases  offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)x       offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out     offset:0 atIndex:4];
    [enc setBytes:&M          length:sizeof(M)          atIndex:5];
    [enc setBytes:&K          length:sizeof(K)          atIndex:6];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:7];

    // One threadgroup per output row, 256 threads each
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// Q4 matmul using offsets into a shared buffer (for mmap'd weights)
void kernel_matmul_q4_fma_offsets(MetalContext *ctx,
                                   void *shared_buf,
                                   size_t w_offset, size_t s_offset, size_t b_offset,
                                   void *x, void *out,
                                   uint32_t M, uint32_t K, uint32_t group_size) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_MATMUL_Q4_FMA];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)shared_buf;
    [enc setBuffer:buf offset:(NSUInteger)w_offset atIndex:0];
    [enc setBuffer:buf offset:(NSUInteger)s_offset atIndex:1];
    [enc setBuffer:buf offset:(NSUInteger)b_offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)x   offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:4];
    [enc setBytes:&M          length:sizeof(M)          atIndex:5];
    [enc setBytes:&K          length:sizeof(K)          atIndex:6];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:7];

    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// --- Batched command buffer API ---
// Encodes multiple matmuls into a single command buffer, then commits once.
// Eliminates per-dispatch commit+waitUntilCompleted overhead.

typedef struct {
    MetalContext *ctx;
    id<MTLCommandBuffer> cb;
} BatchContext;

void *kernel_begin_batch(MetalContext *ctx) {
    BatchContext *batch = calloc(1, sizeof(BatchContext));
    batch->ctx = ctx;
    batch->cb = [ctx->queue commandBuffer];
    return batch;
}

void kernel_batch_q4_fma_offsets(void *batch_ptr,
                                  void *weight_buf,
                                  size_t w_offset, size_t s_offset, size_t b_offset,
                                  void *x, void *out,
                                  uint32_t M, uint32_t K, uint32_t group_size) {
    BatchContext *batch = batch_ptr;
    id<MTLComputePipelineState> pipeline = batch->ctx->pipelines[PIPE_MATMUL_Q4_FMA];
    if (!pipeline) return;

    id<MTLComputeCommandEncoder> enc = [batch->cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)weight_buf;
    [enc setBuffer:wbuf offset:(NSUInteger)w_offset atIndex:0];
    [enc setBuffer:wbuf offset:(NSUInteger)s_offset atIndex:1];
    [enc setBuffer:wbuf offset:(NSUInteger)b_offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)x   offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:4];
    [enc setBytes:&M          length:sizeof(M)          atIndex:5];
    [enc setBytes:&K          length:sizeof(K)          atIndex:6];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:7];

    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
}

void kernel_end_batch(void *batch_ptr) {
    BatchContext *batch = batch_ptr;
    [batch->cb commit];
    [batch->cb waitUntilCompleted];
    free(batch);
}

// BF16 matmul: 256-thread threadgroups, shared memory
void kernel_matmul_bf16(MetalContext *ctx,
                        void *A, void *x, void *out,
                        uint32_t M, uint32_t K) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_MATMUL_BF16];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)A   offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)x   offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(M) atIndex:3];
    [enc setBytes:&K length:sizeof(K) atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// Fused gate+up+SwiGLU: single kernel for MoE expert FFN first half
void kernel_fused_gate_up_swiglu(MetalContext *ctx,
                                  void *gate_w, void *gate_s, void *gate_b,
                                  void *up_w, void *up_s, void *up_b,
                                  void *x, void *out,
                                  uint32_t moe_dim, uint32_t K,
                                  uint32_t group_size) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_FUSED_GATE_UP_SWIGLU];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)gate_w offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate_s offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate_b offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)up_w   offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)up_s   offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)up_b   offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)x      offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)out    offset:0 atIndex:7];
    [enc setBytes:&moe_dim    length:sizeof(moe_dim)    atIndex:8];
    [enc setBytes:&K          length:sizeof(K)          atIndex:9];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:10];

    [enc dispatchThreadgroups:MTLSizeMake(moe_dim, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// Fused gate+up+SwiGLU with offsets into a shared buffer (for mmap'd expert weights)
void kernel_fused_gate_up_swiglu_offsets(MetalContext *ctx,
                                          void *weight_buf,
                                          size_t gw_off, size_t gs_off, size_t gb_off,
                                          size_t uw_off, size_t us_off, size_t ub_off,
                                          void *x, void *out,
                                          uint32_t moe_dim, uint32_t K,
                                          uint32_t group_size) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_FUSED_GATE_UP_SWIGLU];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)weight_buf;
    [enc setBuffer:wbuf offset:(NSUInteger)gw_off atIndex:0];
    [enc setBuffer:wbuf offset:(NSUInteger)gs_off atIndex:1];
    [enc setBuffer:wbuf offset:(NSUInteger)gb_off atIndex:2];
    [enc setBuffer:wbuf offset:(NSUInteger)uw_off atIndex:3];
    [enc setBuffer:wbuf offset:(NSUInteger)us_off atIndex:4];
    [enc setBuffer:wbuf offset:(NSUInteger)ub_off atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)x   offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:7];
    [enc setBytes:&moe_dim    length:sizeof(moe_dim)    atIndex:8];
    [enc setBytes:&K          length:sizeof(K)          atIndex:9];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:10];

    [enc dispatchThreadgroups:MTLSizeMake(moe_dim, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

// Batch-compatible fused gate+up+SwiGLU with offsets
void kernel_batch_fused_gate_up_swiglu_offsets(void *batch_ptr,
                                                void *weight_buf,
                                                size_t gw_off, size_t gs_off, size_t gb_off,
                                                size_t uw_off, size_t us_off, size_t ub_off,
                                                void *x, void *out,
                                                uint32_t moe_dim, uint32_t K,
                                                uint32_t group_size) {
    BatchContext *batch = batch_ptr;
    id<MTLComputePipelineState> pipeline = batch->ctx->pipelines[PIPE_FUSED_GATE_UP_SWIGLU];
    if (!pipeline) return;

    id<MTLComputeCommandEncoder> enc = [batch->cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    id<MTLBuffer> wbuf = (__bridge id<MTLBuffer>)weight_buf;
    [enc setBuffer:wbuf offset:(NSUInteger)gw_off atIndex:0];
    [enc setBuffer:wbuf offset:(NSUInteger)gs_off atIndex:1];
    [enc setBuffer:wbuf offset:(NSUInteger)gb_off atIndex:2];
    [enc setBuffer:wbuf offset:(NSUInteger)uw_off atIndex:3];
    [enc setBuffer:wbuf offset:(NSUInteger)us_off atIndex:4];
    [enc setBuffer:wbuf offset:(NSUInteger)ub_off atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)x   offset:0 atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:7];
    [enc setBytes:&moe_dim    length:sizeof(moe_dim)    atIndex:8];
    [enc setBytes:&K          length:sizeof(K)          atIndex:9];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:10];

    [enc dispatchThreadgroups:MTLSizeMake(moe_dim, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
}

// Fused MoE combine + shared expert + residual
void kernel_moe_combine_residual(MetalContext *ctx,
                                  void *residual, void *shared_expert,
                                  void *expert_outs, void *expert_weights,
                                  void *out,
                                  float shared_gate,
                                  uint32_t K, uint32_t hidden_dim) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_MOE_COMBINE_RESIDUAL];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)residual       offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)shared_expert  offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)expert_outs    offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)expert_weights offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out            offset:0 atIndex:4];
    [enc setBytes:&shared_gate length:sizeof(shared_gate) atIndex:5];
    [enc setBytes:&K           length:sizeof(K)           atIndex:6];
    [enc setBytes:&hidden_dim  length:sizeof(hidden_dim)  atIndex:7];

    NSUInteger tg_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)hidden_dim);
    [enc dispatchThreads:MTLSizeMake(hidden_dim, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void kernel_rmsnorm(MetalContext *ctx,
                    void *x, void *weight, void *out,
                    uint32_t N, float eps) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_RMSNORM];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)x      offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight  offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)out     offset:0 atIndex:2];
    [enc setBytes:&N   length:sizeof(N)   atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];

    // RMSNorm uses a single threadgroup with shared memory
    NSUInteger tg_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256u);
    [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void kernel_rope(MetalContext *ctx,
                 void *x,
                 uint32_t head_dim, uint32_t rotary_dim,
                 uint32_t position, float theta_base,
                 uint32_t num_heads) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_ROPE];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
    [enc setBytes:&head_dim   length:sizeof(head_dim)   atIndex:1];
    [enc setBytes:&rotary_dim length:sizeof(rotary_dim) atIndex:2];
    [enc setBytes:&position   length:sizeof(position)   atIndex:3];
    [enc setBytes:&theta_base length:sizeof(theta_base) atIndex:4];
    [enc setBytes:&num_heads  length:sizeof(num_heads)  atIndex:5];

    uint32_t grid = num_heads * (rotary_dim / 2);
    NSUInteger tg_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)grid);
    [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void kernel_softmax(MetalContext *ctx, void *x, void *out, uint32_t N) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_SOFTMAX];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)x   offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
    [enc setBytes:&N length:sizeof(N) atIndex:2];

    NSUInteger tg_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256u);
    [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void kernel_silu(MetalContext *ctx, void *x, void *out, uint32_t N) {
    void *bufs[] = { x, out };
    uint32_t params[] = { N };
    dispatch_1d(ctx, PIPE_SILU, bufs, params, 2, 1, N);
}

void kernel_gelu(MetalContext *ctx, void *x, void *out, uint32_t N) {
    void *bufs[] = { x, out };
    uint32_t params[] = { N };
    dispatch_1d(ctx, PIPE_GELU, bufs, params, 2, 1, N);
}

void kernel_add(MetalContext *ctx, void *a, void *b, void *out, uint32_t N) {
    void *bufs[] = { a, b, out };
    uint32_t params[] = { N };
    dispatch_1d(ctx, PIPE_ADD, bufs, params, 3, 1, N);
}

void kernel_mul(MetalContext *ctx, void *a, void *b, void *out, uint32_t N) {
    void *bufs[] = { a, b, out };
    uint32_t params[] = { N };
    dispatch_1d(ctx, PIPE_MUL, bufs, params, 3, 1, N);
}

void kernel_embedding(MetalContext *ctx,
                      void *table, void *out,
                      uint32_t dim, uint32_t token_id) {
    void *bufs[] = { table, out };
    uint32_t params[] = { dim, token_id };
    dispatch_1d(ctx, PIPE_EMBEDDING, bufs, params, 2, 2, dim);
}

void kernel_attention_scores(MetalContext *ctx,
                             void *Q, void *K, void *scores,
                             uint32_t num_heads, uint32_t num_kv_heads,
                             uint32_t head_dim, uint32_t seq_len,
                             float scale) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_ATTENTION_SCORES];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)Q      offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)K      offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)scores offset:0 atIndex:2];
    [enc setBytes:&num_heads    length:sizeof(num_heads)    atIndex:3];
    [enc setBytes:&num_kv_heads length:sizeof(num_kv_heads) atIndex:4];
    [enc setBytes:&head_dim     length:sizeof(head_dim)     atIndex:5];
    [enc setBytes:&seq_len      length:sizeof(seq_len)      atIndex:6];
    [enc setBytes:&scale        length:sizeof(scale)        atIndex:7];

    [enc dispatchThreads:MTLSizeMake(num_heads, seq_len, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(num_heads, 32), MIN(seq_len, 32), 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void kernel_attention_values(MetalContext *ctx,
                             void *attn, void *V, void *out,
                             uint32_t num_heads, uint32_t num_kv_heads,
                             uint32_t head_dim, uint32_t seq_len) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_ATTENTION_VALUES];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)attn offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)V    offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)out  offset:0 atIndex:2];
    [enc setBytes:&num_heads    length:sizeof(num_heads)    atIndex:3];
    [enc setBytes:&num_kv_heads length:sizeof(num_kv_heads) atIndex:4];
    [enc setBytes:&head_dim     length:sizeof(head_dim)     atIndex:5];
    [enc setBytes:&seq_len      length:sizeof(seq_len)      atIndex:6];

    [enc dispatchThreads:MTLSizeMake(num_heads, head_dim, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(num_heads, 32), MIN(head_dim, 32), 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void kernel_moe_gate_topk(MetalContext *ctx,
                          void *probs, void *indices, void *weights,
                          uint32_t num_experts, uint32_t K) {
    void *bufs[] = { probs, indices, weights };
    uint32_t params[] = { num_experts, K };
    dispatch_1d(ctx, PIPE_MOE_GATE_TOPK, bufs, params, 3, 2, 1);
}

void kernel_moe_combine(MetalContext *ctx,
                        void *expert_outs, void *weights, void *out,
                        uint32_t K, uint32_t hidden_dim) {
    void *bufs[] = { expert_outs, weights, out };
    uint32_t params[] = { K, hidden_dim };
    dispatch_1d(ctx, PIPE_MOE_COMBINE, bufs, params, 3, 2, hidden_dim);
}

void kernel_deltanet_recurrent(MetalContext *ctx,
                               void *S, void *k, void *v,
                               float beta,
                               uint32_t key_dim, uint32_t value_dim) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_DELTANET_RECURRENT];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)S offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)k offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)v offset:0 atIndex:2];
    [enc setBytes:&beta      length:sizeof(beta)      atIndex:3];
    [enc setBytes:&key_dim   length:sizeof(key_dim)   atIndex:4];
    [enc setBytes:&value_dim length:sizeof(value_dim) atIndex:5];

    [enc dispatchThreads:MTLSizeMake(key_dim, value_dim, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(key_dim, 32), MIN(value_dim, 32), 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void kernel_deltanet_gate(MetalContext *ctx,
                          void *S, void *q, void *out,
                          uint32_t key_dim, uint32_t value_dim) {
    id<MTLComputePipelineState> pipeline = ctx->pipelines[PIPE_DELTANET_GATE];
    if (!pipeline) return;

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    [enc setBuffer:(__bridge id<MTLBuffer>)S   offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)q   offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
    [enc setBytes:&key_dim   length:sizeof(key_dim)   atIndex:3];
    [enc setBytes:&value_dim length:sizeof(value_dim) atIndex:4];

    NSUInteger tg_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)value_dim);
    [enc dispatchThreads:MTLSizeMake(value_dim, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}
