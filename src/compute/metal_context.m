#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "compute/metal_context.h"
#include "util/log.h"
#include "util/timer.h"

// Shader source is embedded at build time
extern const char *shader_matmul_src;
extern const char *shader_rmsnorm_src;
extern const char *shader_rope_src;
extern const char *shader_softmax_src;
extern const char *shader_elementwise_src;
extern const char *shader_attention_src;
extern const char *shader_moe_gate_src;
extern const char *shader_deltanet_src;
extern const char *shader_fused_gate_up_swiglu_src;
extern const char *shader_moe_combine_residual_src;

// Pipeline indices
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

// Compile a single shader function into a pipeline
static id<MTLComputePipelineState> compile_function(id<MTLDevice> device,
                                                     const char *source,
                                                     const char *func_name) {
    NSError *error = nil;
    NSString *src = [NSString stringWithUTF8String:source];
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;

    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&error];
    if (!lib) {
        LOG_ERROR("metal: shader compile failed for %s: %s",
                  func_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    NSString *name = [NSString stringWithUTF8String:func_name];
    id<MTLFunction> func = [lib newFunctionWithName:name];
    if (!func) {
        LOG_ERROR("metal: function '%s' not found in shader", func_name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        LOG_ERROR("metal: pipeline creation failed for %s: %s",
                  func_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    return pipeline;
}

MetalContext *metal_init(void) {
    uint64_t t0 = timer_now_ns();

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        LOG_ERROR("metal: no Metal device available");
        return NULL;
    }

    LOG_INFO("metal: device = %s", [[device name] UTF8String]);
    LOG_INFO("metal: unified memory = %s",
             [device hasUnifiedMemory] ? "yes" : "no");
    LOG_INFO("metal: max buffer size = %zu MB",
             [device maxBufferLength] / (1024 * 1024));

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        LOG_ERROR("metal: failed to create command queue");
        return NULL;
    }

    MetalContext *ctx = calloc(1, sizeof(MetalContext));
    ctx->device = device;
    ctx->queue = queue;

    // Compile all shader pipelines
    struct {
        const char *source;
        const char *func_name;
        int         pipe_idx;
    } shaders[] = {
        { shader_matmul_src,      "matmul_q4_fma",       PIPE_MATMUL_Q4_FMA },
        { shader_matmul_src,      "matmul_bf16",         PIPE_MATMUL_BF16 },
        { shader_rmsnorm_src,     "rmsnorm",             PIPE_RMSNORM },
        { shader_rope_src,        "rope_apply",          PIPE_ROPE },
        { shader_softmax_src,     "softmax",             PIPE_SOFTMAX },
        { shader_elementwise_src, "silu",                PIPE_SILU },
        { shader_elementwise_src, "gelu",                PIPE_GELU },
        { shader_elementwise_src, "add",                 PIPE_ADD },
        { shader_elementwise_src, "mul",                 PIPE_MUL },
        { shader_elementwise_src, "embedding_lookup",    PIPE_EMBEDDING },
        { shader_attention_src,   "attention_scores",    PIPE_ATTENTION_SCORES },
        { shader_attention_src,   "attention_values",    PIPE_ATTENTION_VALUES },
        { shader_moe_gate_src,    "moe_gate_topk",       PIPE_MOE_GATE_TOPK },
        { shader_moe_gate_src,    "moe_combine",         PIPE_MOE_COMBINE },
        { shader_moe_combine_residual_src, "moe_combine_residual", PIPE_MOE_COMBINE_RESIDUAL },
        { shader_fused_gate_up_swiglu_src, "fused_gate_up_swiglu", PIPE_FUSED_GATE_UP_SWIGLU },
        { shader_deltanet_src,    "deltanet_recurrent",  PIPE_DELTANET_RECURRENT },
        { shader_deltanet_src,    "deltanet_gate",       PIPE_DELTANET_GATE },
    };

    int compiled = 0;
    for (int i = 0; i < (int)(sizeof(shaders) / sizeof(shaders[0])); i++) {
        if (!shaders[i].source) continue;
        ctx->pipelines[shaders[i].pipe_idx] =
            compile_function(device, shaders[i].source, shaders[i].func_name);
        if (ctx->pipelines[shaders[i].pipe_idx]) compiled++;
    }

    uint64_t t1 = timer_now_ns();
    LOG_INFO("metal: compiled %d/%d pipelines in %.1f ms",
             compiled, PIPE_COUNT, timer_elapsed_ms(t0, t1));

    return ctx;
}

void metal_free(MetalContext *ctx) {
    if (!ctx) return;
    // ARC handles release of ObjC objects
    free(ctx);
}

void *metal_wrap_buffer(MetalContext *ctx, void *data, size_t size) {
    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:data
                                                       length:size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!buf) {
        LOG_ERROR("metal: failed to wrap buffer (%zu bytes)", size);
        return NULL;
    }
    return (__bridge_retained void *)buf;
}

void *metal_alloc_buffer(MetalContext *ctx, size_t size) {
    id<MTLBuffer> buf = [ctx->device newBufferWithLength:size
                                                 options:MTLResourceStorageModeShared];
    if (!buf) {
        LOG_ERROR("metal: failed to allocate buffer (%zu bytes)", size);
        return NULL;
    }
    return (__bridge_retained void *)buf;
}

void metal_free_buffer(void *buffer) {
    if (!buffer) return;
    (void)(__bridge_transfer id<MTLBuffer>)buffer; // ARC releases
}

void metal_sync(MetalContext *ctx) {
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
}
