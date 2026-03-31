#include <metal_stdlib>
using namespace metal;

// Matrix-vector multiply: out = A * x
// A is [M, K], x is [K], out is [M]
// Used for all linear projections (QKV, output, gate, up, down)

kernel void matmul_f16(
    device const half    *A     [[buffer(0)]],
    device const half    *x     [[buffer(1)]],
    device       float   *out   [[buffer(2)]],
    constant     uint    &M     [[buffer(3)]],
    constant     uint    &K     [[buffer(4)]],
    uint                  gid   [[thread_position_in_grid]])
{
    if (gid >= M) return;

    float sum = 0.0f;
    device const half *row = A + gid * K;

    // Process in chunks of 4 for better throughput
    uint k = 0;
    for (; k + 4 <= K; k += 4) {
        sum += float(row[k])     * float(x[k]);
        sum += float(row[k + 1]) * float(x[k + 1]);
        sum += float(row[k + 2]) * float(x[k + 2]);
        sum += float(row[k + 3]) * float(x[k + 3]);
    }
    for (; k < K; k++) {
        sum += float(row[k]) * float(x[k]);
    }

    out[gid] = sum;
}

// 4-bit quantized matmul (Q4_K format)
// Each group of 32 weights is stored as:
//   - 1 half scale, 1 half zero_point
//   - 16 bytes of packed 4-bit values (32 values, 2 per byte)
//
// Layout per group: [scale:f16][zero:f16][data:16B] = 20 bytes per 32 values

struct Q4Group {
    half   scale;
    half   zero_point;
    uchar  data[16]; // 32 x 4-bit values, packed 2 per byte
};

kernel void matmul_q4(
    device const uchar  *A_raw  [[buffer(0)]],
    device const half   *x      [[buffer(1)]],
    device       float  *out    [[buffer(2)]],
    constant     uint   &M      [[buffer(3)]],
    constant     uint   &K      [[buffer(4)]],
    uint                 gid    [[thread_position_in_grid]])
{
    if (gid >= M) return;

    const uint groups_per_row = K / 32;
    const uint bytes_per_group = 20; // 2(scale) + 2(zero) + 16(data)
    device const uchar *row = A_raw + gid * groups_per_row * bytes_per_group;

    float sum = 0.0f;

    for (uint g = 0; g < groups_per_row; g++) {
        device const Q4Group *group = (device const Q4Group *)(row + g * bytes_per_group);
        float scale = float(group->scale);
        float zero  = float(group->zero_point);

        uint x_offset = g * 32;
        for (uint i = 0; i < 16; i++) {
            uchar packed = group->data[i];
            float v0 = (float(packed & 0x0F) - zero) * scale;
            float v1 = (float(packed >> 4)    - zero) * scale;
            sum += v0 * float(x[x_offset + i * 2]);
            sum += v1 * float(x[x_offset + i * 2 + 1]);
        }
    }

    out[gid] = sum;
}
