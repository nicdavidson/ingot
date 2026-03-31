#ifndef INGOT_TENSOR_H
#define INGOT_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    DTYPE_F32,
    DTYPE_F16,
    DTYPE_BF16,
    DTYPE_Q4_K,   // 4-bit quantized (k-quants)
    DTYPE_Q8_0,   // 8-bit quantized
} DType;

typedef struct {
    void    *data;
    int      ndim;
    int      shape[4];
    int      stride[4];
    DType    dtype;
    size_t   nbytes;
} Tensor;

// Bytes per element for a given dtype.
size_t dtype_size(DType dt);

// Human-readable dtype name.
const char *dtype_name(DType dt);

#endif
