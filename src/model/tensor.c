#include "model/tensor.h"

size_t dtype_size(DType dt) {
    switch (dt) {
        case DTYPE_F32:  return 4;
        case DTYPE_F16:  return 2;
        case DTYPE_BF16: return 2;
        case DTYPE_Q4_K: return 0; // variable, use group size
        case DTYPE_Q8_0: return 1;
    }
    return 0;
}

const char *dtype_name(DType dt) {
    switch (dt) {
        case DTYPE_F32:  return "f32";
        case DTYPE_F16:  return "f16";
        case DTYPE_BF16: return "bf16";
        case DTYPE_Q4_K: return "q4_k";
        case DTYPE_Q8_0: return "q8_0";
    }
    return "unknown";
}
