#define _POSIX_C_SOURCE 200809L

#include "util/timer.h"

#include <time.h>

uint64_t timer_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

double timer_elapsed_ms(uint64_t start, uint64_t end) {
    return (double)(end - start) / 1000000.0;
}
