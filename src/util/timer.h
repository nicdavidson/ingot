#ifndef INGOT_TIMER_H
#define INGOT_TIMER_H

#include <stdint.h>

// Returns monotonic time in nanoseconds
uint64_t timer_now_ns(void);

// Returns elapsed milliseconds between two timer_now_ns() values
double timer_elapsed_ms(uint64_t start, uint64_t end);

#endif
