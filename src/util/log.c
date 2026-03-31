#define _POSIX_C_SOURCE 200809L

#include "util/log.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

static LogLevel g_min_level = LOG_LEVEL_INFO;

static const char *level_strings[] = {
    [LOG_LEVEL_DEBUG] = "DEBUG",
    [LOG_LEVEL_INFO]  = "INFO",
    [LOG_LEVEL_WARN]  = "WARN",
    [LOG_LEVEL_ERROR] = "ERROR",
};

static const char *level_colors[] = {
    [LOG_LEVEL_DEBUG] = "\033[36m",   // cyan
    [LOG_LEVEL_INFO]  = "\033[32m",   // green
    [LOG_LEVEL_WARN]  = "\033[33m",   // yellow
    [LOG_LEVEL_ERROR] = "\033[31m",   // red
};

void log_init(void) {
    const char *env = getenv("INGOT_LOG");
    if (env) {
        if (strcmp(env, "debug") == 0) g_min_level = LOG_LEVEL_DEBUG;
        else if (strcmp(env, "warn") == 0) g_min_level = LOG_LEVEL_WARN;
        else if (strcmp(env, "error") == 0) g_min_level = LOG_LEVEL_ERROR;
    }
}

void log_set_level(LogLevel level) {
    g_min_level = level;
}

void log_msg(LogLevel level, const char *file, int line, const char *fmt, ...) {
    if (level < g_min_level) return;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);

    // Strip path prefix — just show filename
    const char *basename = strrchr(file, '/');
    basename = basename ? basename + 1 : file;

    fprintf(stderr, "%s%02d:%02d:%02d.%03ld %-5s\033[0m %s:%d: ",
            level_colors[level],
            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000,
            level_strings[level],
            basename, line);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    fputc('\n', stderr);
}
