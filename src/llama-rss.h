#pragma once

#include "llama-impl.h"

#include <cstdarg>
#include <cstdlib>
#include <cstdio>

static inline bool llama_rss_stage_log_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_RSS_STAGE_LOG");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();

    return enabled;
}

static inline void llama_rss_stage_log(const char * stage, const char * label, const char * fmt = nullptr, ...) {
    if (!llama_rss_stage_log_enabled()) {
        return;
    }

    char extra[512] = { 0 };

    if (fmt != nullptr && fmt[0] != '\0') {
        va_list args;
        va_start(args, fmt);
        vsnprintf(extra, sizeof(extra), fmt, args);
        va_end(args);
    }

    if (extra[0] != '\0') {
        LLAMA_LOG_INFO("RSS_STAGE %s %s %s\n", stage, label, extra);
    } else {
        LLAMA_LOG_INFO("RSS_STAGE %s %s\n", stage, label);
    }
}
