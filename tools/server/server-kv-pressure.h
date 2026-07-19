#pragma once

#include "llama-kv-pressure.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>

struct server_kv_pressure_config {
    static constexpr int64_t DEFAULT_SAMPLE_INTERVAL_MS = 250;
    static constexpr int64_t DEFAULT_LOG_INTERVAL_MS    = 60000;

    std::chrono::milliseconds sample_interval { DEFAULT_SAMPLE_INTERVAL_MS };
    std::chrono::milliseconds log_interval    { DEFAULT_LOG_INTERVAL_MS };
};

// Parse server-side cadence settings. Invalid values fail closed; intervals
// below their supported minimum are clamped.
bool server_kv_pressure_config_from_env(server_kv_pressure_config & config, std::string & error);

struct server_kv_pressure_event {
    kv_pressure_telemetry telemetry;
    uint64_t sample_count = 0;
    uint64_t skip_count   = 0;
    bool idle             = false;
    bool first_sample     = false;
    bool state_changed    = false;
    bool source_changed   = false;
    bool stale_changed    = false;
    bool periodic         = false;

    bool should_log() const {
        return first_sample || state_changed || source_changed || stale_changed || periodic;
    }
};

std::string server_kv_pressure_format_marker(const server_kv_pressure_event & event);

class server_kv_pressure_runtime {
public:
    using clock      = std::chrono::steady_clock;
    using time_point = clock::time_point;

    void enable(const server_kv_pressure_config & config) {
        config_ = config;
        enabled_ = true;
        next_sample_ = time_point {};
        last_log_ = time_point {};
        last_state_ = kv_pressure_state::NORMAL;
        last_source_ = kv_pressure_source::NONE;
        last_stale_ = false;
        sample_count_ = 0;
        skip_count_ = 0;
    }

    void disable() {
        enabled_ = false;
        next_sample_ = time_point {};
        last_log_ = time_point {};
        last_state_ = kv_pressure_state::NORMAL;
        last_source_ = kv_pressure_source::NONE;
        last_stale_ = false;
        sample_count_ = 0;
        skip_count_ = 0;
    }

    bool enabled() const {
        return enabled_;
    }

    uint64_t sample_count() const {
        return sample_count_;
    }

    uint64_t skip_count() const {
        return skip_count_;
    }

    // Called at the update_slots() owner boundary. A false result guarantees
    // the caller must not enter sampler file I/O.
    bool sample_due(time_point now);

    // Publish a completed sample. 'now' is taken after sampling so a slow read
    // cannot collapse the next wall-clock interval.
    server_kv_pressure_event record_sample(
            time_point now, bool idle, const kv_pressure_telemetry & telemetry);

private:
    static void saturating_increment(uint64_t & value) {
        if (value != std::numeric_limits<uint64_t>::max()) {
            ++value;
        }
    }

    server_kv_pressure_config config_;
    bool enabled_ = false;
    time_point next_sample_ {};
    time_point last_log_ {};
    kv_pressure_state last_state_ = kv_pressure_state::NORMAL;
    kv_pressure_source last_source_ = kv_pressure_source::NONE;
    bool last_stale_ = false;
    uint64_t sample_count_ = 0;
    uint64_t skip_count_ = 0;
};
