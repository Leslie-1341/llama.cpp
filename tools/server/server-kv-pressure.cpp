#include "server-kv-pressure.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace {

constexpr uint64_t MIN_SAMPLE_INTERVAL_MS = 100;
constexpr uint64_t MIN_LOG_INTERVAL_MS    = 1000;

bool parse_env_milliseconds(const char * name, uint64_t minimum,
                            std::chrono::milliseconds & result, std::string & error) {
    const char * value = std::getenv(name);
    if (value == nullptr) {
        return true;
    }

    while (std::isspace((unsigned char) *value)) {
        ++value;
    }
    const char * end = value;
    while (*end != '\0') {
        ++end;
    }
    while (end > value && std::isspace((unsigned char) end[-1])) {
        --end;
    }
    if (end == value) {
        error = std::string(name) + " is empty";
        return false;
    }

    uint64_t parsed = 0;
    for (const char * cursor = value; cursor != end; ++cursor) {
        const unsigned char ch = (unsigned char) *cursor;
        if (!std::isdigit(ch)) {
            error = std::string(name) + " must be a complete non-negative integer";
            return false;
        }
        const uint64_t digit = ch - '0';
        if (parsed > (UINT64_MAX - digit) / 10) {
            error = std::string(name) + " is out of range";
            return false;
        }
        parsed = parsed * 10 + digit;
    }

    parsed = std::max(parsed, minimum);
    using milliseconds_rep = std::chrono::milliseconds::rep;
    const long double max_clock_ms = std::chrono::duration<long double, std::milli>(
            server_kv_pressure_runtime::clock::duration::max()).count();
    if (parsed > (uint64_t) std::numeric_limits<milliseconds_rep>::max() ||
            (long double) parsed > max_clock_ms) {
        error = std::string(name) + " is out of range";
        return false;
    }

    result = std::chrono::milliseconds(parsed);
    return true;
}

std::string event_trigger(const server_kv_pressure_event & event) {
    std::string result;
    const auto append = [&result](const char * value) {
        if (!result.empty()) {
            result += ',';
        }
        result += value;
    };

    if (event.first_sample) {
        append("first");
    }
    if (event.state_changed) {
        append("state");
    }
    if (event.source_changed) {
        append("source");
    }
    if (event.stale_changed) {
        append("stale");
    }
    if (event.periodic) {
        append("periodic");
    }
    return result;
}

server_kv_pressure_runtime::time_point saturating_deadline(
        server_kv_pressure_runtime::time_point now, std::chrono::milliseconds interval) {
    using clock = server_kv_pressure_runtime::clock;
    const long double interval_ticks = std::chrono::duration<long double, clock::period>(interval).count();
    if (interval_ticks >= (long double) std::numeric_limits<clock::rep>::max()) {
        return server_kv_pressure_runtime::time_point::max();
    }
    const auto delta = std::chrono::duration_cast<clock::duration>(interval);
    if (delta <= clock::duration::zero()) {
        return now;
    }
    if (now > server_kv_pressure_runtime::time_point::max() - delta) {
        return server_kv_pressure_runtime::time_point::max();
    }
    return now + delta;
}

} // namespace

bool server_kv_pressure_config_from_env(server_kv_pressure_config & config, std::string & error) {
    server_kv_pressure_config parsed;
    error.clear();
    if (!parse_env_milliseconds("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS",
                                MIN_SAMPLE_INTERVAL_MS, parsed.sample_interval, error)) {
        return false;
    }
    if (!parse_env_milliseconds("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS",
                                MIN_LOG_INTERVAL_MS, parsed.log_interval, error)) {
        return false;
    }
    config = parsed;
    return true;
}

bool server_kv_pressure_runtime::sample_due(time_point now) {
    if (!enabled_) {
        return false;
    }
    if (sample_count_ > 0 && now < next_sample_) {
        saturating_increment(skip_count_);
        return false;
    }
    return true;
}

server_kv_pressure_event server_kv_pressure_runtime::record_sample(
        time_point now, bool idle, const kv_pressure_telemetry & telemetry) {
    const bool first_sample = sample_count_ == 0;
    saturating_increment(sample_count_);

    server_kv_pressure_event event;
    event.telemetry = telemetry;
    event.sample_count = sample_count_;
    event.skip_count = skip_count_;
    event.idle = idle;
    event.first_sample = first_sample;
    event.state_changed  = telemetry.state  != last_state_;
    event.source_changed = telemetry.source != last_source_;
    event.stale_changed  = telemetry.stale  != last_stale_;
    event.periodic = !first_sample && now >= saturating_deadline(last_log_, config_.log_interval);

    next_sample_ = saturating_deadline(now, config_.sample_interval);
    last_state_ = telemetry.state;
    last_source_ = telemetry.source;
    last_stale_ = telemetry.stale;

    if (first_sample || event.should_log()) {
        last_log_ = now;
    }

    return event;
}

std::string server_kv_pressure_format_marker(const server_kv_pressure_event & event) {
    const kv_pressure_telemetry & telemetry = event.telemetry;

    std::ostringstream out;
    out << "kv_pressure_telemetry"
        << " state=" << kv_pressure_state_name(telemetry.state)
        << " previous_state=" << kv_pressure_state_name(telemetry.previous_state)
        << " source=" << kv_pressure_source_name(telemetry.source)
        << " sample_valid=" << (telemetry.sample_valid ? 1 : 0)
        << " stale=" << (telemetry.stale ? 1 : 0)
        << " config_valid=" << (telemetry.config_valid ? 1 : 0)
        << " rss_kb=" << telemetry.rss_kb
        << " cgroup_current_bytes=" << telemetry.cgroup_current_bytes
        << " cgroup_max_bytes=" << telemetry.cgroup_max_bytes
        << " cgroup_current_kb=" << telemetry.cgroup_current_kb
        << " cgroup_max_kb=" << telemetry.cgroup_max_kb
        << " cgroup_high_kb=" << telemetry.cgroup_high_kb
        << " psi_some_avg10=" << telemetry.psi_some_avg10
        << " psi_full_avg10=" << telemetry.psi_full_avg10
        << " sample_latency_ns=" << telemetry.sample_latency_ns
        << " sample_count=" << event.sample_count
        << " skip_count=" << event.skip_count
        << " idle=" << (event.idle ? 1 : 0)
        << " trigger=" << event_trigger(event);
    return out.str();
}
