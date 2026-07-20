#include "server-kv-pressure.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
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

// --- Dry-run policy implementation ---

namespace {

constexpr uint32_t MIN_COOLDOWN_MS = 500;

bool parse_bool_env(const char * name) {
    const char * val = std::getenv(name);
    return val && std::strcmp(val, "1") == 0;
}

bool parse_uint64_env(const char * name, uint64_t & out, std::string & error) {
    const char * val = std::getenv(name);
    if (!val) return true;
    while (std::isspace((unsigned char) *val)) ++val;
    if (*val == '\0') { error = std::string(name) + " is empty"; return false; }
    const char * end = val;
    while (*end != '\0') ++end;
    while (end > val && std::isspace((unsigned char) end[-1])) --end;

    uint64_t parsed = 0;
    for (const char * c = val; c != end; ++c) {
        if (!std::isdigit((unsigned char) *c)) {
            error = std::string(name) + " must be a non-negative integer";
            return false;
        }
        const uint64_t digit = *c - '0';
        if (parsed > (UINT64_MAX - digit) / 10) {
            error = std::string(name) + " is out of range";
            return false;
        }
        parsed = parsed * 10 + digit;
    }
    out = parsed;
    return true;
}

bool parse_uint32_env(const char * name, uint32_t minimum, uint32_t & out,
                      std::string & error) {
    uint64_t val = 0;
    if (!parse_uint64_env(name, val, error)) return false;
    if (val > UINT32_MAX) { error = std::string(name) + " is out of range"; return false; }
    out = std::max<uint32_t>((uint32_t) val, minimum);
    return true;
}

} // namespace

bool server_kv_pressure_dry_run_config_from_env(
        server_kv_pressure_dry_run_config & config, std::string & error) {
    server_kv_pressure_dry_run_config parsed;
    error.clear();

    parsed.enabled = parse_bool_env("LLAMA_KV_PRESSURE_DRY_RUN");

    if (!parse_uint64_env("LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES",
                          parsed.target_bytes, error)) {
        return false;
    }

    if (!parse_uint32_env("LLAMA_KV_PRESSURE_DRY_RUN_MAX_SCAN_BLOCKS",
                          1, parsed.max_scan_blocks, error)) {
        return false;
    }

    if (!parse_uint32_env("LLAMA_KV_PRESSURE_DRY_RUN_COOLDOWN_MS",
                          server_kv_pressure_dry_run_config::MIN_COOLDOWN_MS,
                          parsed.cooldown_ms, error)) {
        return false;
    }

    if (!parse_uint32_env("LLAMA_KV_PRESSURE_DRY_RUN_BACKOFF_MS",
                          server_kv_pressure_dry_run_config::MIN_COOLDOWN_MS,
                          parsed.backoff_ms, error)) {
        return false;
    }

    // If target_bytes is 0, treat as effectively disabled regardless of the
    // master switch — there is nothing to evaluate.
    if (parsed.enabled && parsed.target_bytes == 0) {
        parsed.enabled = false;
    }

    config = parsed;
    return true;
}

bool server_kv_pressure_runtime::dry_run_due(
        time_point now, kv_pressure_state state,
        bool stale) const {
    if (!dry_run_config_.enabled || dry_run_config_.target_bytes == 0) {
        return false;
    }
    if (stale) {
        return false;
    }
    if (state != kv_pressure_state::PRESSURE &&
            state != kv_pressure_state::CRITICAL) {
        return false;
    }

    // State-entry semantics: when the pressure state transitions INTO
    // PRESSURE or CRITICAL from a different state, we are entering a new
    // episode.  Reset the cooldown timer so the first evaluation fires
    // without waiting.
    //
    // CRITICAL entry: bypass cooldown entirely for the first evaluation
    // within the new episode.  Sustained CRITICAL calls are still rate-limited.
    //
    // PRESSURE entry: first evaluation fires immediately (cooldown was
    // reset).  Subsequent calls are rate-limited by the base cooldown.
    const bool state_entered = (state != last_dry_run_state_);
    if (state_entered) {
        last_dry_run_state_ = state;
        last_dry_run_ = time_point {};
        return true;  // Evaluate immediately on state entry.
    }

    // Within the same state episode: apply cooldown.
    if (last_dry_run_ == time_point {}) {
        return true;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_dry_run_);
    return elapsed.count() >= (int64_t) current_cooldown_ms_;
}

void server_kv_pressure_runtime::dry_run_record(bool had_shortfall, time_point now) {
    last_dry_run_ = now;
    if (had_shortfall) {
        current_cooldown_ms_ = dry_run_config_.backoff_ms;
    } else {
        current_cooldown_ms_ = dry_run_config_.cooldown_ms;
    }
}

std::string server_kv_pressure_dry_run_format_marker(
        const server_kv_pressure_dry_run_event & event) {
    const auto & r = event.result;

    const char * reason = event.skipped_reason;
    if (!reason) reason = "none";

    std::ostringstream out;
    out << "kv_pressure_dry_run"
        << " state=" << kv_pressure_state_name(event.pressure_state)
        << " source=" << kv_pressure_source_name(event.pressure_source)
        << " stale=" << (event.stale ? 1 : 0)
        << " release_enabled=" << (event.release_enabled ? 1 : 0)
        << " would_release_bytes=" << r.released_bytes
        << " would_release_blocks=" << r.released_blocks
        << " blocks_scanned=" << r.blocks_scanned
        << " blocks_skipped_owned=" << r.blocks_skipped_owned
        << " blocks_skipped_state=" << r.blocks_skipped_state
        << " shortfall_bytes=" << r.shortfall_bytes
        << " overshoot_bytes=" << r.overshoot_bytes
        << " block_scan_exhausted=" << (r.block_scan_exhausted ? 1 : 0)
        << " ownership_aborted=" << (r.ownership_aborted ? 1 : 0)
        << " target_bytes=" << event.target_bytes
        << " max_scan_blocks=" << event.max_scan_blocks
        << " skipped_reason=" << reason
        << " cooldown_ms=" << event.cooldown_ms
        << " sample_count=" << event.sample_count
        << " idle=" << (event.idle ? 1 : 0);
    return out.str();
}
