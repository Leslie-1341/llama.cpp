#pragma once

#include "llama-kv-cache-release.h"
#include "llama-kv-pressure.h"
#include "llama-memory.h"

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

// --- Dry-run bounded release types (must precede runtime class) ---

struct server_kv_pressure_dry_run_config {
    static constexpr uint32_t DEFAULT_MAX_SCAN_BLOCKS = 64;
    static constexpr uint32_t DEFAULT_COOLDOWN_MS     = 2000;
    static constexpr uint32_t DEFAULT_BACKOFF_MS      = 10000;
    static constexpr uint32_t MIN_COOLDOWN_MS         = 500;

    bool     enabled         = false;    // LLAMA_KV_PRESSURE_DRY_RUN=1
    uint64_t target_bytes    = 0;        // LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES
    uint32_t max_scan_blocks = DEFAULT_MAX_SCAN_BLOCKS;
    uint32_t cooldown_ms     = DEFAULT_COOLDOWN_MS;
    uint32_t backoff_ms      = DEFAULT_BACKOFF_MS;
};

struct server_kv_pressure_dry_run_event {
    llama_kv_bounded_release_result result;
    kv_pressure_state pressure_state   = kv_pressure_state::NORMAL;
    kv_pressure_source pressure_source = kv_pressure_source::NONE;
    bool    stale           = false;
    bool    idle             = false;
    bool    release_enabled  = false;   // LLAMA_KV_PAGED_RELEASE=1 is active
    uint64_t sample_count    = 0;
    uint64_t target_bytes    = 0;
    uint32_t max_scan_blocks = 0;
    uint32_t cooldown_ms     = 0;
    const char * skipped_reason = nullptr;  // nullptr = not skipped
};

bool server_kv_pressure_dry_run_config_from_env(
        server_kv_pressure_dry_run_config & config, std::string & error);

std::string server_kv_pressure_dry_run_format_marker(
        const server_kv_pressure_dry_run_event & event);

// --- Bounded destructive release types (server pressure path) ---

struct server_kv_pressure_bounded_release_config {
    static constexpr uint32_t DEFAULT_MAX_SCAN_BLOCKS = 64;
    static constexpr uint32_t DEFAULT_COOLDOWN_MS     = 2000;
    static constexpr uint32_t DEFAULT_BACKOFF_MS      = 10000;
    static constexpr uint32_t MIN_COOLDOWN_MS         = 500;

    bool     enabled         = false;   // LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1
    uint64_t target_bytes    = 0;       // LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES
    uint32_t max_scan_blocks = DEFAULT_MAX_SCAN_BLOCKS;
    uint32_t cooldown_ms     = DEFAULT_COOLDOWN_MS;
    uint32_t backoff_ms      = DEFAULT_BACKOFF_MS;
};

struct server_kv_pressure_bounded_release_event {
    llama_kv_bounded_release_result result;
    kv_pressure_state pressure_state   = kv_pressure_state::NORMAL;
    kv_pressure_source pressure_source = kv_pressure_source::NONE;
    bool    stale           = false;
    bool    idle             = false;
    bool    legacy_enabled   = false;  // paged_block_release_enabled at sample time
    uint64_t sample_count    = 0;
    uint64_t episode         = 0;      // pressure-episode counter
    uint64_t target_bytes    = 0;
    uint32_t max_scan_blocks = 0;
    uint32_t cooldown_ms     = 0;
    uint64_t mincore_before_bytes = 0; // KV resident bytes before release (mincore)
    uint64_t mincore_after_bytes  = 0; // KV resident bytes after release (mincore)
    uint64_t bounded_cnt_bytes_delta  = 0; // independent counter increment this call
    uint64_t bounded_cnt_blocks_delta = 0;
    const char * skipped_reason = nullptr;
    // Per-condition capability decomposition at evaluation time.
    // Populated from bounded_release_can_enable_diagnose().
    int can_enable        = -1;  // -1 = not queried
    int cap_paged         = -1;
    int cap_ingraph       = -1;
    int cap_layers        = -1;
    int cap_row_idx       = -1;
    int cap_swap_disabled = -1;
    int cap_layout        = -1;
};

bool server_kv_pressure_bounded_release_config_from_env(
        server_kv_pressure_bounded_release_config & config, std::string & error);

std::string server_kv_pressure_bounded_release_format_marker(
        const server_kv_pressure_bounded_release_event & event);

// --- Runtime class ---

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

    // --- dry-run bounded release policy ---

    void dry_run_enable(const server_kv_pressure_dry_run_config & cfg) {
        dry_run_config_ = cfg;
        last_dry_run_ = time_point {};
        last_dry_run_state_ = kv_pressure_state::NORMAL;
        dry_run_episode_active_ = false;
        current_cooldown_ms_ = cfg.cooldown_ms;
    }

    void dry_run_disable() {
        dry_run_config_.enabled = false;
        last_dry_run_ = time_point {};
        last_dry_run_state_ = kv_pressure_state::NORMAL;
        dry_run_episode_active_ = false;
        current_cooldown_ms_ = 0;
    }

    // Returns true when a dry-run scan should be evaluated.  Respects:
    //   - master enable + target_bytes > 0
    //   - only PRESSURE / CRITICAL states (fail-closed on NORMAL / RECOVERY)
    //   - stale rejection (no valid sample → no evaluation)
    //   - cooldown / backoff between evaluations within one pressure episode
    //   - NORMAL / RECOVERY end the episode and clear retained cadence state
    //   - state-entry semantics: entering PRESSURE or CRITICAL resets the
    //     cooldown timer so the first evaluation fires immediately.
    //   - CRITICAL: on state entry, bypasses cooldown entirely for that first
    //     evaluation only; subsequent evaluations in sustained CRITICAL are
    //     still subject to cooldown.
    bool dry_run_due(time_point now, kv_pressure_state state,
                     bool stale);

    // Advance cooldown after a completed dry-run scan.
    // now: the time_point used for the evaluation (usually the same `now`
    // passed to dry_run_due).  Extends to backoff_ms on shortfall;
    // otherwise resets to base cooldown_ms.
    void dry_run_record(bool had_shortfall, time_point now = clock::now());

    const server_kv_pressure_dry_run_config & dry_run_config() const {
        return dry_run_config_;
    }

    // --- bounded destructive release policy ---

    void bounded_release_enable(const server_kv_pressure_bounded_release_config & cfg) {
        bounded_release_config_ = cfg;
        last_bounded_release_ = time_point {};
        last_bounded_release_state_ = kv_pressure_state::NORMAL;
        bounded_release_episode_active_ = false;
        bounded_release_current_cooldown_ms_ = cfg.cooldown_ms;
    }

    void bounded_release_disable() {
        bounded_release_config_.enabled = false;
        last_bounded_release_ = time_point {};
        last_bounded_release_state_ = kv_pressure_state::NORMAL;
        bounded_release_episode_active_ = false;
        bounded_release_current_cooldown_ms_ = 0;
    }

    // Returns true when a bounded destructive release should be evaluated.
    // Same episode reset, cooldown, state-entry, and backoff semantics as dry_run_due().
    bool bounded_release_due(time_point now, kv_pressure_state state,
                             bool stale);

    // Advance cooldown after a completed bounded release call.
    void bounded_release_record(bool had_shortfall, time_point now = clock::now());

    const server_kv_pressure_bounded_release_config & bounded_release_config() const {
        return bounded_release_config_;
    }

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

    server_kv_pressure_dry_run_config dry_run_config_;
    time_point last_dry_run_ {};
    kv_pressure_state last_dry_run_state_ = kv_pressure_state::NORMAL;
    bool dry_run_episode_active_ = false;
    uint32_t current_cooldown_ms_ = 0;

    server_kv_pressure_bounded_release_config bounded_release_config_;
    time_point last_bounded_release_ {};
    kv_pressure_state last_bounded_release_state_ = kv_pressure_state::NORMAL;
    bool bounded_release_episode_active_ = false;
    uint32_t bounded_release_current_cooldown_ms_ = 0;
};
