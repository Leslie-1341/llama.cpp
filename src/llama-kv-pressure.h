#pragma once

// KV memory pressure sampler and state machine.
// Linux-only. All functionality is opt-in via LLAMA_KV_PRESSURE_SAMPLER=1.
// Default-off. When disabled, sample() returns NORMAL and telemetry shows enabled=false.
//
// This round implements the sampler + NORMAL/PRESSURE/CRITICAL/RECOVERY state machine
// with structured telemetry output. It does NOT trigger any reclaim, bounded-store
// decisions, swap-out, prefetch, or block-age tracking.
//
// Data sources (C++ direct file reads, no shell):
//   /proc/self/statm          — RSS (resident pages * page_size)
//   cgroup memory.current      — current cgroup memory usage
//   cgroup memory.max          — cgroup memory hard limit ("max" = no limit)
//   cgroup memory.high         — cgroup memory soft-high throttle threshold
//   cgroup memory.pressure     — cgroup v2 PSI (some/full avg10/avg60/avg300 total)
//   /proc/pressure/memory      — system-level memory PSI
//
// State transitions:
//   NORMAL  -> PRESSURE  : pressure threshold exceeded for hysteresis samples + cooldown
//   NORMAL  -> CRITICAL  : critical threshold exceeded — IMMEDIATE, no cooldown
//   PRESSURE-> CRITICAL  : critical threshold exceeded — IMMEDIATE
//   PRESSURE-> CRITICAL  : PSI upgrade (PSI above threshold while in PRESSURE)
//   PRESSURE-> RECOVERY  : below low-water for hysteresis samples + cooldown
//   CRITICAL-> RECOVERY  : below low-water for hysteresis samples + cooldown
//   RECOVERY-> NORMAL    : below low-water for hysteresis samples (continuing) + cooldown
//   RECOVERY-> PRESSURE  : above pressure for hysteresis samples + cooldown
//   RECOVERY-> CRITICAL  : above critical — IMMEDIATE
//
// CRITICAL entry is never blocked by cooldown.
// PRESSURE/CRITICAL exit requires low-water mark, hysteresis, and consecutive valid samples.
// Sample failure marks stale; stale never auto-demotes to NORMAL.
// PSI can only confirm or upgrade persistent pressure, not trigger destructive transitions alone.

#include <cstdint>
#include <cstdio>
#include <string>

struct kv_pressure_paths {
    // Production defaults read the real procfs. Tests may point proc_root at a
    // fixture tree and file_root at a prefix containing mountinfo mount points.
    std::string proc_root = "/proc";
    std::string file_root;
    uint64_t    page_size = 0; // 0 = sysconf(_SC_PAGESIZE)
};

enum class kv_pressure_state : uint8_t {
    NORMAL    = 0,
    PRESSURE  = 1,
    CRITICAL  = 2,
    RECOVERY  = 3,
};

enum class kv_pressure_source : uint8_t {
    NONE            = 0,
    CGROUP_RATIO    = 1,
    RSS_ABSOLUTE    = 2,
    CGROUP_ABSOLUTE = 3,
};

enum class kv_pressure_enablement : uint8_t {
    KV_PRESSURE_ENABLEMENT_DISABLED = 0,
    KV_PRESSURE_ENABLEMENT_ENABLED  = 1,
    KV_PRESSURE_ENABLEMENT_INVALID  = 2,
};

// Read only the existing master switch. This does not initialize the sampler
// or open procfs/cgroup files.
kv_pressure_enablement kv_pressure_sampler_environment_enablement();

const char * kv_pressure_state_name(kv_pressure_state state);
const char * kv_pressure_source_name(kv_pressure_source source);

struct kv_pressure_telemetry {
    bool                enabled          = false;
    kv_pressure_source  source           = kv_pressure_source::NONE;
    kv_pressure_state   state            = kv_pressure_state::NORMAL;
    kv_pressure_state   previous_state   = kv_pressure_state::NORMAL;
    bool                sample_valid     = false;
    bool                stale            = false;
    bool                config_valid     = true;
    uint64_t            rss_kb           = 0;
    uint64_t            cgroup_current_bytes = 0;
    uint64_t            cgroup_max_bytes = 0;
    uint64_t            cgroup_current_kb = 0;
    uint64_t            cgroup_max_kb    = 0;
    uint64_t            cgroup_high_kb   = 0;
    uint32_t            psi_some_avg10   = 0;
    uint32_t            psi_full_avg10   = 0;
    bool                pressure_basis_valid = false;
    uint64_t            pressure_current_bytes = 0;
    uint64_t            pressure_low_water_bytes = 0;
    uint64_t            pressure_basis_generation = 0;
    uint64_t            sample_latency_ns = 0;
    char                transition_reason[128] = {};
    char                disabled_reason[160] = {};
};

struct kv_pressure_config {
    bool     enabled = false;

    // Ratio thresholds in basis points (0-10000 = 0%-100.00%).
    // Only used when memory.max is finite and CGROUP_RATIO is the source.
    uint32_t pressure_ratio   = 8000;
    uint32_t critical_ratio   = 9500;
    uint32_t low_water_ratio  = 7000;

    // Absolute RSS thresholds in KiB (0 = disabled).
    // Used when explicitly configured; takes effect regardless of memory.max value.
    uint64_t pressure_rss_kb   = 0;
    uint64_t critical_rss_kb   = 0;
    uint64_t low_water_rss_kb  = 0;

    // Absolute cgroup current thresholds in KiB (0 = disabled).
    uint64_t pressure_cgroup_kb   = 0;
    uint64_t critical_cgroup_kb   = 0;
    uint64_t low_water_cgroup_kb  = 0;

    // Hysteresis: consecutive valid samples required to exit PRESSURE/CRITICAL.
    uint32_t hysteresis_samples = 3;

    // Cooldown between non-CRITICAL-entry transitions (microseconds).
    // CRITICAL entry bypasses cooldown entirely.
    uint64_t cooldown_us = 5000000;

    // PSI some avg10 threshold (* 100, e.g. 1000 = 10.00%).
    // When PSI exceeds this while in PRESSURE, state can upgrade to CRITICAL.
    // 0 = PSI upgrade disabled.
    uint32_t psi_pressure_upgrade_threshold = 1000;

    // Retained as a validated compatibility knob. Fail-closed sampling marks
    // stale on the first invalid selected-source sample.
    uint32_t stale_threshold_samples = 5;
};

class kv_pressure_sampler {
public:
    kv_pressure_sampler();
    explicit kv_pressure_sampler(const kv_pressure_paths & paths);
    ~kv_pressure_sampler();

    // Parse config from environment variables. Call once before sampling.
    // Returns true when the sampler opt-in is valid. Invalid configuration or
    // unavailable/ambiguous cgroups disable transitions and record a reason;
    // telemetry may remain enabled.
    bool init();

    // Initialize from a previously parsed master-switch snapshot. This keeps a
    // single server initialization decision while preserving init() for other callers.
    bool init(kv_pressure_enablement enablement);

    // Take a real pressure sample from /proc and cgroup, evaluate the state machine.
    // Returns the current state after the sample.
    kv_pressure_state sample();

    // Read-only telemetry snapshot (last sample values + current state).
    const kv_pressure_telemetry & telemetry() const { return telemetry_; }

    // Current state without triggering a new sample.
    kv_pressure_state state() const { return telemetry_.state; }

    // Whether the sampler is enabled.
    bool enabled() const { return config_.enabled; }

    // Exposed for testing: feed a synthetic telemetry snapshot directly to the state
    // machine, bypassing real /proc and cgroup reads. The caller fills in the pressure
    // fields (rss_kb, cgroup_current_kb, cgroup_max_kb, psi_*) and sample_valid/stale;
    // the state machine evaluates transitions against the configured thresholds.
    // Returns the new state.
    kv_pressure_state sample_synthetic(const kv_pressure_telemetry & synthetic);

    // Exposed for testing: get the resolved cgroup version.
    int cgroup_version() const { return cgroup_version_; }

    // Exposed for testing: whether memory.max is the "max" sentinel.
    bool cgroup_max_unlimited() const { return cgroup_max_is_max_; }

    // Exposed for testing: whether state transitions are enabled.
    bool transitions_enabled() const { return transitions_enabled_; }

private:
    friend struct kv_pressure_sampler_test_access;

    kv_pressure_config  config_;
    kv_pressure_telemetry telemetry_;
    kv_pressure_paths   paths_;

    // Timing
    uint64_t last_sample_ns_;
    uint64_t last_transition_ns_;

    // Hysteresis counters
    uint32_t consecutive_pressure_samples_;
    uint32_t consecutive_below_low_water_;
    uint32_t consecutive_failures_;
    uint32_t consecutive_psi_pressure_;

    struct transition_basis {
        bool               initialized = false;
        kv_pressure_source source = kv_pressure_source::NONE;
        bool               cgroup_max_unlimited = false;
        uint64_t           low_water = 0;
        uint64_t           pressure = 0;
        uint64_t           critical = 0;
        uint64_t           ratio_maximum = 0;
    };

    transition_basis transition_basis_;
    uint64_t pressure_basis_generation_ = 0;

    // Resolved cgroup state
    std::string cgroup_mem_path_;
    int         cgroup_version_;   // 2, 1, or 0 (none)
    bool        cgroup_max_is_max_;
    bool        config_valid_;

    // Whether state transitions are enabled (vs telemetry-only mode)
    bool        transitions_enabled_;

    // -- file reading helpers (no shell) --

    enum class limit_kind : uint8_t {
        INVALID,
        FINITE,
        UNLIMITED,
    };

    static bool read_file_text(const std::string & path, std::string & value);
    static bool parse_uint64(const std::string & text, uint64_t & value);
    static limit_kind parse_limit(const std::string & text, uint64_t & value);
    static bool checked_add(uint64_t a, uint64_t b, uint64_t & result);
    static bool checked_mul(uint64_t a, uint64_t b, uint64_t & result);
    static bool ratio_at_least(uint64_t current, uint64_t maximum, uint32_t basis_points);

    bool read_rss_kb(uint64_t & rss_kb) const;
    bool read_cgroup_current(uint64_t & current_bytes) const;
    limit_kind read_cgroup_max(uint64_t & max_bytes) const;
    limit_kind read_cgroup_high(uint64_t & high_bytes) const;
    bool read_psi(const char * path,
                   uint32_t & some_avg10, uint32_t & full_avg10,
                   uint64_t & some_total, uint64_t & full_total) const;

    // -- cgroup resolution --

    bool resolve_cgroup();
    void determine_source();
    bool validate_config();
    void disable_transitions(const char * reason);

    // -- state machine --

    void evaluate_transition();
    void set_state(kv_pressure_state new_state, const char * reason);
    void reset_transition_counters();
    void refresh_transition_basis(const kv_pressure_telemetry & sample);

    static void saturating_increment(uint32_t & value);

    static uint64_t get_time_ns();
};
