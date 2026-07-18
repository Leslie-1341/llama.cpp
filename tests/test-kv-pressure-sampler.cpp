#include "../src/llama-kv-pressure.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

struct kv_pressure_sampler_test_access {
    static uint32_t increment(uint32_t value) {
        kv_pressure_sampler::saturating_increment(value);
        return value;
    }

    static uint32_t consecutive_psi_pressure(const kv_pressure_sampler & sampler) {
        return sampler.consecutive_psi_pressure_;
    }
};

// ── minimal test harness ────────────────────────────────────────────────────

static int tests_total  = 0;
static int tests_failed = 0;

static void check(bool condition, const char * message) {
    tests_total++;
    if (!condition) {
        tests_failed++;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

#define CHECK(cond) check((cond), #cond)

// Debug helper to print state values on failure
static void check_state(kv_pressure_state actual, kv_pressure_state expected,
                         const char * file, int line) {
    tests_total++;
    if (actual != expected) {
        tests_failed++;
        std::fprintf(stderr, "FAIL [%s:%d]: state=%d expected=%d\n",
                     file, line, (int)actual, (int)expected);
    }
}
#define CHECK_STATE(actual, expected) check_state((actual), (expected), __FILE__, __LINE__)

// ── helpers ─────────────────────────────────────────────────────────────────

static kv_pressure_telemetry make_telemetry(
        bool valid, uint64_t rss_kb, uint64_t cg_current_kb, uint64_t cg_max_kb,
        uint32_t psi_some_avg10, uint32_t psi_full_avg10,
        kv_pressure_source src = kv_pressure_source::CGROUP_RATIO)
{
    kv_pressure_telemetry t{};
    t.enabled           = true;
    t.source            = src;
    t.sample_valid      = valid;
    t.stale             = false;
    t.rss_kb            = rss_kb;
    t.cgroup_current_kb = cg_current_kb;
    t.cgroup_max_kb     = cg_max_kb;
    t.psi_some_avg10    = psi_some_avg10;
    t.psi_full_avg10    = psi_full_avg10;
    t.sample_latency_ns = 1000;
    return t;
}

// Inject N identical synthetic samples and return the final state
static kv_pressure_state inject_n(kv_pressure_sampler & s,
                                   const kv_pressure_telemetry & tpl, int n) {
    kv_pressure_state st = kv_pressure_state::NORMAL;
    for (int i = 0; i < n; i++) {
        st = s.sample_synthetic(tpl);
    }
    return st;
}

static const char * pressure_env_names[] = {
    "LLAMA_KV_PRESSURE_SAMPLER",
    "LLAMA_KV_PRESSURE_RATIO",
    "LLAMA_KV_CRITICAL_RATIO",
    "LLAMA_KV_LOW_WATER_RATIO",
    "LLAMA_KV_PRESSURE_RSS_KB",
    "LLAMA_KV_CRITICAL_RSS_KB",
    "LLAMA_KV_LOW_WATER_RSS_KB",
    "LLAMA_KV_PRESSURE_CGROUP_KB",
    "LLAMA_KV_CRITICAL_CGROUP_KB",
    "LLAMA_KV_LOW_WATER_CGROUP_KB",
    "LLAMA_KV_PRESSURE_HYSTERESIS",
    "LLAMA_KV_PRESSURE_COOLDOWN_US",
    "LLAMA_KV_PSI_UPGRADE_THRESHOLD",
    "LLAMA_KV_PRESSURE_STALE_THRESHOLD",
};

static void clear_pressure_env() {
    for (size_t i = 0; i < sizeof(pressure_env_names) / sizeof(pressure_env_names[0]); ++i) {
        unsetenv(pressure_env_names[i]);
    }
}

static void set_base_pressure_env() {
    clear_pressure_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "1", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);
}

struct pressure_fixture {
    std::string root;

    pressure_fixture() {
        char path[] = "/tmp/llama-kv-pressure-XXXXXX";
        char * made = mkdtemp(path);
        CHECK(made != nullptr);
        if (made) {
            root = made;
        }
    }

    ~pressure_fixture() {
        if (!root.empty()) {
            std::error_code error;
            std::filesystem::remove_all(root, error);
        }
    }

    std::string path(const std::string & absolute) const {
        return root + absolute;
    }

    void write(const std::string & absolute, const std::string & contents) const {
        const std::filesystem::path target(path(absolute));
        std::error_code error;
        std::filesystem::create_directories(target.parent_path(), error);
        CHECK(!error);
        FILE * file = std::fopen(target.string().c_str(), "wb");
        CHECK(file != nullptr);
        if (!file) {
            return;
        }
        CHECK(std::fwrite(contents.data(), 1, contents.size(), file) == contents.size());
        CHECK(std::fclose(file) == 0);
    }

    void remove(const std::string & absolute) const {
        std::error_code error;
        std::filesystem::remove(path(absolute), error);
    }

    kv_pressure_paths paths(uint64_t page_size = 4096) const {
        kv_pressure_paths result;
        result.proc_root = path("/proc");
        result.file_root = root;
        result.page_size = page_size;
        return result;
    }

    void write_proc_defaults(const std::string & statm = "100 50 0 0 0 0 0\n") const {
        write("/proc/self/statm", statm);
        write("/proc/pressure/memory",
              "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
              "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n");
    }

    void write_v2(const std::string & cgroup_path, const std::string & mount_root,
                  const std::string & mount_point_encoded, const std::string & mount_point_decoded,
                  const std::string & current, const std::string & maximum,
                  const std::string & high) const {
        write("/proc/self/cgroup", "0::" + cgroup_path + "\n");
        std::string root_encoded = mount_root;
        size_t pos = 0;
        while ((pos = root_encoded.find(' ', pos)) != std::string::npos) {
            root_encoded.replace(pos, 1, "\\040");
            pos += 4;
        }
        write("/proc/self/mountinfo", "36 25 0:32 " + root_encoded + " " + mount_point_encoded +
              " rw - cgroup2 cgroup2 rw\n");
        std::string suffix;
        if (mount_root == "/") {
            suffix = cgroup_path == "/" ? "" : cgroup_path;
        } else if (cgroup_path == mount_root) {
            suffix = "";
        } else {
            suffix = cgroup_path.substr(mount_root.size());
        }
        const std::string group = mount_point_decoded + suffix;
        write(group + "/memory.current", current);
        write(group + "/memory.max", maximum);
        write(group + "/memory.high", high);
        write(group + "/memory.pressure",
              "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
              "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n");
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// Test: default-off behavior
// ═════════════════════════════════════════════════════════════════════════════

static void test_default_off() {
    // Without LLAMA_KV_PRESSURE_SAMPLER env, init() should return false
    // and sample() should return NORMAL with telemetry showing enabled=false.
    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    kv_pressure_sampler s;
    CHECK(!s.init());            // returns false when env not set
    CHECK(!s.enabled());
    CHECK_STATE(s.state(), kv_pressure_state::NORMAL);
    CHECK(!s.telemetry().enabled);

    auto st = s.sample();        // no-op
    CHECK_STATE(st, kv_pressure_state::NORMAL);
    CHECK(!s.telemetry().enabled);

    // Also test with explicit disable
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "0", 1);
    kv_pressure_sampler s2;
    CHECK(!s2.init());
    CHECK(!s2.enabled());
    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: CGROUP_RATIO source — NORMAL → PRESSURE → CRITICAL → RECOVERY → NORMAL
// ═════════════════════════════════════════════════════════════════════════════

static void test_full_cycle_cgroup_ratio() {
    // Configure environment for ratio-based transitions:
    // pressure_ratio=80%  critical_ratio=95%  low_water_ratio=70%
    // hysteresis=2  cooldown=0 (immediate for test)
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);  // disable PSI for this test

    kv_pressure_sampler s;
    // Note: init() may return false on systems without cgroup v2, but the state
    // machine logic is still testable via sample_synthetic().
    s.init();

    // Override for testing: manually enable transitions with CGROUP_RATIO source
    // The sampler's init() resolves cgroup state; for synthetic testing we can
    // exercise the state machine independent of cgroup availability.

    // For synthetic testing, we bypass init's source resolution by directly
    // calling sample_synthetic which evaluates the state machine using the
    // provided telemetry values against (already configured) thresholds.

    // Start in NORMAL
    CHECK_STATE(s.state(), kv_pressure_state::NORMAL);

    // --- Phase 1: NORMAL → PRESSURE (pressure_ratio=80%, need hysteresis=2) ---
    // cgroup: 85% usage (8500/10000)
    auto t_pressure = make_telemetry(true, 0, 8500, 10000, 0, 0);

    // First sample at 85% — building hysteresis
    auto st = s.sample_synthetic(t_pressure);
    CHECK(st == kv_pressure_state::NORMAL || st == kv_pressure_state::PRESSURE);
    // After hysteresis=2 samples, should enter PRESSURE
    st = s.sample_synthetic(t_pressure);
    CHECK_STATE(st, kv_pressure_state::PRESSURE);
    CHECK(s.telemetry().state == kv_pressure_state::PRESSURE);

    // --- Phase 2: PRESSURE → CRITICAL (critical_ratio=95%, immediate entry) ---
    // cgroup: 96% usage
    auto t_critical = make_telemetry(true, 0, 9600, 10000, 0, 0);
    st = s.sample_synthetic(t_critical);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);
    CHECK(s.telemetry().state == kv_pressure_state::CRITICAL);
    // Verify transition reason contains "critical" and "immediate"
    CHECK(std::strstr(s.telemetry().transition_reason, "critical") != nullptr);

    // --- Phase 3: CRITICAL → RECOVERY (below low_water=70%, hysteresis=2) ---
    // cgroup: 60% usage
    auto t_low = make_telemetry(true, 0, 6000, 10000, 0, 0);
    st = s.sample_synthetic(t_low);
    // first sample below low-water — building exit hysteresis (1/2)
    CHECK_STATE(st, kv_pressure_state::CRITICAL);
    st = s.sample_synthetic(t_low);
    // second sample below low-water + cooldown=0 → RECOVERY
    CHECK_STATE(st, kv_pressure_state::RECOVERY);
    CHECK_STATE(s.telemetry().state, kv_pressure_state::RECOVERY);

    // --- Phase 4: RECOVERY → NORMAL (fresh hysteresis from RECOVERY) ---
    // Still at 60% — building NORMAL exit hysteresis (1/2)
    st = s.sample_synthetic(t_low);
    CHECK_STATE(st, kv_pressure_state::RECOVERY);
    // Second below-low-water from RECOVERY → NORMAL
    st = s.sample_synthetic(t_low);
    CHECK_STATE(st, kv_pressure_state::NORMAL);
    CHECK_STATE(s.telemetry().state, kv_pressure_state::NORMAL);

    // Clean up environment
    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: CRITICAL entry bypasses cooldown
// ═════════════════════════════════════════════════════════════════════════════

static void test_critical_bypasses_cooldown() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "999999999", 1); // effectively infinite cooldown
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // First, enter PRESSURE with hysteresis=2 samples (first transition always allowed)
    auto t_pressure = make_telemetry(true, 0, 8500, 10000, 0, 0);
    auto st = s.sample_synthetic(t_pressure); // building hysteresis
    st = s.sample_synthetic(t_pressure);       // enters PRESSURE (first transition)
    CHECK_STATE(st, kv_pressure_state::PRESSURE);

    // Now, with infinite cooldown, PRESSURE→RECOVERY should be blocked
    auto t_low = make_telemetry(true, 0, 6000, 10000, 0, 0);
    st = s.sample_synthetic(t_low);  // below low-water #1, building
    st = s.sample_synthetic(t_low);  // below low-water #2, cooldown blocks RECOVERY
    CHECK_STATE(st, kv_pressure_state::PRESSURE); // cooldown not elapsed for exit

    // But PRESSURE→CRITICAL should happen immediately regardless of cooldown
    auto t_critical = make_telemetry(true, 0, 9600, 10000, 0, 0);
    st = s.sample_synthetic(t_critical);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);
    CHECK_STATE(s.telemetry().state, kv_pressure_state::CRITICAL);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: CRITICAL exit requires low-water + hysteresis + cooldown
// ═════════════════════════════════════════════════════════════════════════════

static void test_critical_exit_hysteresis() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "3", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // Enter CRITICAL immediately
    auto t_critical = make_telemetry(true, 0, 9600, 10000, 0, 0);
    auto st = s.sample_synthetic(t_critical);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);

    // Drop below low-water, but only 2 samples — should stay in CRITICAL
    auto t_low = make_telemetry(true, 0, 6000, 10000, 0, 0);
    st = s.sample_synthetic(t_low);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);
    st = s.sample_synthetic(t_low);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);

    // Third sample → RECOVERY
    st = s.sample_synthetic(t_low);
    CHECK_STATE(st, kv_pressure_state::RECOVERY);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: threshold jitter — brief spikes don't trigger premature transitions
// ═════════════════════════════════════════════════════════════════════════════

static void test_threshold_jitter() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "3", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();
    CHECK_STATE(s.state(), kv_pressure_state::NORMAL);

    // One-off spike above pressure — not enough for hysteresis
    auto t_spike = make_telemetry(true, 0, 8500, 10000, 0, 0);
    auto st = s.sample_synthetic(t_spike);
    CHECK_STATE(st, kv_pressure_state::NORMAL);

    // Drop back below
    auto t_normal = make_telemetry(true, 0, 5000, 10000, 0, 0);
    st = s.sample_synthetic(t_normal);
    CHECK_STATE(st, kv_pressure_state::NORMAL);

    // Hysteresis counter should have been reset by the normal sample
    // Another spike should restart the counter
    st = s.sample_synthetic(t_spike);
    CHECK_STATE(st, kv_pressure_state::NORMAL);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: memory.max = max — telemetry-only mode
// ═════════════════════════════════════════════════════════════════════════════

static void test_max_unlimited_telemetry_only() {
    // Simulate memory.max = "max" with no absolute thresholds
    // The determine_source() logic would set source=NONE and transitions_enabled_=false.
    // For the synthetic test, we directly verify that when source is NONE,
    // the state machine stays in NORMAL regardless of pressure.
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "1", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // Feed samples with NONE source — should not change state
    auto t = make_telemetry(true, 100000, 9600, 0, 0, 0, kv_pressure_source::NONE);
    // cgroup_max_kb=0 simulates "max" sentinel

    for (int i = 0; i < 10; i++) {
        auto st = s.sample_synthetic(t);
        // In telemetry-only mode, should stay NORMAL regardless of pressure
        CHECK_STATE(st, kv_pressure_state::NORMAL);
    }

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: RSS absolute threshold mode
// ═════════════════════════════════════════════════════════════════════════════

static void test_rss_absolute_thresholds() {
    // Simulate scenario where memory.max="max" but RSS thresholds are explicitly set
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RSS_KB", "1000000", 1);    // 1 GB
    setenv("LLAMA_KV_CRITICAL_RSS_KB", "2000000", 1);    // 2 GB
    setenv("LLAMA_KV_LOW_WATER_RSS_KB", "800000", 1);    // 800 MB
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "1", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // Feed samples with RSS_ABSOLUTE source
    auto t = make_telemetry(true, 1500000, 0, 0, 0, 0, kv_pressure_source::RSS_ABSOLUTE);
    // 1.5 GB RSS (> 1 GB pressure threshold)

    auto st = s.sample_synthetic(t);
    CHECK_STATE(st, kv_pressure_state::PRESSURE);

    // Critical threshold
    t.rss_kb = 2500000;  // 2.5 GB RSS (> 2 GB critical)
    st = s.sample_synthetic(t);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);

    // Recovery
    t.rss_kb = 700000;  // 700 MB (< 800 MB low water)
    t.source = kv_pressure_source::RSS_ABSOLUTE;
    st = s.sample_synthetic(t);
    CHECK_STATE(st, kv_pressure_state::RECOVERY);

    // Back to NORMAL
    st = s.sample_synthetic(t);
    CHECK_STATE(st, kv_pressure_state::NORMAL);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RSS_KB");
    unsetenv("LLAMA_KV_CRITICAL_RSS_KB");
    unsetenv("LLAMA_KV_LOW_WATER_RSS_KB");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: PSI upgrade — PRESSURE + high PSI → CRITICAL
// ═════════════════════════════════════════════════════════════════════════════

static void test_psi_upgrade() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "1000", 1);  // 10% PSI some avg10

    kv_pressure_sampler s;
    s.init();

    // Enter PRESSURE first (82% usage, normal PSI)
    auto t_pressure = make_telemetry(true, 0, 8200, 10000, 500, 200);
    inject_n(s, t_pressure, 2);
    CHECK_STATE(s.state(), kv_pressure_state::PRESSURE);

    // Now add high PSI while still in PRESSURE zone (not critical by ratio)
    auto t_psi_high = make_telemetry(true, 0, 8200, 10000, 1500, 800);
    // PSI some_avg10 = 15% > 10% upgrade threshold

    // First PSI-high sample — building hysteresis
    auto st = s.sample_synthetic(t_psi_high);
    CHECK_STATE(st, kv_pressure_state::PRESSURE); // still building hysteresis

    // Second PSI-high sample — PSI upgrade to CRITICAL
    st = s.sample_synthetic(t_psi_high);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);
    CHECK_STATE(s.telemetry().state, kv_pressure_state::CRITICAL);
    // Verify transition reason mentions PSI
    CHECK(std::strstr(s.telemetry().transition_reason, "PSI") != nullptr);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: PSI upgrade accumulation resets across a full recovery cycle
// ═════════════════════════════════════════════════════════════════════════════

static void test_psi_upgrade_resets_across_full_recovery_cycle() {
    clear_pressure_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "1000", 1);

    kv_pressure_sampler sampler;
    CHECK(sampler.init());

    const auto pressure = make_telemetry(true, 0, 8500, 10000, 0, 0);
    const auto pressure_high_psi = make_telemetry(true, 0, 8500, 10000, 1500, 0);
    const auto low = make_telemetry(true, 0, 6000, 10000, 0, 0);

    inject_n(sampler, pressure, 2);
    CHECK_STATE(sampler.state(), kv_pressure_state::PRESSURE);
    CHECK_STATE(sampler.sample_synthetic(pressure_high_psi), kv_pressure_state::PRESSURE);
    CHECK(kv_pressure_sampler_test_access::consecutive_psi_pressure(sampler) == 1);
    CHECK_STATE(sampler.sample_synthetic(pressure_high_psi), kv_pressure_state::CRITICAL);
    CHECK(kv_pressure_sampler_test_access::consecutive_psi_pressure(sampler) == 0);

    inject_n(sampler, low, 2);
    CHECK_STATE(sampler.state(), kv_pressure_state::RECOVERY);
    CHECK(kv_pressure_sampler_test_access::consecutive_psi_pressure(sampler) == 0);
    inject_n(sampler, low, 2);
    CHECK_STATE(sampler.state(), kv_pressure_state::NORMAL);
    CHECK(kv_pressure_sampler_test_access::consecutive_psi_pressure(sampler) == 0);

    CHECK_STATE(sampler.sample_synthetic(pressure_high_psi), kv_pressure_state::NORMAL);
    CHECK_STATE(sampler.sample_synthetic(pressure_high_psi), kv_pressure_state::PRESSURE);
    CHECK(kv_pressure_sampler_test_access::consecutive_psi_pressure(sampler) == 0);

    CHECK_STATE(sampler.sample_synthetic(pressure_high_psi), kv_pressure_state::PRESSURE);
    CHECK(kv_pressure_sampler_test_access::consecutive_psi_pressure(sampler) == 1);
    CHECK_STATE(sampler.sample_synthetic(pressure_high_psi), kv_pressure_state::CRITICAL);
    CHECK(kv_pressure_sampler_test_access::consecutive_psi_pressure(sampler) == 0);

    clear_pressure_env();
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: PSI alone does not trigger transitions from NORMAL
// ═════════════════════════════════════════════════════════════════════════════

static void test_psi_alone_no_transition() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "500", 1);

    kv_pressure_sampler s;
    s.init();
    CHECK_STATE(s.state(), kv_pressure_state::NORMAL);

    // High PSI but low cgroup usage (50%)
    auto t_psi_only = make_telemetry(true, 0, 5000, 10000, 2000, 1000);

    // Should NOT transition even with many samples — PSI alone doesn't trigger
    for (int i = 0; i < 10; i++) {
        auto st = s.sample_synthetic(t_psi_only);
        CHECK_STATE(st, kv_pressure_state::NORMAL);
    }

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

static void test_low_water_wins_over_high_psi() {
    clear_pressure_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "1000", 1);

    kv_pressure_sampler sampler;
    CHECK(sampler.init());
    const auto pressure = make_telemetry(true, 0, 8500, 10000, 0, 0);
    inject_n(sampler, pressure, 2);
    CHECK_STATE(sampler.state(), kv_pressure_state::PRESSURE);

    const auto low_with_high_psi = make_telemetry(true, 0, 6000, 10000, 5000, 2500);
    CHECK_STATE(sampler.sample_synthetic(low_with_high_psi), kv_pressure_state::PRESSURE);
    CHECK_STATE(sampler.sample_synthetic(low_with_high_psi), kv_pressure_state::RECOVERY);
    CHECK(std::strstr(sampler.telemetry().transition_reason, "low-water") != nullptr);
    clear_pressure_env();
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: sample failure → stale; never auto-demotes to NORMAL
// ═════════════════════════════════════════════════════════════════════════════

static void test_sample_failure_stale() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "1", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PRESSURE_STALE_THRESHOLD", "3", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // First: enter PRESSURE with valid samples
    auto t_pressure = make_telemetry(true, 0, 8500, 10000, 0, 0);
    auto st = s.sample_synthetic(t_pressure);
    CHECK_STATE(st, kv_pressure_state::PRESSURE);
    CHECK(s.telemetry().state == kv_pressure_state::PRESSURE);
    CHECK(!s.telemetry().stale);

    // Now: inject invalid samples (sample_valid=false)
    auto t_invalid = make_telemetry(false, 0, 0, 0, 0, 0);
    // should NOT be valid
    t_invalid.sample_valid = false;

    st = s.sample_synthetic(t_invalid);
    CHECK_STATE(st, kv_pressure_state::PRESSURE); // stays in PRESSURE
    st = s.sample_synthetic(t_invalid);
    CHECK_STATE(st, kv_pressure_state::PRESSURE);
    // Third consecutive failure → stale
    st = s.sample_synthetic(t_invalid);
    CHECK_STATE(st, kv_pressure_state::PRESSURE); // STILL PRESSURE, not demoted
    CHECK(s.telemetry().stale);               // but marked stale

    // Even more failures — still should not demote to NORMAL
    for (int i = 0; i < 10; i++) {
        st = s.sample_synthetic(t_invalid);
        CHECK(st != kv_pressure_state::NORMAL);
        CHECK(st != kv_pressure_state::RECOVERY);
    }

    // Recovery: a valid sample should clear stale and re-evaluate
    st = s.sample_synthetic(t_pressure);
    CHECK(!s.telemetry().stale);
    CHECK_STATE(st, kv_pressure_state::PRESSURE); // still in PRESSURE (pressure is high)

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PRESSURE_STALE_THRESHOLD");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: stale recovery — valid sample after stale restores state evaluation
// ═════════════════════════════════════════════════════════════════════════════

static void test_stale_recovery() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PRESSURE_STALE_THRESHOLD", "2", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // Enter PRESSURE
    auto t_pressure = make_telemetry(true, 0, 8500, 10000, 0, 0);
    inject_n(s, t_pressure, 2);
    CHECK_STATE(s.state(), kv_pressure_state::PRESSURE);

    // Go stale with invalid samples
    auto t_invalid = make_telemetry(false, 0, 0, 0, 0, 0);
    inject_n(s, t_invalid, 3);
    CHECK(s.telemetry().stale);

    // Now feed a valid sample showing pressure has dropped below low-water
    auto t_low = make_telemetry(true, 0, 6000, 10000, 0, 0);
    auto st = s.sample_synthetic(t_low);
    CHECK(!s.telemetry().stale); // stale cleared by valid sample

    // State should re-evaluate: was in PRESSURE, now below low-water
    // First sample below low-water: building exit hysteresis
    // But the pressure counter was reset during invalid samples, so this is
    // the first "below low-water" sample → stay in PRESSURE
    CHECK_STATE(st, kv_pressure_state::PRESSURE);

    // Second below-low-water sample → RECOVERY
    st = s.sample_synthetic(t_low);
    CHECK_STATE(st, kv_pressure_state::RECOVERY);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PRESSURE_STALE_THRESHOLD");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: CGROUP_ABSOLUTE source
// ═════════════════════════════════════════════════════════════════════════════

static void test_cgroup_absolute_thresholds() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_CGROUP_KB", "500000", 1);    // 500 MB
    setenv("LLAMA_KV_CRITICAL_CGROUP_KB", "900000", 1);    // 900 MB
    setenv("LLAMA_KV_LOW_WATER_CGROUP_KB", "400000", 1);   // 400 MB
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "1", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    auto t = make_telemetry(true, 0, 600000, 0, 0, 0, kv_pressure_source::CGROUP_ABSOLUTE);
    // cgroup current = 600 MB (> 500 MB pressure threshold)

    auto st = s.sample_synthetic(t);
    CHECK_STATE(st, kv_pressure_state::PRESSURE);

    // Critical
    t.cgroup_current_kb = 1000000; // 1 GB
    st = s.sample_synthetic(t);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);

    // Recovery
    t.cgroup_current_kb = 300000; // 300 MB (< 400 MB low water)
    inject_n(s, t, 2);
    CHECK_STATE(s.state(), kv_pressure_state::NORMAL);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_CGROUP_KB");
    unsetenv("LLAMA_KV_CRITICAL_CGROUP_KB");
    unsetenv("LLAMA_KV_LOW_WATER_CGROUP_KB");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: telemetry struct reports correct fields
// ═════════════════════════════════════════════════════════════════════════════

static void test_telemetry_fields() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "1", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // Feed a critical sample
    auto t = make_telemetry(true, 500000, 9600, 10000, 300, 150);
    t.sample_latency_ns = 12345;
    s.sample_synthetic(t);

    const auto & tel = s.telemetry();
    CHECK(tel.enabled);
    CHECK(tel.sample_valid);
    CHECK(!tel.stale);
    CHECK(tel.rss_kb == 500000);
    CHECK(tel.cgroup_current_kb == 9600);
    CHECK(tel.cgroup_max_kb == 10000);
    CHECK(tel.psi_some_avg10 == 300);
    CHECK(tel.psi_full_avg10 == 150);
    CHECK(tel.sample_latency_ns > 0);
    CHECK(std::strlen(tel.transition_reason) > 0);

    // previous_state should reflect the transition from NORMAL to CRITICAL
    CHECK(tel.state == kv_pressure_state::CRITICAL);
    // previous_state was NORMAL before the CRITICAL transition
    // (first sample entered CRITICAL immediately)
    CHECK(tel.previous_state == kv_pressure_state::NORMAL);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: config parse edge cases
// ═════════════════════════════════════════════════════════════════════════════

static void test_config_edge_cases() {
    // Test with invalid values — should use defaults
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "not_a_number", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "0", 1);   // should clamp to 1
    setenv("LLAMA_KV_PRESSURE_STALE_THRESHOLD", "0", 1); // should clamp to 1

    kv_pressure_sampler s;
    s.init();
    // Should not crash; should use defaults
    CHECK(s.enabled());

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_STALE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: RECOVERY → PRESSURE re-entry
// ═════════════════════════════════════════════════════════════════════════════

static void test_recovery_reentry_pressure() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // Enter CRITICAL
    auto t_crit = make_telemetry(true, 0, 9600, 10000, 0, 0);
    s.sample_synthetic(t_crit);
    CHECK_STATE(s.state(), kv_pressure_state::CRITICAL);

    // Exit to RECOVERY
    auto t_low = make_telemetry(true, 0, 6000, 10000, 0, 0);
    inject_n(s, t_low, 2);
    CHECK_STATE(s.state(), kv_pressure_state::RECOVERY);

    // Re-enter PRESSURE from RECOVERY (pressure returned, hysteresis=2)
    auto t_pressure = make_telemetry(true, 0, 8500, 10000, 0, 0);
    auto st = s.sample_synthetic(t_pressure);
    CHECK_STATE(st, kv_pressure_state::RECOVERY); // building hysteresis
    st = s.sample_synthetic(t_pressure);
    CHECK_STATE(st, kv_pressure_state::PRESSURE); // re-entered PRESSURE

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: immediate CRITICAL from RECOVERY
// ═════════════════════════════════════════════════════════════════════════════

static void test_critical_from_recovery() {
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    setenv("LLAMA_KV_CRITICAL_RATIO", "9500", 1);
    setenv("LLAMA_KV_LOW_WATER_RATIO", "7000", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    kv_pressure_sampler s;
    s.init();

    // Enter CRITICAL, then RECOVERY
    auto t_crit = make_telemetry(true, 0, 9600, 10000, 0, 0);
    s.sample_synthetic(t_crit);
    CHECK_STATE(s.state(), kv_pressure_state::CRITICAL);

    // CRITICAL→RECOVERY: 2 low-water samples (hysteresis=2), counter reset on RECOVERY entry
    // RECOVERY→NORMAL: 2 more low-water samples (fresh hysteresis)
    // After 3 low-water samples: CRITICAL→RECOVERY (samples 1-2)→RECOVERY (sample 3, building)
    auto t_low = make_telemetry(true, 0, 6000, 10000, 0, 0);
    inject_n(s, t_low, 3);
    CHECK_STATE(s.state(), kv_pressure_state::RECOVERY);

    // Jump straight back to CRITICAL from RECOVERY (immediate, no hysteresis buildup)
    auto st = s.sample_synthetic(t_crit);
    CHECK_STATE(st, kv_pressure_state::CRITICAL);

    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_RATIO");
    unsetenv("LLAMA_KV_CRITICAL_RATIO");
    unsetenv("LLAMA_KV_LOW_WATER_RATIO");
    unsetenv("LLAMA_KV_PRESSURE_HYSTERESIS");
    unsetenv("LLAMA_KV_PRESSURE_COOLDOWN_US");
    unsetenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD");
}

static void test_real_v2_resolution_non_root_and_escape() {
    set_base_pressure_env();
    pressure_fixture fixture;
    fixture.write_proc_defaults();
    fixture.write_v2("/tenant root/job", "/tenant root", "/sys/fs/cgroup\\040space",
                     "/sys/fs/cgroup space", "8704000\n", "10240000\n", "9216000\n");
    std::string mountinfo;
    for (int i = 0; i < 120; ++i) {
        mountinfo += std::to_string(100 + i) + " 25 0:1 / /tmp/irrelevant rw - tmpfs tmpfs rw\n";
    }
    mountinfo += "36 25 0:32 /tenant\\040root /sys/fs/cgroup\\040space rw - cgroup2 cgroup2 rw\n";
    CHECK(mountinfo.size() > 4096);
    fixture.write("/proc/self/mountinfo", mountinfo);

    kv_pressure_sampler sampler(fixture.paths());
    CHECK(sampler.init());
    CHECK(sampler.cgroup_version() == 2);
    CHECK(sampler.transitions_enabled());
    CHECK(sampler.telemetry().source == kv_pressure_source::CGROUP_RATIO);
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
    CHECK(sampler.telemetry().sample_valid);
    CHECK(sampler.telemetry().cgroup_current_kb == 8500);
    CHECK(sampler.telemetry().cgroup_max_kb == 10000);
    CHECK(sampler.telemetry().cgroup_high_kb == 9000);

    fixture.write("/sys/fs/cgroup space/job/memory.current", "8001\n");
    fixture.write("/sys/fs/cgroup space/job/memory.max", "10001\n");
    kv_pressure_sampler byte_exact_ratio(fixture.paths());
    CHECK(byte_exact_ratio.init());
    CHECK_STATE(byte_exact_ratio.sample(), kv_pressure_state::PRESSURE);
    CHECK(byte_exact_ratio.telemetry().cgroup_current_bytes == 8001);
    CHECK(byte_exact_ratio.telemetry().cgroup_max_bytes == 10001);
    clear_pressure_env();
}

static void test_cgroup_ambiguity_disables_source() {
    set_base_pressure_env();
    pressure_fixture fixture;
    fixture.write_proc_defaults();
    fixture.write_v2("/tenant/job", "/tenant", "/sys/fs/cgroup-a", "/sys/fs/cgroup-a",
                     "8704000\n", "10240000\n", "max\n");
    fixture.write("/proc/self/mountinfo",
                  "36 25 0:32 /tenant /sys/fs/cgroup-a rw - cgroup2 cgroup2 rw\n"
                  "37 25 0:33 /tenant /sys/fs/cgroup-b rw - cgroup2 cgroup2 rw\n");

    kv_pressure_sampler duplicate_mount(fixture.paths());
    CHECK(duplicate_mount.init());
    CHECK(duplicate_mount.cgroup_version() == 0);
    CHECK(!duplicate_mount.transitions_enabled());
    CHECK(std::strstr(duplicate_mount.telemetry().disabled_reason, "ambiguous") != nullptr);

    fixture.write("/proc/self/cgroup", "0::/tenant/job\n0::/tenant/job\n");
    fixture.write("/proc/self/mountinfo",
                  "36 25 0:32 /tenant /sys/fs/cgroup-a rw - cgroup2 cgroup2 rw\n");
    kv_pressure_sampler duplicate_membership(fixture.paths());
    CHECK(duplicate_membership.init());
    CHECK(duplicate_membership.cgroup_version() == 0);
    CHECK(!duplicate_membership.transitions_enabled());

    fixture.write("/proc/self/cgroup", "0::/tenant/job\n5:memory:/legacy/job\n");
    fixture.write("/proc/self/mountinfo",
                  "36 25 0:32 /tenant /sys/fs/cgroup-a rw - cgroup2 cgroup2 rw\n"
                  "38 25 0:34 /legacy /sys/fs/cgroup-v1 rw - cgroup cgroup rw,memory\n");
    kv_pressure_sampler hybrid(fixture.paths());
    CHECK(hybrid.init());
    CHECK(hybrid.cgroup_version() == 0);
    CHECK(!hybrid.transitions_enabled());
    clear_pressure_env();
}

static void test_real_v1_resolution_non_root() {
    set_base_pressure_env();
    pressure_fixture fixture;
    fixture.write_proc_defaults();
    fixture.write("/proc/self/cgroup", "5:cpu,memory:/docker/root/job\n");
    fixture.write("/proc/self/mountinfo",
                  "38 25 0:34 /docker/root /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n");
    fixture.write("/sys/fs/cgroup/memory/job/memory.usage_in_bytes", "8704000\n");
    fixture.write("/sys/fs/cgroup/memory/job/memory.limit_in_bytes", "10240000\n");
    fixture.write("/sys/fs/cgroup/memory/job/memory.soft_limit_in_bytes", "9216000\n");

    kv_pressure_sampler sampler(fixture.paths());
    CHECK(sampler.init());
    CHECK(sampler.cgroup_version() == 1);
    CHECK(sampler.transitions_enabled());
    CHECK(sampler.telemetry().source == kv_pressure_source::CGROUP_RATIO);
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
    CHECK(sampler.telemetry().sample_valid);
    CHECK(sampler.telemetry().cgroup_current_kb == 8500);
    CHECK(sampler.telemetry().cgroup_max_kb == 10000);
    clear_pressure_env();
}

static void test_selected_source_fail_closed_and_strict_values() {
    set_base_pressure_env();
    pressure_fixture fixture;
    fixture.write_proc_defaults();
    fixture.write_v2("/job", "/", "/sys/fs/cgroup", "/sys/fs/cgroup",
                     "9830400\n", "10240000\n", "9216000\n");
    kv_pressure_sampler sampler(fixture.paths());
    CHECK(sampler.init());
    CHECK_STATE(sampler.sample(), kv_pressure_state::CRITICAL);
    CHECK(sampler.telemetry().cgroup_high_kb == 9000);

    const char * invalid_current[] = {
        "0junk\n", "-1\n", "+1\n", "18446744073709551616\n", "max\n",
    };
    for (size_t i = 0; i < sizeof(invalid_current) / sizeof(invalid_current[0]); ++i) {
        fixture.write("/sys/fs/cgroup/job/memory.current", invalid_current[i]);
        CHECK_STATE(sampler.sample(), kv_pressure_state::CRITICAL);
        CHECK(!sampler.telemetry().sample_valid);
        CHECK(sampler.telemetry().stale);
    }

    fixture.write("/sys/fs/cgroup/job/memory.current", "6144000\n");
    fixture.remove("/sys/fs/cgroup/job/memory.max");
    CHECK_STATE(sampler.sample(), kv_pressure_state::CRITICAL);
    CHECK(!sampler.telemetry().sample_valid);
    CHECK(sampler.telemetry().stale);

    const char * invalid_maximum[] = {
        "10240000junk\n", "-1\n", "+1\n", "18446744073709551616\n", "MAX\n",
    };
    for (size_t i = 0; i < sizeof(invalid_maximum) / sizeof(invalid_maximum[0]); ++i) {
        fixture.write("/sys/fs/cgroup/job/memory.max", invalid_maximum[i]);
        CHECK_STATE(sampler.sample(), kv_pressure_state::CRITICAL);
        CHECK(!sampler.telemetry().sample_valid);
        CHECK(sampler.telemetry().stale);
    }

    fixture.write("/sys/fs/cgroup/job/memory.max", "10240000\n");
    fixture.write("/sys/fs/cgroup/job/memory.high", "7garbage\n");
    CHECK_STATE(sampler.sample(), kv_pressure_state::RECOVERY);
    CHECK(sampler.telemetry().sample_valid);
    CHECK(sampler.telemetry().cgroup_high_kb == 9000);

    fixture.remove("/sys/fs/cgroup/job/memory.high");
    CHECK_STATE(sampler.sample(), kv_pressure_state::NORMAL);
    CHECK(sampler.telemetry().sample_valid);
    CHECK(sampler.telemetry().cgroup_high_kb == 9000);

    fixture.write("/sys/fs/cgroup/job/memory.max", "max!\n");
    CHECK_STATE(sampler.sample(), kv_pressure_state::NORMAL);
    CHECK(!sampler.telemetry().sample_valid);
    CHECK(sampler.telemetry().stale);
    clear_pressure_env();
}

static void test_rss_strict_parse_and_page_overflow() {
    set_base_pressure_env();
    setenv("LLAMA_KV_LOW_WATER_RSS_KB", "100", 1);
    setenv("LLAMA_KV_PRESSURE_RSS_KB", "150", 1);
    setenv("LLAMA_KV_CRITICAL_RSS_KB", "250", 1);
    pressure_fixture fixture;
    fixture.write_proc_defaults("100 50 0 0 0 0 0\n");
    fixture.write("/proc/self/cgroup", "");
    fixture.write("/proc/self/mountinfo", "");
    kv_pressure_sampler sampler(fixture.paths());
    CHECK(sampler.init());
    CHECK(sampler.telemetry().source == kv_pressure_source::RSS_ABSOLUTE);
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);

    const char * invalid_statm[] = {
        "100 -1 0 0 0 0 0\n",
        "100 50junk 0 0 0 0 0\n",
        "100 18446744073709551616 0 0 0 0 0\n",
        "100 50 0 0 0 0 garbage\n",
    };
    for (size_t i = 0; i < sizeof(invalid_statm) / sizeof(invalid_statm[0]); ++i) {
        fixture.write("/proc/self/statm", invalid_statm[i]);
        CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
        CHECK(!sampler.telemetry().sample_valid);
        CHECK(sampler.telemetry().stale);
    }

    fixture.write("/proc/self/statm", "1 18446744073709551615 0 0 0 0 0\n");
    kv_pressure_sampler overflow(fixture.paths(4096));
    CHECK(overflow.init());
    CHECK_STATE(overflow.sample(), kv_pressure_state::NORMAL);
    CHECK(!overflow.telemetry().sample_valid);
    CHECK(overflow.telemetry().stale);
    clear_pressure_env();
}

static void test_dynamic_memory_max_reselects_source() {
    set_base_pressure_env();
    pressure_fixture fixture;
    fixture.write_proc_defaults();
    fixture.write_v2("/job", "/", "/sys/fs/cgroup", "/sys/fs/cgroup",
                     "8704000\n", "10240000\n", "max\n");
    kv_pressure_sampler sampler(fixture.paths());
    CHECK(sampler.init());
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);

    fixture.write("/sys/fs/cgroup/job/memory.max", "max\n");
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
    CHECK(sampler.telemetry().source == kv_pressure_source::NONE);
    CHECK(!sampler.transitions_enabled());
    CHECK(sampler.telemetry().sample_valid);

    fixture.write("/sys/fs/cgroup/job/memory.max", "max-junk\n");
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
    CHECK(!sampler.telemetry().sample_valid);
    CHECK(sampler.telemetry().stale);

    fixture.write("/sys/fs/cgroup/job/memory.current", "6144000\n");
    fixture.write("/sys/fs/cgroup/job/memory.max", "10240000\n");
    CHECK_STATE(sampler.sample(), kv_pressure_state::RECOVERY);
    CHECK(sampler.telemetry().source == kv_pressure_source::CGROUP_RATIO);
    CHECK(sampler.transitions_enabled());
    CHECK(sampler.telemetry().sample_valid);
    clear_pressure_env();
}

static void test_finite_max_finite_resets_transition_accumulation() {
    clear_pressure_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);

    pressure_fixture fixture;
    fixture.write_proc_defaults();
    fixture.write_v2("/job", "/", "/sys/fs/cgroup", "/sys/fs/cgroup",
                     "8704000\n", "10240000\n", "max\n");
    kv_pressure_sampler sampler(fixture.paths());
    CHECK(sampler.init());
    CHECK_STATE(sampler.sample(), kv_pressure_state::NORMAL);
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);

    fixture.write("/sys/fs/cgroup/job/memory.current", "6144000\n");
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);

    fixture.write("/sys/fs/cgroup/job/memory.max", "max\n");
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
    CHECK(sampler.telemetry().source == kv_pressure_source::NONE);

    fixture.write("/sys/fs/cgroup/job/memory.max", "10240000\n");
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
    CHECK(sampler.telemetry().source == kv_pressure_source::CGROUP_RATIO);
    CHECK_STATE(sampler.sample(), kv_pressure_state::RECOVERY);
    clear_pressure_env();
}

static void test_source_and_effective_threshold_changes_reset_accumulation() {
    clear_pressure_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);
    setenv("LLAMA_KV_LOW_WATER_RSS_KB", "700", 1);
    setenv("LLAMA_KV_PRESSURE_RSS_KB", "800", 1);
    setenv("LLAMA_KV_CRITICAL_RSS_KB", "950", 1);

    kv_pressure_sampler source_change;
    CHECK(source_change.init());
    const auto ratio_pressure = make_telemetry(
        true, 0, 8500, 10000, 0, 0, kv_pressure_source::CGROUP_RATIO);
    CHECK_STATE(source_change.sample_synthetic(ratio_pressure), kv_pressure_state::NORMAL);
    const auto rss_pressure = make_telemetry(
        true, 850, 0, 0, 0, 0, kv_pressure_source::RSS_ABSOLUTE);
    CHECK_STATE(source_change.sample_synthetic(rss_pressure), kv_pressure_state::NORMAL);
    CHECK_STATE(source_change.sample_synthetic(rss_pressure), kv_pressure_state::PRESSURE);

    clear_pressure_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "0", 1);
    kv_pressure_sampler threshold_change;
    CHECK(threshold_change.init());
    CHECK_STATE(threshold_change.sample_synthetic(
        make_telemetry(true, 0, 8500, 10000, 0, 0)), kv_pressure_state::NORMAL);
    const auto changed_maximum = make_telemetry(true, 0, 17000, 20000, 0, 0);
    CHECK_STATE(threshold_change.sample_synthetic(changed_maximum), kv_pressure_state::NORMAL);
    CHECK_STATE(threshold_change.sample_synthetic(changed_maximum), kv_pressure_state::PRESSURE);
    clear_pressure_env();
}

static void test_strict_psi_inputs() {
    clear_pressure_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1", 1);
    setenv("LLAMA_KV_PRESSURE_HYSTERESIS", "2", 1);
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "0", 1);
    setenv("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "1000", 1);

    pressure_fixture fixture;
    fixture.write_proc_defaults();
    fixture.write_v2("/job", "/", "/sys/fs/cgroup", "/sys/fs/cgroup",
                     "8704000\n", "10240000\n", "max\n");
    kv_pressure_sampler sampler(fixture.paths());
    CHECK(sampler.init());
    inject_n(sampler, make_telemetry(true, 0, 8500, 10000, 0, 0), 2);
    CHECK_STATE(sampler.state(), kv_pressure_state::PRESSURE);

    const char * invalid_psi[] = {
        "some avg10=15.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00 avg60=1.00 avg300=1.00 total=1\n"
        "some avg10=15.00 avg60=1.00 avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00 avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00 avg60=1.00 avg300=1.00 total=1 garbage\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=NaN avg60=1.00 avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=Inf avg60=1.00 avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=1e309 avg60=1.00 avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=100.01 avg60=1.00 avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00junk avg60=1.00 avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00 avg60=1.00junk avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00 avg60=1.00 avg300=1.00junk total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00 avg60=1.00 avg300=1.00 total=18446744073709551616\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00 avg60=1.00 avg300=1.00 total=1junk\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
        "some avg10=15.00 avg10=15.00 avg300=1.00 total=1\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n",
    };
    for (size_t i = 0; i < sizeof(invalid_psi) / sizeof(invalid_psi[0]); ++i) {
        fixture.write("/proc/pressure/memory", invalid_psi[i]);
        fixture.write("/sys/fs/cgroup/job/memory.pressure", invalid_psi[i]);
        CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
        CHECK(sampler.telemetry().psi_some_avg10 == 0);
        CHECK(sampler.telemetry().psi_full_avg10 == 0);
    }

    const std::string valid =
        "some avg10=15.00 avg60=1.00 avg300=1.00 total=18446744073709551615\n"
        "full avg10=1.00 avg60=1.00 avg300=1.00 total=1\n";
    fixture.write("/proc/pressure/memory", valid);
    fixture.write("/sys/fs/cgroup/job/memory.pressure", valid);
    CHECK_STATE(sampler.sample(), kv_pressure_state::PRESSURE);
    CHECK_STATE(sampler.sample(), kv_pressure_state::CRITICAL);
    clear_pressure_env();
}

static void test_saturating_counter_boundary() {
    CHECK(kv_pressure_sampler_test_access::increment(UINT32_MAX - 1) == UINT32_MAX);
    CHECK(kv_pressure_sampler_test_access::increment(UINT32_MAX) == UINT32_MAX);
}

static void check_invalid_single_env(const char * name, const char * value) {
    set_base_pressure_env();
    setenv(name, value, 1);
    kv_pressure_sampler sampler;
    CHECK(sampler.init());
    CHECK(!sampler.telemetry().config_valid);
    CHECK(!sampler.transitions_enabled());
    CHECK(std::strlen(sampler.telemetry().disabled_reason) > 0);
    clear_pressure_env();
}

static void test_invalid_configuration_disables_transitions() {
    check_invalid_single_env("LLAMA_KV_PRESSURE_RATIO", "8000junk");
    check_invalid_single_env("LLAMA_KV_PRESSURE_RATIO", "-1");
    check_invalid_single_env("LLAMA_KV_PRESSURE_RATIO", "10001");
    check_invalid_single_env("LLAMA_KV_PRESSURE_HYSTERESIS", "0");
    check_invalid_single_env("LLAMA_KV_PRESSURE_COOLDOWN_US", "18446744073709551616");
    check_invalid_single_env("LLAMA_KV_PRESSURE_COOLDOWN_US", "1us");
    check_invalid_single_env("LLAMA_KV_PSI_UPGRADE_THRESHOLD", "10001");

    clear_pressure_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "1junk", 1);
    kv_pressure_sampler invalid_enable;
    CHECK(!invalid_enable.init());
    CHECK(!invalid_enable.enabled());
    CHECK(!invalid_enable.telemetry().config_valid);
    CHECK(std::strlen(invalid_enable.telemetry().disabled_reason) > 0);

    set_base_pressure_env();
    setenv("LLAMA_KV_LOW_WATER_RATIO", "9000", 1);
    setenv("LLAMA_KV_PRESSURE_RATIO", "8000", 1);
    kv_pressure_sampler ratio_order;
    CHECK(ratio_order.init());
    CHECK(!ratio_order.transitions_enabled());
    CHECK(std::strstr(ratio_order.telemetry().disabled_reason, "low < pressure") != nullptr);

    set_base_pressure_env();
    setenv("LLAMA_KV_LOW_WATER_RSS_KB", "30", 1);
    setenv("LLAMA_KV_PRESSURE_RSS_KB", "20", 1);
    setenv("LLAMA_KV_CRITICAL_RSS_KB", "40", 1);
    kv_pressure_sampler absolute_order;
    CHECK(absolute_order.init());
    CHECK(!absolute_order.transitions_enabled());
    CHECK(std::strstr(absolute_order.telemetry().disabled_reason, "ordered") != nullptr);

    set_base_pressure_env();
    setenv("LLAMA_KV_PRESSURE_RSS_KB", "100", 1);
    kv_pressure_sampler incomplete_absolute;
    CHECK(incomplete_absolute.init());
    CHECK(!incomplete_absolute.transitions_enabled());
    CHECK(std::strstr(incomplete_absolute.telemetry().disabled_reason, "complete") != nullptr);

    set_base_pressure_env();
    setenv("LLAMA_KV_LOW_WATER_RSS_KB", "10", 1);
    setenv("LLAMA_KV_PRESSURE_RSS_KB", "20", 1);
    setenv("LLAMA_KV_CRITICAL_RSS_KB", "30", 1);
    setenv("LLAMA_KV_LOW_WATER_CGROUP_KB", "10", 1);
    setenv("LLAMA_KV_PRESSURE_CGROUP_KB", "20", 1);
    setenv("LLAMA_KV_CRITICAL_CGROUP_KB", "30", 1);
    kv_pressure_sampler ambiguous_absolute;
    CHECK(ambiguous_absolute.init());
    CHECK(!ambiguous_absolute.transitions_enabled());
    CHECK(std::strstr(ambiguous_absolute.telemetry().disabled_reason, "ambiguous") != nullptr);
    clear_pressure_env();
}

static void test_safe_ratio_and_cooldown_boundaries() {
    set_base_pressure_env();
    setenv("LLAMA_KV_PRESSURE_COOLDOWN_US", "18446744073709551615", 1);
    kv_pressure_sampler sampler;
    CHECK(sampler.init());
    const uint64_t largest = std::numeric_limits<uint64_t>::max();
    auto pressure = make_telemetry(true, 0, largest - largest / 10, largest, 0, 0);
    CHECK_STATE(sampler.sample_synthetic(pressure), kv_pressure_state::PRESSURE);
    auto low = make_telemetry(true, 0, largest / 2, largest, 0, 0);
    CHECK_STATE(sampler.sample_synthetic(low), kv_pressure_state::PRESSURE);

    auto critical = make_telemetry(true, 0, largest, largest, 0, 0);
    CHECK_STATE(sampler.sample_synthetic(critical), kv_pressure_state::CRITICAL);
    clear_pressure_env();
}

// ═════════════════════════════════════════════════════════════════════════════

int main() {
    test_default_off();
    test_full_cycle_cgroup_ratio();
    test_critical_bypasses_cooldown();
    test_critical_exit_hysteresis();
    test_threshold_jitter();
    test_max_unlimited_telemetry_only();
    test_rss_absolute_thresholds();
    test_psi_upgrade();
    test_psi_upgrade_resets_across_full_recovery_cycle();
    test_psi_alone_no_transition();
    test_low_water_wins_over_high_psi();
    test_sample_failure_stale();
    test_stale_recovery();
    test_cgroup_absolute_thresholds();
    test_telemetry_fields();
    test_config_edge_cases();
    test_recovery_reentry_pressure();
    test_critical_from_recovery();
    test_real_v2_resolution_non_root_and_escape();
    test_cgroup_ambiguity_disables_source();
    test_real_v1_resolution_non_root();
    test_selected_source_fail_closed_and_strict_values();
    test_rss_strict_parse_and_page_overflow();
    test_dynamic_memory_max_reselects_source();
    test_finite_max_finite_resets_transition_accumulation();
    test_source_and_effective_threshold_changes_reset_accumulation();
    test_strict_psi_inputs();
    test_saturating_counter_boundary();
    test_invalid_configuration_disables_transitions();
    test_safe_ratio_and_cooldown_boundaries();

    std::printf("\n=== Results ===\n");
    std::printf("Total:  %d\n", tests_total);
    std::printf("Passed: %d\n", tests_total - tests_failed);
    std::printf("Failed: %d\n", tests_failed);

    if (tests_failed > 0) {
        std::fprintf(stderr, "\nFAIL: %d/%d tests failed\n", tests_failed, tests_total);
        return 1;
    }

    std::printf("PASS: all kv_pressure_sampler tests\n");
    return 0;
}
