#include "server-kv-pressure.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <unistd.h>

static int tests_total = 0;
static int tests_failed = 0;

static bool check(bool condition, const char * message) {
    ++tests_total;
    if (!condition) {
        ++tests_failed;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

#define CHECK(condition) check((condition), #condition)

static void clear_env() {
    unsetenv("LLAMA_KV_PRESSURE_SAMPLER");
    unsetenv("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS");
    unsetenv("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS");
}

static kv_pressure_telemetry make_telemetry(
        kv_pressure_state state = kv_pressure_state::NORMAL,
        kv_pressure_source source = kv_pressure_source::CGROUP_RATIO,
        bool stale = false) {
    kv_pressure_telemetry telemetry;
    telemetry.enabled = true;
    telemetry.state = state;
    telemetry.previous_state = kv_pressure_state::NORMAL;
    telemetry.source = source;
    telemetry.sample_valid = !stale;
    telemetry.stale = stale;
    telemetry.config_valid = true;
    telemetry.rss_kb = 101;
    telemetry.cgroup_current_bytes = 202 * 1024;
    telemetry.cgroup_max_bytes = 303 * 1024;
    telemetry.cgroup_current_kb = 202;
    telemetry.cgroup_max_kb = 303;
    telemetry.cgroup_high_kb = 250;
    telemetry.psi_some_avg10 = 404;
    telemetry.psi_full_avg10 = 505;
    telemetry.sample_latency_ns = 606;
    return telemetry;
}

static void test_default_off() {
    clear_env();
    CHECK(kv_pressure_sampler_environment_enablement() ==
          kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_DISABLED);

    server_kv_pressure_runtime runtime;
    CHECK(!runtime.sample_due(server_kv_pressure_runtime::time_point {}));
    CHECK(runtime.sample_count() == 0);
    CHECK(runtime.skip_count() == 0);
}

static void test_interval_config() {
    clear_env();
    server_kv_pressure_config config;
    std::string error;

    CHECK(server_kv_pressure_config_from_env(config, error));
    CHECK(config.sample_interval == std::chrono::milliseconds(250));
    CHECK(config.log_interval == std::chrono::milliseconds(60000));

    setenv("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS", "99", 1);
    setenv("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS", "500", 1);
    CHECK(server_kv_pressure_config_from_env(config, error));
    CHECK(config.sample_interval == std::chrono::milliseconds(100));
    CHECK(config.log_interval == std::chrono::milliseconds(1000));

    setenv("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS", " 250 ", 1);
    setenv("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS", " 60000 ", 1);
    CHECK(server_kv_pressure_config_from_env(config, error));
    CHECK(config.sample_interval == std::chrono::milliseconds(250));
    CHECK(config.log_interval == std::chrono::milliseconds(60000));

    setenv("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS", "100ms", 1);
    CHECK(!server_kv_pressure_config_from_env(config, error));
    CHECK(!error.empty());

    setenv("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS", "9223372036854775807", 1);
    CHECK(!server_kv_pressure_config_from_env(config, error));
    clear_env();
}

static void test_rate_limit_and_idle_path() {
    server_kv_pressure_config config;
    server_kv_pressure_runtime runtime;
    runtime.enable(config);

    const auto start = server_kv_pressure_runtime::time_point {};
    int calls = 0;
    const auto sample = [&]() {
        ++calls;
        return make_telemetry();
    };

    CHECK(runtime.sample_due(start));
    const auto first = runtime.record_sample(start, true, sample());
    CHECK(first.idle);
    CHECK(first.first_sample);
    CHECK(first.source_changed);
    CHECK(first.should_log());
    CHECK(server_kv_pressure_format_marker(first).find("trigger=first,source") != std::string::npos);
    CHECK(calls == 1);

    CHECK(!runtime.sample_due(start + std::chrono::milliseconds(249)));
    CHECK(calls == 1);
    CHECK(runtime.sample_count() == 1);
    CHECK(runtime.skip_count() == 1);

    CHECK(runtime.sample_due(start + std::chrono::milliseconds(250)));
    const auto second = runtime.record_sample(
            start + std::chrono::milliseconds(250), false, sample());
    CHECK(!second.idle);
    CHECK(!second.first_sample);
    CHECK(!second.should_log());
    CHECK(calls == 2);
    CHECK(runtime.sample_count() == 2);
    CHECK(runtime.skip_count() == 1);
}

static void test_first_critical_and_stale_logging() {
    server_kv_pressure_runtime runtime;
    runtime.enable(server_kv_pressure_config {});

    const auto event = runtime.record_sample(
            server_kv_pressure_runtime::time_point {}, true,
            make_telemetry(kv_pressure_state::CRITICAL, kv_pressure_source::RSS_ABSOLUTE, true));
    CHECK(event.state_changed);
    CHECK(event.source_changed);
    CHECK(event.stale_changed);
    CHECK(event.should_log());
}

static void test_change_and_periodic_logging() {
    server_kv_pressure_config config;
    config.sample_interval = std::chrono::milliseconds(100);
    config.log_interval = std::chrono::milliseconds(1000);

    server_kv_pressure_runtime runtime;
    runtime.enable(config);
    const auto start = server_kv_pressure_runtime::time_point {};

    kv_pressure_telemetry telemetry = make_telemetry(
            kv_pressure_state::NORMAL, kv_pressure_source::NONE, false);
    const auto first = runtime.record_sample(start, false, telemetry);
    CHECK(first.first_sample);
    CHECK(first.should_log());

    auto unchanged = runtime.record_sample(start + std::chrono::milliseconds(100), false, telemetry);
    CHECK(!unchanged.should_log());

    telemetry.state = kv_pressure_state::PRESSURE;
    auto state = runtime.record_sample(start + std::chrono::milliseconds(200), false, telemetry);
    CHECK(state.state_changed);
    CHECK(state.should_log());

    telemetry.source = kv_pressure_source::RSS_ABSOLUTE;
    auto source = runtime.record_sample(start + std::chrono::milliseconds(300), false, telemetry);
    CHECK(source.source_changed);
    CHECK(source.should_log());

    telemetry.stale = true;
    telemetry.sample_valid = false;
    auto stale = runtime.record_sample(start + std::chrono::milliseconds(400), false, telemetry);
    CHECK(stale.stale_changed);
    CHECK(stale.should_log());

    auto periodic = runtime.record_sample(start + std::chrono::milliseconds(1400), true, telemetry);
    CHECK(periodic.periodic);
    CHECK(periodic.should_log());
}

static void test_completion_based_deadline_and_overflow() {
    server_kv_pressure_config config;
    server_kv_pressure_runtime runtime;
    runtime.enable(config);

    const auto completed = server_kv_pressure_runtime::time_point {} + std::chrono::seconds(5);
    runtime.record_sample(completed, false, make_telemetry());
    CHECK(!runtime.sample_due(completed + std::chrono::milliseconds(249)));
    CHECK(runtime.sample_due(completed + std::chrono::milliseconds(250)));

    runtime.enable(config);
    const auto near_max = server_kv_pressure_runtime::time_point::max() - std::chrono::milliseconds(50);
    runtime.record_sample(near_max, false, make_telemetry());
    CHECK(!runtime.sample_due(server_kv_pressure_runtime::time_point::max() - std::chrono::milliseconds(1)));
    CHECK(runtime.sample_due(server_kv_pressure_runtime::time_point::max()));
}

static void test_lifecycle_reset() {
    server_kv_pressure_runtime runtime;
    runtime.enable(server_kv_pressure_config {});

    const auto old_now = server_kv_pressure_runtime::time_point {} + std::chrono::seconds(10);
    runtime.record_sample(old_now, false,
            make_telemetry(kv_pressure_state::CRITICAL, kv_pressure_source::RSS_ABSOLUTE, true));
    CHECK(runtime.sample_count() == 1);

    runtime.disable();
    CHECK(!runtime.enabled());
    CHECK(runtime.sample_count() == 0);
    CHECK(runtime.skip_count() == 0);

    runtime.enable(server_kv_pressure_config {});
    CHECK(runtime.sample_due(server_kv_pressure_runtime::time_point {}));
    const auto first = runtime.record_sample(
            server_kv_pressure_runtime::time_point {}, false, make_telemetry());
    CHECK(first.first_sample);
    CHECK(first.sample_count == 1);
    CHECK(first.skip_count == 0);
    CHECK(first.should_log());
}

static void test_marker_fields() {
    server_kv_pressure_event event;
    event.telemetry = make_telemetry(
            kv_pressure_state::PRESSURE, kv_pressure_source::CGROUP_ABSOLUTE, true);
    event.telemetry.previous_state = kv_pressure_state::NORMAL;
    event.sample_count = 7;
    event.skip_count = 11;
    event.idle = true;
    event.state_changed = true;
    event.stale_changed = true;

    const std::string marker = server_kv_pressure_format_marker(event);
    CHECK(marker.find("kv_pressure_telemetry") != std::string::npos);
    CHECK(marker.find("state=PRESSURE") != std::string::npos);
    CHECK(marker.find("previous_state=NORMAL") != std::string::npos);
    CHECK(marker.find("source=CGROUP_ABSOLUTE") != std::string::npos);
    CHECK(marker.find("sample_valid=0") != std::string::npos);
    CHECK(marker.find("stale=1") != std::string::npos);
    CHECK(marker.find("rss_kb=101") != std::string::npos);
    CHECK(marker.find("cgroup_current_bytes=206848") != std::string::npos);
    CHECK(marker.find("cgroup_max_bytes=310272") != std::string::npos);
    CHECK(marker.find("cgroup_current_kb=202") != std::string::npos);
    CHECK(marker.find("cgroup_max_kb=303") != std::string::npos);
    CHECK(marker.find("psi_some_avg10=404") != std::string::npos);
    CHECK(marker.find("psi_full_avg10=505") != std::string::npos);
    CHECK(marker.find("sample_latency_ns=606") != std::string::npos);
    CHECK(marker.find("sample_count=7") != std::string::npos);
    CHECK(marker.find("skip_count=11") != std::string::npos);
    CHECK(marker.find("idle=1") != std::string::npos);
    CHECK(marker.find("trigger=state,stale") != std::string::npos);
}

static void test_init_failure_stays_disabled() {
    clear_env();
    setenv("LLAMA_KV_PRESSURE_SAMPLER", "invalid", 1);
    const auto enablement = kv_pressure_sampler_environment_enablement();
    CHECK(enablement == kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_INVALID);

    kv_pressure_sampler sampler;
    server_kv_pressure_runtime runtime;
    server_kv_pressure_config config;
    if (sampler.init(enablement)) {
        runtime.enable(config);
    }

    CHECK(!runtime.enabled());
    CHECK(!runtime.sample_due(server_kv_pressure_runtime::time_point {}));
    clear_env();
}

// --- dry-run policy tests -----------------------------------------------------

static void test_dry_run_normal_recovery_not_due() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_dry_run_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.dry_run_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();

    // NORMAL: dry_run_due must return false
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::NORMAL, false));

    // RECOVERY: dry_run_due must return false
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::RECOVERY, false));

    // stale: false regardless of state
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::CRITICAL, true));
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::PRESSURE, true));
}

static void test_dry_run_pressure_critical_entry() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_dry_run_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.dry_run_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();

    // State entry to PRESSURE: immediate evaluation
    CHECK(runtime.dry_run_due(t0, kv_pressure_state::PRESSURE, false));

    // Record at evaluation time t0
    runtime.dry_run_record(false, t0);

    // 500 ms after: cooldown not elapsed → false
    auto t1 = t0 + std::chrono::milliseconds(500);
    CHECK(!runtime.dry_run_due(t1, kv_pressure_state::PRESSURE, false));

    // After cooldown: true again
    auto t2 = t0 + std::chrono::milliseconds(2500);
    CHECK(runtime.dry_run_due(t2, kv_pressure_state::PRESSURE, false));
}

static void test_dry_run_critical_entry_immediate() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_dry_run_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.dry_run_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();

    // State entry to CRITICAL: bypass cooldown → immediate
    CHECK(runtime.dry_run_due(t0, kv_pressure_state::CRITICAL, false));

    runtime.dry_run_record(false, t0);

    // Within cooldown → false
    auto t1 = t0 + std::chrono::milliseconds(500);
    CHECK(!runtime.dry_run_due(t1, kv_pressure_state::CRITICAL, false));
}

static void test_dry_run_cooldown_blocking() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_dry_run_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.dry_run_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();

    // Entry evaluation
    CHECK(runtime.dry_run_due(t0, kv_pressure_state::PRESSURE, false));
    runtime.dry_run_record(false, t0);

    // 500 ms: still within 2000ms cooldown → blocked
    auto t1 = t0 + std::chrono::milliseconds(500);
    CHECK(!runtime.dry_run_due(t1, kv_pressure_state::PRESSURE, false));

    // 2001 ms: cooldown elapsed → allowed
    auto t2 = t0 + std::chrono::milliseconds(2001);
    CHECK(runtime.dry_run_due(t2, kv_pressure_state::PRESSURE, false));

    runtime.dry_run_record(false, t2);

    // 500 ms after second eval: cooldown not elapsed → blocked
    auto t3 = t2 + std::chrono::milliseconds(500);
    CHECK(!runtime.dry_run_due(t3, kv_pressure_state::PRESSURE, false));
}

static void test_dry_run_shortfall_backoff() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_dry_run_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.dry_run_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();

    // Entry evaluation
    CHECK(runtime.dry_run_due(t0, kv_pressure_state::CRITICAL, false));
    runtime.dry_run_record(true, t0);  // shortfall → backoff

    // 2001 ms: still within 10s backoff → blocked
    auto t1 = t0 + std::chrono::milliseconds(2001);
    CHECK(!runtime.dry_run_due(t1, kv_pressure_state::CRITICAL, false));

    // 5000 ms: still within 10s backoff → blocked
    auto t2 = t0 + std::chrono::milliseconds(5000);
    CHECK(!runtime.dry_run_due(t2, kv_pressure_state::CRITICAL, false));

    // 10001 ms: backoff elapsed → allowed
    auto t3 = t0 + std::chrono::milliseconds(10001);
    CHECK(runtime.dry_run_due(t3, kv_pressure_state::CRITICAL, false));
}

static void test_dry_run_state_transition_resets_cooldown() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_dry_run_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.dry_run_enable(cfg);

    const auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(runtime.dry_run_due(t0, kv_pressure_state::PRESSURE, false));
    runtime.dry_run_record(true, t0);
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::CRITICAL, false));
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::RECOVERY, false));
    CHECK(runtime.dry_run_due(t0, kv_pressure_state::CRITICAL, false));
}


static void test_dry_run_normal_exit_clears_backoff() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_dry_run_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.dry_run_enable(cfg);

    const auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(runtime.dry_run_due(t0, kv_pressure_state::PRESSURE, false));
    runtime.dry_run_record(true, t0);
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::NORMAL, false));
    CHECK(runtime.dry_run_due(t0, kv_pressure_state::PRESSURE, false));
}

static void test_dry_run_config_disabled() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_dry_run_config cfg;
    cfg.enabled = false;
    cfg.target_bytes = 0;
    runtime.dry_run_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();

    // Disabled: dry_run_due always false
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::PRESSURE, false));
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::CRITICAL, false));
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::NORMAL, false));

    // Enable with target_bytes=0 → effectively disabled
    cfg.enabled = true;
    cfg.target_bytes = 0;
    runtime.dry_run_enable(cfg);
    CHECK(!runtime.dry_run_due(t0, kv_pressure_state::CRITICAL, false));
}

// --- bounded release config tests --------------------------------------------

static void clear_bounded_release_env() {
    unsetenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE");
    unsetenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES");
    unsetenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS");
    unsetenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS");
    unsetenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS");
}

static void test_bounded_release_config_default_disabled() {
    clear_bounded_release_env();
    server_kv_pressure_bounded_release_config cfg;
    std::string error;
    CHECK(server_kv_pressure_bounded_release_config_from_env(cfg, error));
    CHECK(!cfg.enabled);
    CHECK(cfg.target_bytes == 0);
}


static void test_bounded_release_config_enabled_defaults_preserved() {
    clear_bounded_release_env();
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE", "1", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES", "1048576", 1);

    server_kv_pressure_bounded_release_config cfg;
    std::string error;
    CHECK(server_kv_pressure_bounded_release_config_from_env(cfg, error));
    CHECK(cfg.enabled);
    CHECK(cfg.max_scan_blocks == server_kv_pressure_bounded_release_config::DEFAULT_MAX_SCAN_BLOCKS);
    CHECK(cfg.cooldown_ms == server_kv_pressure_bounded_release_config::DEFAULT_COOLDOWN_MS);
    CHECK(cfg.backoff_ms == server_kv_pressure_bounded_release_config::DEFAULT_BACKOFF_MS);
    clear_bounded_release_env();
}

static void test_bounded_release_config_explicit_boundaries_and_invalid_values() {
    clear_bounded_release_env();
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE", "1", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES", "1", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS", "0", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS", "0", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS", "500", 1);

    server_kv_pressure_bounded_release_config cfg;
    std::string error;
    CHECK(server_kv_pressure_bounded_release_config_from_env(cfg, error));
    CHECK(cfg.max_scan_blocks == 1);
    CHECK(cfg.cooldown_ms == server_kv_pressure_bounded_release_config::MIN_COOLDOWN_MS);
    CHECK(cfg.backoff_ms == server_kv_pressure_bounded_release_config::MIN_COOLDOWN_MS);

    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS", "4294967296", 1);
    error.clear();
    CHECK(!server_kv_pressure_bounded_release_config_from_env(cfg, error));
    CHECK(!error.empty());
    clear_bounded_release_env();
}

static void test_bounded_release_config_valid() {
    clear_bounded_release_env();
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE", "1", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES", "33554432", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS", "128", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS", "3000", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS", "20000", 1);

    server_kv_pressure_bounded_release_config cfg;
    std::string error;
    CHECK(server_kv_pressure_bounded_release_config_from_env(cfg, error));
    CHECK(cfg.enabled);
    CHECK(cfg.target_bytes == 33554432);
    CHECK(cfg.max_scan_blocks == 128);
    CHECK(cfg.cooldown_ms == 3000);
    CHECK(cfg.backoff_ms == 20000);
    clear_bounded_release_env();
}

static void test_bounded_release_config_target_zero_disables() {
    clear_bounded_release_env();
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE", "1", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES", "0", 1);

    server_kv_pressure_bounded_release_config cfg;
    std::string error;
    CHECK(server_kv_pressure_bounded_release_config_from_env(cfg, error));
    CHECK(!cfg.enabled);  // zero target → effectively disabled
    clear_bounded_release_env();
}

static void test_bounded_release_config_invalid_target() {
    clear_bounded_release_env();
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE", "1", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES", "not_a_number", 1);

    server_kv_pressure_bounded_release_config cfg;
    std::string error;
    CHECK(!server_kv_pressure_bounded_release_config_from_env(cfg, error));
    CHECK(!error.empty());
    clear_bounded_release_env();
}

static void test_bounded_release_config_cooldown_min_clamped() {
    clear_bounded_release_env();
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE", "1", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES", "1048576", 1);
    setenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS", "100", 1);

    server_kv_pressure_bounded_release_config cfg;
    std::string error;
    CHECK(server_kv_pressure_bounded_release_config_from_env(cfg, error));
    CHECK(cfg.cooldown_ms == 500);  // clamped to MIN_COOLDOWN_MS
    clear_bounded_release_env();
}

// --- bounded release cooldown / state-entry tests -----------------------------

static void test_bounded_release_normal_recovery_not_due() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.bounded_release_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::NORMAL, false));
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::RECOVERY, false));
}

static void test_bounded_release_pressure_critical_entry() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.bounded_release_enable(cfg);

    const auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));
    runtime.bounded_release_record(false, t0);
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::CRITICAL, false));
    CHECK(runtime.bounded_release_due(t0 + std::chrono::milliseconds(2000),
            kv_pressure_state::CRITICAL, false));
}

static void test_bounded_release_cooldown_blocking() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.bounded_release_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));

    // Record a successful (non-shortfall) evaluation
    runtime.bounded_release_record(false, t0);

    // Immediately after: blocked by cooldown
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));

    // After cooldown: allows evaluation
    auto t2 = t0 + std::chrono::milliseconds(2000);
    CHECK(runtime.bounded_release_due(t2, kv_pressure_state::PRESSURE, false));
}

static void test_bounded_release_shortfall_backoff() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.bounded_release_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));

    // Record a shortfall evaluation
    runtime.bounded_release_record(true, t0);

    // Immediately after: blocked by backoff (longer than cooldown)
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));

    // After cooldown but before backoff: still blocked
    auto t2 = t0 + std::chrono::milliseconds(2000);
    CHECK(!runtime.bounded_release_due(t2, kv_pressure_state::PRESSURE, false));

    // After backoff: allows
    auto t3 = t0 + std::chrono::milliseconds(10000);
    CHECK(runtime.bounded_release_due(t3, kv_pressure_state::PRESSURE, false));
}

static void test_bounded_release_state_transition_resets_cooldown() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.bounded_release_enable(cfg);

    const auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));
    runtime.bounded_release_record(true, t0);
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::CRITICAL, false));
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::RECOVERY, false));
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));
}


static void test_bounded_release_episode_exit_clears_backoff() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.bounded_release_enable(cfg);

    const auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));
    runtime.bounded_release_record(true, t0);
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::NORMAL, false));
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));

    runtime.bounded_release_record(true, t0);
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::CRITICAL, false));
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::RECOVERY, false));
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::CRITICAL, false));
}

static void test_bounded_release_same_episode_transition_keeps_backoff() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.bounded_release_enable(cfg);

    const auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));
    runtime.bounded_release_record(true, t0);
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::CRITICAL, false));
    CHECK(!runtime.bounded_release_due(t0 + std::chrono::milliseconds(2000),
            kv_pressure_state::PRESSURE, false));
    CHECK(runtime.bounded_release_due(t0 + std::chrono::milliseconds(10000),
            kv_pressure_state::CRITICAL, false));
}

static void test_bounded_release_config_disabled() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = false;
    cfg.target_bytes = 0;
    runtime.bounded_release_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, false));
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::CRITICAL, false));
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::NORMAL, false));

    cfg.enabled = true;
    cfg.target_bytes = 0;
    runtime.bounded_release_enable(cfg);
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::CRITICAL, false));
}

static void test_bounded_release_stale_rejection() {
    server_kv_pressure_runtime runtime;
    server_kv_pressure_bounded_release_config cfg;
    cfg.enabled = true;
    cfg.target_bytes = 4194304;
    cfg.cooldown_ms = 2000;
    cfg.backoff_ms = 10000;
    runtime.bounded_release_enable(cfg);

    auto t0 = server_kv_pressure_runtime::clock::now();
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::PRESSURE, true));
    CHECK(!runtime.bounded_release_due(t0, kv_pressure_state::CRITICAL, true));
}

static void test_bounded_release_marker_format() {
    server_kv_pressure_bounded_release_event event;
    event.pressure_state = kv_pressure_state::CRITICAL;
    event.pressure_source = kv_pressure_source::RSS_ABSOLUTE;
    event.result.released_bytes = 1048576;
    event.result.released_blocks = 4;
    event.result.blocks_scanned = 10;
    event.result.blocks_skipped_owned = 3;
    event.result.blocks_skipped_state = 2;
    event.result.madvise_failures = 0;
    event.result.shortfall_bytes = 0;
    event.result.overshoot_bytes = 512;
    event.result.block_scan_exhausted = false;
    event.result.ownership_aborted = false;
    event.target_bytes = 1048576;
    event.max_scan_blocks = 64;
    event.legacy_enabled = false;
    event.sample_count = 7;
    event.episode = 2;
    event.cooldown_ms = 2000;
    event.skipped_reason = "none";
    event.idle = false;
    event.stale = false;

    const std::string marker =
        server_kv_pressure_bounded_release_format_marker(event);
    CHECK(marker.find("kv_pressure_bounded_release") != std::string::npos);
    CHECK(marker.find("state=CRITICAL") != std::string::npos);
    CHECK(marker.find("released_bytes=1048576") != std::string::npos);
    CHECK(marker.find("released_blocks=4") != std::string::npos);
    CHECK(marker.find("blocks_scanned=10") != std::string::npos);
    CHECK(marker.find("blocks_skipped_owned=3") != std::string::npos);
    CHECK(marker.find("blocks_skipped_state=2") != std::string::npos);
    CHECK(marker.find("madvise_failures=0") != std::string::npos);
    CHECK(marker.find("ownership_aborted=0") != std::string::npos);
    CHECK(marker.find("legacy_enabled=0") != std::string::npos);
    CHECK(marker.find("episode=2") != std::string::npos);
    CHECK(marker.find("skipped_reason=none") != std::string::npos);
}

int main() {
    test_default_off();
    test_interval_config();
    test_rate_limit_and_idle_path();
    test_first_critical_and_stale_logging();
    test_change_and_periodic_logging();
    test_completion_based_deadline_and_overflow();
    test_lifecycle_reset();
    test_marker_fields();
    test_init_failure_stays_disabled();
    test_dry_run_normal_recovery_not_due();
    test_dry_run_pressure_critical_entry();
    test_dry_run_critical_entry_immediate();
    test_dry_run_cooldown_blocking();
    test_dry_run_shortfall_backoff();
    test_dry_run_state_transition_resets_cooldown();
    test_dry_run_normal_exit_clears_backoff();
    test_dry_run_config_disabled();
    test_bounded_release_config_default_disabled();
    test_bounded_release_config_enabled_defaults_preserved();
    test_bounded_release_config_explicit_boundaries_and_invalid_values();
    test_bounded_release_config_valid();
    test_bounded_release_config_target_zero_disables();
    test_bounded_release_config_invalid_target();
    test_bounded_release_config_cooldown_min_clamped();
    test_bounded_release_normal_recovery_not_due();
    test_bounded_release_pressure_critical_entry();
    test_bounded_release_cooldown_blocking();
    test_bounded_release_shortfall_backoff();
    test_bounded_release_state_transition_resets_cooldown();
    test_bounded_release_episode_exit_clears_backoff();
    test_bounded_release_same_episode_transition_keeps_backoff();
    test_bounded_release_config_disabled();
    test_bounded_release_stale_rejection();
    test_bounded_release_marker_format();

    std::printf("server KV pressure tests: %d/%d passed\n", tests_total - tests_failed, tests_total);
    return tests_failed == 0 ? 0 : 1;
}
