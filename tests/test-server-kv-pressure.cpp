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

    std::printf("server KV pressure tests: %d/%d passed\n", tests_total - tests_failed, tests_total);
    return tests_failed == 0 ? 0 : 1;
}
