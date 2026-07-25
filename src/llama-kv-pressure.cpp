#include "llama-kv-pressure.h"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cctype>
#include <cmath>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

// ── helpers ─────────────────────────────────────────────────────────────────

static std::string trim_ascii_space(const std::string & text) {
    size_t first = 0;
    while (first < text.size() && std::isspace((unsigned char) text[first])) {
        ++first;
    }
    size_t last = text.size();
    while (last > first && std::isspace((unsigned char) text[last - 1])) {
        --last;
    }
    return text.substr(first, last - first);
}

static bool parse_decimal_uint64(const std::string & text, uint64_t & value) {
    const std::string token = trim_ascii_space(text);
    if (token.empty()) {
        return false;
    }

    uint64_t result = 0;
    for (size_t i = 0; i < token.size(); ++i) {
        const unsigned char ch = (unsigned char) token[i];
        if (!std::isdigit(ch)) {
            return false;
        }
        const uint64_t digit = (uint64_t) (ch - '0');
        if (result > (UINT64_MAX - digit) / 10) {
            return false;
        }
        result = result * 10 + digit;
    }
    value = result;
    return true;
}

static bool parse_env_bool(const char * name, bool default_val, bool & value) {
    const char * val = std::getenv(name);
    if (!val) {
        value = default_val;
        return true;
    }
    if (std::strcmp(val, "1") == 0 || std::strcmp(val, "true") == 0 || std::strcmp(val, "yes") == 0) {
        value = true;
        return true;
    }
    if (std::strcmp(val, "0") == 0 || std::strcmp(val, "false") == 0 || std::strcmp(val, "no") == 0) {
        value = false;
        return true;
    }
    return false;
}

static bool parse_env_uint32(const char * name, uint32_t default_val, uint32_t & value) {
    const char * val = std::getenv(name);
    if (!val) {
        value = default_val;
        return true;
    }
    uint64_t parsed = 0;
    if (!parse_decimal_uint64(val, parsed) || parsed > UINT32_MAX) {
        return false;
    }
    value = (uint32_t) parsed;
    return true;
}

static bool parse_env_uint64(const char * name, uint64_t default_val, uint64_t & value) {
    const char * val = std::getenv(name);
    if (!val) {
        value = default_val;
        return true;
    }
    if (!parse_decimal_uint64(val, value)) {
        return false;
    }
    return true;
}

// ── public environment / name helpers ───────────────────────────────────────

kv_pressure_enablement kv_pressure_sampler_environment_enablement() {
    bool enabled = false;
    if (!parse_env_bool("LLAMA_KV_PRESSURE_SAMPLER", false, enabled)) {
        return kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_INVALID;
    }
    return enabled ? kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_ENABLED : kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_DISABLED;
}

const char * kv_pressure_state_name(kv_pressure_state state) {
    switch (state) {
        case kv_pressure_state::NORMAL:    return "NORMAL";
        case kv_pressure_state::PRESSURE:  return "PRESSURE";
        case kv_pressure_state::CRITICAL:  return "CRITICAL";
        case kv_pressure_state::RECOVERY:  return "RECOVERY";
    }
    return "UNKNOWN";
}

const char * kv_pressure_source_name(kv_pressure_source source) {
    switch (source) {
        case kv_pressure_source::NONE:            return "NONE";
        case kv_pressure_source::CGROUP_RATIO:    return "CGROUP_RATIO";
        case kv_pressure_source::RSS_ABSOLUTE:    return "RSS_ABSOLUTE";
        case kv_pressure_source::CGROUP_ABSOLUTE: return "CGROUP_ABSOLUTE";
    }
    return "UNKNOWN";
}

// ── file reading (no shell) ─────────────────────────────────────────────────

bool kv_pressure_sampler::read_file_text(const std::string & path, std::string & value) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    char buf[4096];
    value.clear();
    bool too_large = false;
    while (true) {
        const size_t count = std::fread(buf, 1, sizeof(buf), f);
        if (count > 0) {
            if (value.size() > (UINT64_C(1) << 20) - count) {
                too_large = true;
                break;
            }
            value.append(buf, count);
        }
        if (count < sizeof(buf)) {
            break;
        }
    }
    const bool read_error = std::ferror(f) != 0;
    const bool close_ok = std::fclose(f) == 0;
    if (read_error || too_large || !close_ok) {
        return false;
    }
    return true;
}

bool kv_pressure_sampler::parse_uint64(const std::string & text, uint64_t & value) {
    return parse_decimal_uint64(text, value);
}

kv_pressure_sampler::limit_kind kv_pressure_sampler::parse_limit(const std::string & text, uint64_t & value) {
    const std::string token = trim_ascii_space(text);
    if (token == "max") {
        value = 0;
        return limit_kind::UNLIMITED;
    }
    return parse_uint64(token, value) ? limit_kind::FINITE : limit_kind::INVALID;
}

bool kv_pressure_sampler::checked_add(uint64_t a, uint64_t b, uint64_t & result) {
    if (a > UINT64_MAX - b) {
        return false;
    }
    result = a + b;
    return true;
}

bool kv_pressure_sampler::checked_mul(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > UINT64_MAX / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool kv_pressure_sampler::ratio_at_least(uint64_t current, uint64_t maximum, uint32_t basis_points) {
    if (maximum == 0 || basis_points > 10000) {
        return false;
    }
    const uint64_t quotient = maximum / 10000;
    const uint64_t remainder = maximum % 10000;
    uint64_t whole = 0;
    uint64_t fractional = 0;
    uint64_t numerator = 0;
    if (!checked_mul(quotient, basis_points, whole) ||
        !checked_mul(remainder, basis_points, numerator)) {
        return false;
    }
    fractional = numerator / 10000 + (numerator % 10000 != 0 ? 1 : 0);
    uint64_t threshold = 0;
    return checked_add(whole, fractional, threshold) && current >= threshold;
}

// ── RSS ─────────────────────────────────────────────────────────────────────

bool kv_pressure_sampler::read_rss_kb(uint64_t & rss_kb) const {
#ifdef __linux__
    std::string text;
    if (!read_file_text(paths_.proc_root + "/self/statm", text)) {
        return false;
    }

    std::vector<std::string> fields;
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && std::isspace((unsigned char) text[pos])) {
            ++pos;
        }
        if (pos == text.size()) {
            break;
        }
        const size_t start = pos;
        while (pos < text.size() && !std::isspace((unsigned char) text[pos])) {
            ++pos;
        }
        fields.push_back(text.substr(start, pos - start));
    }
    if (fields.size() < 2) {
        return false;
    }
    uint64_t pages_total = 0;
    uint64_t pages_rss = 0;
    if (!parse_uint64(fields[0], pages_total) || !parse_uint64(fields[1], pages_rss)) {
        return false;
    }
    for (size_t i = 2; i < fields.size(); ++i) {
        uint64_t ignored = 0;
        if (!parse_uint64(fields[i], ignored)) {
            return false;
        }
    }
    (void) pages_total;

    uint64_t page_size = paths_.page_size;
    if (page_size == 0) {
        const long system_page_size = sysconf(_SC_PAGESIZE);
        if (system_page_size <= 0) {
            return false;
        }
        page_size = (uint64_t) system_page_size;
    }
    // Compute floor(pages_rss * page_size / 1024) without forming the
    // potentially overflowing product.
    uint64_t whole_pages_bytes = 0;
    uint64_t remainder_whole_kb = 0;
    uint64_t partial = 0;
    uint64_t result = 0;
    if (!checked_mul(pages_rss / 1024, page_size, whole_pages_bytes) ||
        !checked_mul(pages_rss % 1024, page_size / 1024, remainder_whole_kb) ||
        !checked_add(whole_pages_bytes, remainder_whole_kb, partial)) {
        return false;
    }
    const uint64_t fractional_kb = ((pages_rss % 1024) * (page_size % 1024)) / 1024;
    if (!checked_add(partial, fractional_kb, result)) {
        return false;
    }
    rss_kb = result;
    return true;
#else
    (void) rss_kb;
    return false;
#endif
}

// ── cgroup memory ───────────────────────────────────────────────────────────

bool kv_pressure_sampler::read_cgroup_current(uint64_t & current_bytes) const {
    if (cgroup_mem_path_.empty()) {
        return false;
    }
    const char * file = cgroup_version_ == 2 ? "/memory.current" : "/memory.usage_in_bytes";
    std::string text;
    uint64_t bytes = 0;
    if (!read_file_text(cgroup_mem_path_ + file, text) || !parse_uint64(text, bytes)) {
        return false;
    }
    current_bytes = bytes;
    return true;
}

kv_pressure_sampler::limit_kind kv_pressure_sampler::read_cgroup_max(uint64_t & max_bytes) const {
    if (cgroup_mem_path_.empty()) {
        return limit_kind::INVALID;
    }
    const char * file = cgroup_version_ == 2 ? "/memory.max" : "/memory.limit_in_bytes";
    std::string text;
    if (!read_file_text(cgroup_mem_path_ + file, text)) {
        return limit_kind::INVALID;
    }
    uint64_t bytes = 0;
    limit_kind kind = cgroup_version_ == 2 ? parse_limit(text, bytes) :
                                            (parse_uint64(text, bytes) ? limit_kind::FINITE : limit_kind::INVALID);
    if (kind == limit_kind::FINITE && bytes == 0) {
        kind = limit_kind::INVALID;
    }
    if (kind == limit_kind::FINITE && cgroup_version_ == 1 && bytes > (UINT64_C(1) << 60)) {
        kind = limit_kind::UNLIMITED;
    }
    max_bytes = kind == limit_kind::FINITE ? bytes : 0;
    return kind;
}

kv_pressure_sampler::limit_kind kv_pressure_sampler::read_cgroup_high(uint64_t & high_bytes) const {
    if (cgroup_mem_path_.empty()) {
        return limit_kind::INVALID;
    }
    const char * file = cgroup_version_ == 2 ? "/memory.high" : "/memory.soft_limit_in_bytes";
    std::string text;
    if (!read_file_text(cgroup_mem_path_ + file, text)) {
        return limit_kind::INVALID;
    }
    uint64_t bytes = 0;
    limit_kind kind = cgroup_version_ == 2 ? parse_limit(text, bytes) :
                                            (parse_uint64(text, bytes) ? limit_kind::FINITE : limit_kind::INVALID);
    if (kind == limit_kind::FINITE && cgroup_version_ == 1 && bytes > (UINT64_C(1) << 60)) {
        kind = limit_kind::UNLIMITED;
    }
    high_bytes = kind == limit_kind::FINITE ? bytes : 0;
    return kind;
}

// ── PSI ─────────────────────────────────────────────────────────────────────

static bool parse_psi_average(const std::string & token, uint32_t & hundredths) {
    if (token.empty()) {
        return false;
    }

    size_t pos = 0;
    while (pos < token.size() && std::isdigit((unsigned char) token[pos])) {
        ++pos;
    }
    if (pos == 0) {
        return false;
    }
    if (pos < token.size()) {
        if (token[pos++] != '.' || pos == token.size()) {
            return false;
        }
        const size_t fraction_start = pos;
        while (pos < token.size() && std::isdigit((unsigned char) token[pos])) {
            ++pos;
        }
        if (pos == fraction_start) {
            return false;
        }
    }
    if (pos != token.size()) {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (errno == ERANGE || end != token.c_str() + token.size() ||
        !std::isfinite(value) || value < 0.0 || value > 100.0) {
        return false;
    }
    hundredths = (uint32_t) (value * 100.0 + 0.5);
    return hundredths <= 10000;
}

static bool parse_psi_line(const std::string & line, const char * expected_kind,
                           uint32_t & avg10, uint64_t & total) {
    std::vector<std::string> fields;
    size_t pos = 0;
    while (pos < line.size()) {
        while (pos < line.size() && std::isspace((unsigned char) line[pos])) {
            ++pos;
        }
        if (pos == line.size()) {
            break;
        }
        const size_t start = pos;
        while (pos < line.size() && !std::isspace((unsigned char) line[pos])) {
            ++pos;
        }
        fields.push_back(line.substr(start, pos - start));
    }
    if (fields.size() != 5 || fields[0] != expected_kind) {
        return false;
    }

    bool got_avg10 = false;
    bool got_avg60 = false;
    bool got_avg300 = false;
    bool got_total = false;
    uint32_t ignored_average = 0;
    for (size_t i = 1; i < fields.size(); ++i) {
        const size_t equals = fields[i].find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 == fields[i].size()) {
            return false;
        }
        const std::string key = fields[i].substr(0, equals);
        const std::string value = fields[i].substr(equals + 1);
        if (key == "avg10") {
            if (got_avg10 || !parse_psi_average(value, avg10)) {
                return false;
            }
            got_avg10 = true;
        } else if (key == "avg60") {
            if (got_avg60 || !parse_psi_average(value, ignored_average)) {
                return false;
            }
            got_avg60 = true;
        } else if (key == "avg300") {
            if (got_avg300 || !parse_psi_average(value, ignored_average)) {
                return false;
            }
            got_avg300 = true;
        } else if (key == "total") {
            if (got_total || !parse_decimal_uint64(value, total)) {
                return false;
            }
            got_total = true;
        } else {
            return false;
        }
    }
    return got_avg10 && got_avg60 && got_avg300 && got_total;
}

bool kv_pressure_sampler::read_psi(const char * path,
                                    uint32_t & some_avg10, uint32_t & full_avg10,
                                    uint64_t & some_total, uint64_t & full_total) const {
    // PSI format:
    // some avg10=X.XX avg60=Y.YY avg300=Z.ZZ total=NNNNN
    // full avg10=X.XX avg60=Y.YY avg300=Z.ZZ total=NNNNN
    std::string text;
    if (!read_file_text(path, text)) {
        return false;
    }

    some_avg10 = 0;
    full_avg10 = 0;
    some_total = 0;
    full_total = 0;

    bool got_some = false;
    bool got_full = false;
    size_t line_start = 0;
    while (line_start < text.size()) {
        const size_t line_end = text.find('\n', line_start);
        const std::string line = trim_ascii_space(text.substr(
            line_start, line_end == std::string::npos ? std::string::npos : line_end - line_start));
        line_start = line_end == std::string::npos ? text.size() : line_end + 1;
        if (line.empty()) {
            continue;
        }
        if (line.compare(0, 5, "some ") == 0) {
            if (got_some || !parse_psi_line(line, "some", some_avg10, some_total)) {
                return false;
            }
            got_some = true;
        } else if (line.compare(0, 5, "full ") == 0) {
            if (got_full || !parse_psi_line(line, "full", full_avg10, full_total)) {
                return false;
            }
            got_full = true;
        } else {
            return false;
        }
    }
    return got_some && got_full;
}

// ── cgroup path resolution ─────────────────────────────────────────────────

static std::vector<std::string> split_space_fields(const std::string & line) {
    std::vector<std::string> fields;
    size_t pos = 0;
    while (pos < line.size()) {
        while (pos < line.size() && line[pos] == ' ') {
            ++pos;
        }
        if (pos == line.size()) {
            break;
        }
        const size_t start = pos;
        while (pos < line.size() && line[pos] != ' ') {
            ++pos;
        }
        fields.push_back(line.substr(start, pos - start));
    }
    return fields;
}

static bool unescape_mountinfo_path(const std::string & encoded, std::string & decoded) {
    decoded.clear();
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] != '\\') {
            decoded.push_back(encoded[i]);
            continue;
        }
        if (i + 3 >= encoded.size()) {
            return false;
        }
        const std::string escape = encoded.substr(i, 4);
        if (escape == "\\040") {
            decoded.push_back(' ');
        } else if (escape == "\\011") {
            decoded.push_back('\t');
        } else if (escape == "\\012") {
            decoded.push_back('\n');
        } else if (escape == "\\134") {
            decoded.push_back('\\');
        } else {
            return false;
        }
        i += 3;
    }
    return true;
}

static bool path_under_root(const std::string & path, const std::string & root, std::string & suffix) {
    if (path.empty() || path[0] != '/' || root.empty() || root[0] != '/') {
        return false;
    }
    if (root == "/") {
        suffix = path == "/" ? "" : path;
        return true;
    }
    if (path == root) {
        suffix.clear();
        return true;
    }
    if (path.size() > root.size() && path.compare(0, root.size(), root) == 0 && path[root.size()] == '/') {
        suffix = path.substr(root.size());
        return true;
    }
    return false;
}

static bool controller_list_has_memory(const std::string & list) {
    size_t start = 0;
    while (start <= list.size()) {
        const size_t comma = list.find(',', start);
        const size_t end = comma == std::string::npos ? list.size() : comma;
        if (list.substr(start, end - start) == "memory") {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return false;
}

bool kv_pressure_sampler::resolve_cgroup() {
#ifdef __linux__
    cgroup_mem_path_.clear();
    cgroup_version_ = 0;

    std::string cgroup_text;
    std::string mountinfo_text;
    if (!read_file_text(paths_.proc_root + "/self/cgroup", cgroup_text) ||
        !read_file_text(paths_.proc_root + "/self/mountinfo", mountinfo_text)) {
        return false;
    }

    std::vector<std::string> v2_paths;
    std::vector<std::string> v1_paths;
    size_t line_start = 0;
    while (line_start < cgroup_text.size()) {
        const size_t line_end = cgroup_text.find('\n', line_start);
        const std::string line = trim_ascii_space(cgroup_text.substr(
            line_start, line_end == std::string::npos ? std::string::npos : line_end - line_start));
        line_start = line_end == std::string::npos ? cgroup_text.size() : line_end + 1;
        if (line.empty()) {
            continue;
        }
        const size_t first = line.find(':');
        const size_t second = first == std::string::npos ? std::string::npos : line.find(':', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            return false;
        }
        const std::string hierarchy = line.substr(0, first);
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const std::string path = line.substr(second + 1);
        uint64_t hierarchy_id = 0;
        if (!parse_decimal_uint64(hierarchy, hierarchy_id) || path.empty() || path[0] != '/') {
            return false;
        }
        if (hierarchy_id == 0 && controllers.empty()) {
            v2_paths.push_back(path);
        } else if (controller_list_has_memory(controllers)) {
            v1_paths.push_back(path);
        }
    }

    struct mount_candidate {
        int version;
        std::string root;
        std::string point;
    };
    std::vector<mount_candidate> mounts;
    line_start = 0;
    while (line_start < mountinfo_text.size()) {
        const size_t line_end = mountinfo_text.find('\n', line_start);
        const std::string line = mountinfo_text.substr(
            line_start, line_end == std::string::npos ? std::string::npos : line_end - line_start);
        line_start = line_end == std::string::npos ? mountinfo_text.size() : line_end + 1;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_space_fields(line);
        size_t dash = 0;
        while (dash < fields.size() && fields[dash] != "-") {
            ++dash;
        }
        if (fields.size() < 10 || dash < 6 || dash + 3 >= fields.size()) {
            continue;
        }
        int version = 0;
        if (fields[dash + 1] == "cgroup2") {
            version = 2;
        } else if (fields[dash + 1] == "cgroup" && controller_list_has_memory(fields[dash + 3])) {
            version = 1;
        } else {
            continue;
        }
        std::string root;
        std::string point;
        if (!unescape_mountinfo_path(fields[3], root) || !unescape_mountinfo_path(fields[4], point)) {
            return false;
        }
        mounts.push_back({ version, root, point });
    }

    struct resolved_candidate {
        int version;
        std::string path;
    };
    std::vector<resolved_candidate> resolved;
    for (size_t i = 0; i < mounts.size(); ++i) {
        const std::vector<std::string> & paths = mounts[i].version == 2 ? v2_paths : v1_paths;
        for (size_t j = 0; j < paths.size(); ++j) {
            std::string suffix;
            if (!path_under_root(paths[j], mounts[i].root, suffix)) {
                continue;
            }
            const std::string mounted = mounts[i].point == "/" ? suffix : mounts[i].point + suffix;
            resolved.push_back({ mounts[i].version, paths_.file_root + (mounted.empty() ? "/" : mounted) });
        }
    }

    // Duplicate cgroup entries, duplicate matching mounts, or hybrid v1/v2
    // matches are all ambiguous. Never guess a pressure source.
    if (resolved.size() != 1 || v2_paths.size() > 1 || v1_paths.size() > 1) {
        return false;
    }
    cgroup_version_ = resolved[0].version;
    cgroup_mem_path_ = resolved[0].path;
    return true;
#else
    cgroup_version_ = 0;
    return false;
#endif
}

// ── source determination ────────────────────────────────────────────────────

void kv_pressure_sampler::determine_source() {
    const bool has_absolute_rss = config_.pressure_rss_kb > 0;
    const bool has_absolute_cgroup = config_.pressure_cgroup_kb > 0;

    if (!config_valid_) {
        if (telemetry_.disabled_reason[0] == '\0') {
            disable_transitions("invalid configuration");
        }
    } else if (has_absolute_rss) {
        telemetry_.disabled_reason[0] = '\0';
        telemetry_.source = kv_pressure_source::RSS_ABSOLUTE;
        transitions_enabled_ = true;
    } else if (has_absolute_cgroup && cgroup_version_ > 0) {
        telemetry_.disabled_reason[0] = '\0';
        telemetry_.source = kv_pressure_source::CGROUP_ABSOLUTE;
        transitions_enabled_ = true;
    } else if (has_absolute_cgroup) {
        disable_transitions("cgroup source unavailable or ambiguous");
    } else if (cgroup_version_ > 0 && !cgroup_max_is_max_) {
        telemetry_.disabled_reason[0] = '\0';
        telemetry_.source = kv_pressure_source::CGROUP_RATIO;
        transitions_enabled_ = true;
    } else {
        disable_transitions(cgroup_version_ == 0 ? "cgroup source unavailable or ambiguous" :
                                                   "memory.max is max and no absolute thresholds configured");
    }
}

void kv_pressure_sampler::disable_transitions(const char * reason) {
    telemetry_.source = kv_pressure_source::NONE;
    transitions_enabled_ = false;
    std::snprintf(telemetry_.disabled_reason, sizeof(telemetry_.disabled_reason), "%s", reason);
}

bool kv_pressure_sampler::validate_config() {
    if (!(config_.low_water_ratio < config_.pressure_ratio &&
          config_.pressure_ratio < config_.critical_ratio &&
          config_.critical_ratio <= 10000)) {
        disable_transitions("ratio thresholds must satisfy low < pressure < critical <= 10000");
        return false;
    }
    if (config_.hysteresis_samples == 0 || config_.stale_threshold_samples == 0) {
        disable_transitions("hysteresis and stale thresholds must be non-zero");
        return false;
    }
    if (config_.psi_pressure_upgrade_threshold > 10000) {
        disable_transitions("PSI threshold must be in [0, 10000]");
        return false;
    }

    const bool rss_any = config_.low_water_rss_kb != 0 || config_.pressure_rss_kb != 0 ||
                         config_.critical_rss_kb != 0;
    const bool rss_complete = config_.low_water_rss_kb != 0 && config_.pressure_rss_kb != 0 &&
                              config_.critical_rss_kb != 0;
    const bool cgroup_any = config_.low_water_cgroup_kb != 0 || config_.pressure_cgroup_kb != 0 ||
                            config_.critical_cgroup_kb != 0;
    const bool cgroup_complete = config_.low_water_cgroup_kb != 0 && config_.pressure_cgroup_kb != 0 &&
                                 config_.critical_cgroup_kb != 0;
    if ((rss_any && !rss_complete) ||
        (rss_complete && !(config_.low_water_rss_kb < config_.pressure_rss_kb &&
                           config_.pressure_rss_kb < config_.critical_rss_kb))) {
        disable_transitions("RSS absolute thresholds must be complete and ordered low < pressure < critical");
        return false;
    }
    if ((cgroup_any && !cgroup_complete) ||
        (cgroup_complete && !(config_.low_water_cgroup_kb < config_.pressure_cgroup_kb &&
                              config_.pressure_cgroup_kb < config_.critical_cgroup_kb))) {
        disable_transitions("cgroup absolute thresholds must be complete and ordered low < pressure < critical");
        return false;
    }
    if (rss_any && cgroup_any) {
        disable_transitions("RSS and cgroup absolute thresholds are ambiguous");
        return false;
    }
    return true;
}

// ── constructor / destructor / init ─────────────────────────────────────────

kv_pressure_sampler::kv_pressure_sampler() : kv_pressure_sampler(kv_pressure_paths()) {}

kv_pressure_sampler::kv_pressure_sampler(const kv_pressure_paths & paths)
    : paths_(paths)
    , last_sample_ns_(0)
    , last_transition_ns_(0)
    , consecutive_pressure_samples_(0)
    , consecutive_below_low_water_(0)
    , consecutive_failures_(0)
    , consecutive_psi_pressure_(0)
    , cgroup_version_(0)
    , cgroup_max_is_max_(false)
    , config_valid_(true)
    , transitions_enabled_(false)
{}

kv_pressure_sampler::~kv_pressure_sampler() = default;

bool kv_pressure_sampler::init() {
    return init(kv_pressure_sampler_environment_enablement());
}

bool kv_pressure_sampler::init(kv_pressure_enablement enablement) {
    if (enablement == kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_INVALID) {
        config_.enabled = false;
        config_valid_ = false;
        telemetry_.config_valid = false;
        disable_transitions("LLAMA_KV_PRESSURE_SAMPLER is not a complete boolean");
        return false;
    }
    config_.enabled = enablement == kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_ENABLED;

    if (!config_.enabled) {
        telemetry_.enabled = false;
        telemetry_.source  = kv_pressure_source::NONE;
        telemetry_.state   = kv_pressure_state::NORMAL;
        return false;
    }

    bool env_valid = true;
    env_valid = parse_env_uint32("LLAMA_KV_PRESSURE_RATIO", 8000, config_.pressure_ratio) && env_valid;
    env_valid = parse_env_uint32("LLAMA_KV_CRITICAL_RATIO", 9500, config_.critical_ratio) && env_valid;
    env_valid = parse_env_uint32("LLAMA_KV_LOW_WATER_RATIO", 7000, config_.low_water_ratio) && env_valid;
    env_valid = parse_env_uint64("LLAMA_KV_PRESSURE_RSS_KB", 0, config_.pressure_rss_kb) && env_valid;
    env_valid = parse_env_uint64("LLAMA_KV_CRITICAL_RSS_KB", 0, config_.critical_rss_kb) && env_valid;
    env_valid = parse_env_uint64("LLAMA_KV_LOW_WATER_RSS_KB", 0, config_.low_water_rss_kb) && env_valid;
    env_valid = parse_env_uint64("LLAMA_KV_PRESSURE_CGROUP_KB", 0, config_.pressure_cgroup_kb) && env_valid;
    env_valid = parse_env_uint64("LLAMA_KV_CRITICAL_CGROUP_KB", 0, config_.critical_cgroup_kb) && env_valid;
    env_valid = parse_env_uint64("LLAMA_KV_LOW_WATER_CGROUP_KB", 0, config_.low_water_cgroup_kb) && env_valid;
    env_valid = parse_env_uint32("LLAMA_KV_PRESSURE_HYSTERESIS", 3, config_.hysteresis_samples) && env_valid;
    env_valid = parse_env_uint64("LLAMA_KV_PRESSURE_COOLDOWN_US", 5000000, config_.cooldown_us) && env_valid;
    env_valid = parse_env_uint32("LLAMA_KV_PSI_UPGRADE_THRESHOLD", 1000,
                                 config_.psi_pressure_upgrade_threshold) && env_valid;
    env_valid = parse_env_uint32("LLAMA_KV_PRESSURE_STALE_THRESHOLD", 5,
                                 config_.stale_threshold_samples) && env_valid;

    config_valid_ = env_valid;
    if (!env_valid) {
        disable_transitions("environment value is empty, negative, out of range, or has trailing data");
    } else {
        config_valid_ = validate_config();
    }

    // Resolve cgroup
    resolve_cgroup();

    cgroup_max_is_max_ = false;
    if (cgroup_version_ > 0) {
        uint64_t max_bytes = 0;
        const limit_kind kind = read_cgroup_max(max_bytes);
        cgroup_max_is_max_ = kind == limit_kind::UNLIMITED;
        if (kind == limit_kind::INVALID && config_.pressure_rss_kb == 0 && config_.pressure_cgroup_kb == 0) {
            cgroup_version_ = 0;
            cgroup_mem_path_.clear();
        }
    }

    // Determine the pressure signal source
    determine_source();

    // Set initial telemetry
    telemetry_.enabled       = true;
    telemetry_.state         = kv_pressure_state::NORMAL;
    telemetry_.previous_state = kv_pressure_state::NORMAL;
    telemetry_.sample_valid  = false;
    telemetry_.stale         = false;
    telemetry_.config_valid  = config_valid_;

    return true;
}

// ── set_state ───────────────────────────────────────────────────────────────

void kv_pressure_sampler::set_state(kv_pressure_state new_state, const char * reason) {
    if (new_state != telemetry_.state) {
        // PSI upgrade hysteresis belongs to one continuous PRESSURE lifetime.
        consecutive_psi_pressure_ = 0;
        telemetry_.previous_state = telemetry_.state;
        telemetry_.state          = new_state;
        last_transition_ns_       = last_sample_ns_;

        // Build transition reason
        int written = std::snprintf(telemetry_.transition_reason,
                                     sizeof(telemetry_.transition_reason),
                                     "%s→%s: %s",
                                     kv_pressure_state_name(telemetry_.previous_state),
                                     kv_pressure_state_name(new_state),
                                     reason);
        if (written < 0 || (size_t) written >= sizeof(telemetry_.transition_reason)) {
            telemetry_.transition_reason[sizeof(telemetry_.transition_reason) - 1] = '\0';
        }
    } else {
        // State unchanged — still note the reason for diagnostics
        std::snprintf(telemetry_.transition_reason,
                      sizeof(telemetry_.transition_reason),
                      "%s (no change): %s",
                      kv_pressure_state_name(telemetry_.state),
                      reason);
    }
}

void kv_pressure_sampler::saturating_increment(uint32_t & value) {
    if (value != UINT32_MAX) {
        ++value;
    }
}

void kv_pressure_sampler::reset_transition_counters() {
    consecutive_pressure_samples_ = 0;
    consecutive_below_low_water_ = 0;
    consecutive_failures_ = 0;
    consecutive_psi_pressure_ = 0;
}

void kv_pressure_sampler::refresh_transition_basis(const kv_pressure_telemetry & sample) {
    transition_basis next;
    next.initialized = true;
    next.source = sample.source;
    next.cgroup_max_unlimited = cgroup_max_is_max_;
    switch (sample.source) {
        case kv_pressure_source::CGROUP_RATIO:
            next.low_water = config_.low_water_ratio;
            next.pressure = config_.pressure_ratio;
            next.critical = config_.critical_ratio;
            next.ratio_maximum = sample.cgroup_max_bytes > 0 ?
                                 sample.cgroup_max_bytes : sample.cgroup_max_kb;
            break;
        case kv_pressure_source::RSS_ABSOLUTE:
            next.low_water = config_.low_water_rss_kb;
            next.pressure = config_.pressure_rss_kb;
            next.critical = config_.critical_rss_kb;
            break;
        case kv_pressure_source::CGROUP_ABSOLUTE:
            next.low_water = config_.low_water_cgroup_kb;
            next.pressure = config_.pressure_cgroup_kb;
            next.critical = config_.critical_cgroup_kb;
            break;
        case kv_pressure_source::NONE:
            break;
    }

    const bool changed = transition_basis_.initialized &&
        (transition_basis_.source != next.source ||
         transition_basis_.cgroup_max_unlimited != next.cgroup_max_unlimited ||
         transition_basis_.low_water != next.low_water ||
         transition_basis_.pressure != next.pressure ||
         transition_basis_.critical != next.critical ||
         transition_basis_.ratio_maximum != next.ratio_maximum);
    if (changed) {
        reset_transition_counters();
    }
    if (!transition_basis_.initialized || changed) {
        if (pressure_basis_generation_ != UINT64_MAX) {
            pressure_basis_generation_ += 1;
        }
    }
    transition_basis_ = next;

    telemetry_.pressure_basis_valid = false;
    telemetry_.pressure_current_bytes = 0;
    telemetry_.pressure_low_water_bytes = 0;
    telemetry_.pressure_basis_generation = pressure_basis_generation_;
    uint64_t low_bytes = 0;
    switch (sample.source) {
        case kv_pressure_source::RSS_ABSOLUTE:
            if (checked_mul(sample.rss_kb, 1024, telemetry_.pressure_current_bytes) &&
                    checked_mul(config_.low_water_rss_kb, 1024, low_bytes)) {
                telemetry_.pressure_low_water_bytes = low_bytes;
                telemetry_.pressure_basis_valid = true;
            }
            break;
        case kv_pressure_source::CGROUP_ABSOLUTE:
            if (checked_mul(config_.low_water_cgroup_kb, 1024, low_bytes)) {
                telemetry_.pressure_current_bytes = sample.cgroup_current_bytes;
                telemetry_.pressure_low_water_bytes = low_bytes;
                telemetry_.pressure_basis_valid = true;
            }
            break;
        case kv_pressure_source::CGROUP_RATIO: {
            const uint64_t maximum = sample.cgroup_max_bytes;
            const uint64_t quotient = maximum / 10000;
            const uint64_t remainder = maximum % 10000;
            uint64_t whole = 0;
            uint64_t numerator = 0;
            uint64_t fractional = 0;
            if (maximum > 0 && checked_mul(quotient, config_.low_water_ratio, whole) &&
                    checked_mul(remainder, config_.low_water_ratio, numerator)) {
                fractional = numerator / 10000 + (numerator % 10000 != 0 ? 1 : 0);
                if (checked_add(whole, fractional, low_bytes)) {
                    telemetry_.pressure_current_bytes = sample.cgroup_current_bytes;
                    telemetry_.pressure_low_water_bytes = low_bytes;
                    telemetry_.pressure_basis_valid = true;
                }
            }
            break;
        }
        case kv_pressure_source::NONE:
            break;
    }
}

// ── state machine evaluation ────────────────────────────────────────────────

void kv_pressure_sampler::evaluate_transition() {
    const kv_pressure_telemetry & t = telemetry_; // current sample snapshot

    if (!t.sample_valid) {
        // Sample failed — count towards stale, never demote
        saturating_increment(consecutive_failures_);
        consecutive_pressure_samples_ = 0;
        consecutive_below_low_water_  = 0;
        consecutive_psi_pressure_     = 0;

        telemetry_.stale = true;
        telemetry_.pressure_basis_valid = false;
        set_state(t.state, "sample failure: stale (no auto-demotion)");
        return;
    }

    // Valid sample — reset failure counter
    consecutive_failures_ = 0;
    telemetry_.stale = false;
    refresh_transition_basis(t);

    // Determine the signal values based on source
    //
    // We compute a "pressure level" that abstracts over ratio vs absolute:
    //   pressure_level < 0  : below pressure threshold (normal)
    //   pressure_level == 0 : at or above pressure threshold
    //   pressure_level == 1 : at or above critical threshold
    //
    // For CGROUP_RATIO: current/max * 10000 → compare against pressure_ratio / critical_ratio
    // For RSS_ABSOLUTE: rss_kb → compare against pressure_rss_kb / critical_rss_kb
    // For CGROUP_ABSOLUTE: cgroup_current_kb → compare against pressure_cgroup_kb / critical_cgroup_kb

    bool above_pressure = false;
    bool above_critical = false;
    bool below_low_water = true; // default: assume below low-water unless proven otherwise

    switch (telemetry_.source) {
        case kv_pressure_source::CGROUP_RATIO: {
            const uint64_t current = t.cgroup_max_bytes > 0 ? t.cgroup_current_bytes : t.cgroup_current_kb;
            const uint64_t maximum = t.cgroup_max_bytes > 0 ? t.cgroup_max_bytes : t.cgroup_max_kb;
            above_pressure = ratio_at_least(current, maximum, config_.pressure_ratio);
            above_critical = ratio_at_least(current, maximum, config_.critical_ratio);
            below_low_water = !ratio_at_least(current, maximum, config_.low_water_ratio);
            break;
        }
        case kv_pressure_source::RSS_ABSOLUTE: {
            if (config_.critical_rss_kb > 0 && t.rss_kb >= config_.critical_rss_kb) {
                above_critical = true;
                above_pressure = true;
            } else if (config_.pressure_rss_kb > 0 && t.rss_kb >= config_.pressure_rss_kb) {
                above_pressure = true;
            }
            below_low_water = (config_.low_water_rss_kb > 0)
                              ? (t.rss_kb < config_.low_water_rss_kb)
                              : (!above_pressure);
            break;
        }
        case kv_pressure_source::CGROUP_ABSOLUTE: {
            if (config_.critical_cgroup_kb > 0 && t.cgroup_current_kb >= config_.critical_cgroup_kb) {
                above_critical = true;
                above_pressure = true;
            } else if (config_.pressure_cgroup_kb > 0 && t.cgroup_current_kb >= config_.pressure_cgroup_kb) {
                above_pressure = true;
            }
            below_low_water = (config_.low_water_cgroup_kb > 0)
                              ? (t.cgroup_current_kb < config_.low_water_cgroup_kb)
                              : (!above_pressure);
            break;
        }
        case kv_pressure_source::NONE:
        default:
            // Telemetry-only mode — no state transitions
            set_state(t.state, "telemetry-only: no transitions enabled");
            return;
    }

    // PSI signal: whether PSI indicates sustained pressure
    bool psi_pressure = (config_.psi_pressure_upgrade_threshold > 0)
                        && (t.psi_some_avg10 >= config_.psi_pressure_upgrade_threshold);

    // Current time for cooldown check
    uint64_t now_ns = last_sample_ns_;
    const uint64_t elapsed_since_transition_ns = now_ns >= last_transition_ns_ ?
                                                  now_ns - last_transition_ns_ : 0;
    const bool cooldown_elapsed = last_transition_ns_ == 0 ||
                                  elapsed_since_transition_ns / 1000 >= config_.cooldown_us;

    // ── State machine ───────────────────────────────────────────────────

    switch (telemetry_.state) {
        case kv_pressure_state::NORMAL: {
            if (above_critical) {
                // CRITICAL enters immediately — bypass cooldown, bypass hysteresis
                saturating_increment(consecutive_pressure_samples_);
                consecutive_below_low_water_ = 0;
                set_state(kv_pressure_state::CRITICAL, "critical threshold: immediate entry");
            } else if (above_pressure) {
                saturating_increment(consecutive_pressure_samples_);
                consecutive_below_low_water_ = 0;
                if (consecutive_pressure_samples_ >= config_.hysteresis_samples) {
                    if (cooldown_elapsed) {
                        set_state(kv_pressure_state::PRESSURE, "pressure threshold + hysteresis + cooldown");
                    } else {
                        set_state(kv_pressure_state::NORMAL, "pressure threshold met but cooldown not elapsed");
                    }
                } else {
                    set_state(kv_pressure_state::NORMAL, "pressure threshold: building hysteresis");
                }
            } else {
                consecutive_pressure_samples_ = 0;
                saturating_increment(consecutive_below_low_water_);
                set_state(kv_pressure_state::NORMAL, "below pressure threshold");
            }
            break;
        }

        case kv_pressure_state::PRESSURE: {
            if (above_critical) {
                // Immediate CRITICAL entry — bypass everything
                saturating_increment(consecutive_pressure_samples_);
                consecutive_below_low_water_ = 0;
                set_state(kv_pressure_state::CRITICAL, "critical threshold: immediate entry from PRESSURE");
            } else if (below_low_water) {
                saturating_increment(consecutive_below_low_water_);
                consecutive_pressure_samples_ = 0;
                consecutive_psi_pressure_ = 0;
                if (consecutive_below_low_water_ >= config_.hysteresis_samples) {
                    if (cooldown_elapsed) {
                        set_state(kv_pressure_state::RECOVERY, "below low-water + hysteresis + cooldown");
                        consecutive_below_low_water_ = 0; // fresh hysteresis for RECOVERY→NORMAL
                    } else {
                        set_state(kv_pressure_state::PRESSURE, "below low-water but cooldown not elapsed");
                    }
                } else {
                    set_state(kv_pressure_state::PRESSURE, "below low-water: building exit hysteresis");
                }
            } else if (above_pressure && psi_pressure) {
                saturating_increment(consecutive_psi_pressure_);
                if (consecutive_psi_pressure_ >= config_.hysteresis_samples) {
                    // PSI upgrade: sustained PSI pressure + hysteresis → CRITICAL
                    saturating_increment(consecutive_pressure_samples_);
                    consecutive_below_low_water_ = 0;
                    set_state(kv_pressure_state::CRITICAL, "PSI upgrade: sustained pressure + high PSI");
                } else {
                    saturating_increment(consecutive_pressure_samples_);
                    consecutive_below_low_water_ = 0;
                    set_state(kv_pressure_state::PRESSURE, "pressure sustained + PSI elevated (building upgrade hysteresis)");
                }
            } else {
                // Still in or below the PRESSURE zone, but not below low-water.
                // PSI is only meaningful while the primary source remains above pressure.
                if (above_pressure) {
                    saturating_increment(consecutive_pressure_samples_);
                } else {
                    consecutive_pressure_samples_ = 0;
                }
                consecutive_below_low_water_ = 0;
                consecutive_psi_pressure_ = 0;
                set_state(kv_pressure_state::PRESSURE, "pressure sustained");
            }
            break;
        }

        case kv_pressure_state::CRITICAL: {
            if (above_critical) {
                // Still critical
                consecutive_below_low_water_ = 0;
                saturating_increment(consecutive_pressure_samples_);
                set_state(kv_pressure_state::CRITICAL, "critical sustained");
            } else if (below_low_water) {
                saturating_increment(consecutive_below_low_water_);
                consecutive_pressure_samples_ = 0;
                if (consecutive_below_low_water_ >= config_.hysteresis_samples) {
                    if (cooldown_elapsed) {
                        set_state(kv_pressure_state::RECOVERY, "below low-water after CRITICAL + hysteresis + cooldown");
                        consecutive_below_low_water_ = 0; // fresh hysteresis for RECOVERY→NORMAL
                    } else {
                        set_state(kv_pressure_state::CRITICAL, "below low-water but cooldown not elapsed");
                    }
                } else {
                    set_state(kv_pressure_state::CRITICAL, "below low-water: building exit hysteresis");
                }
            } else {
                // Above low-water but below critical — still CRITICAL (no automatic step-down)
                consecutive_below_low_water_ = 0;
                set_state(kv_pressure_state::CRITICAL, "below critical but above low-water: holding CRITICAL");
            }
            break;
        }

        case kv_pressure_state::RECOVERY: {
            if (above_critical) {
                // Jump back to CRITICAL immediately
                consecutive_below_low_water_ = 0;
                saturating_increment(consecutive_pressure_samples_);
                set_state(kv_pressure_state::CRITICAL, "critical threshold: immediate re-entry from RECOVERY");
            } else if (above_pressure) {
                saturating_increment(consecutive_pressure_samples_);
                consecutive_below_low_water_ = 0;
                if (consecutive_pressure_samples_ >= config_.hysteresis_samples) {
                    if (cooldown_elapsed) {
                        set_state(kv_pressure_state::PRESSURE, "pressure threshold re-entered from RECOVERY");
                    } else {
                        set_state(kv_pressure_state::RECOVERY, "pressure re-detected but cooldown not elapsed");
                    }
                } else {
                    set_state(kv_pressure_state::RECOVERY, "pressure re-detected: building hysteresis");
                }
            } else if (below_low_water) {
                saturating_increment(consecutive_below_low_water_);
                consecutive_pressure_samples_ = 0;
                if (consecutive_below_low_water_ >= config_.hysteresis_samples) {
                    if (cooldown_elapsed) {
                        set_state(kv_pressure_state::NORMAL, "recovery complete: sustained low-water + hysteresis + cooldown");
                    } else {
                        set_state(kv_pressure_state::RECOVERY, "low-water sustained but cooldown not elapsed");
                    }
                } else {
                    set_state(kv_pressure_state::RECOVERY, "recovery: building NORMAL exit hysteresis");
                }
            } else {
                // Between low-water and pressure — stay in RECOVERY
                consecutive_below_low_water_ = 0;
                set_state(kv_pressure_state::RECOVERY, "in recovery zone (between low-water and pressure)");
            }
            break;
        }
    }
}

// ── real sample ─────────────────────────────────────────────────────────────

kv_pressure_state kv_pressure_sampler::sample() {
    if (!config_.enabled) {
        return kv_pressure_state::NORMAL;
    }

    uint64_t t_start = get_time_ns();

    // Reset sample validity
    telemetry_.sample_valid = false;

    // 1. Read RSS. A failed optional source never overwrites the last valid
    // telemetry value with zero.
    uint64_t rss_kb = 0;
    bool rss_ok = read_rss_kb(rss_kb);
    if (rss_ok) {
        telemetry_.rss_kb = rss_kb;
    }

    // 2. Read cgroup memory fields independently so validity can follow the
    // currently selected source. memory.high is telemetry-only.
    uint64_t cg_current_bytes = 0;
    uint64_t cg_max_bytes     = 0;
    uint64_t cg_high_bytes    = 0;
    const bool cgroup_current_ok = cgroup_version_ > 0 && read_cgroup_current(cg_current_bytes);
    const limit_kind max_kind = cgroup_version_ > 0 ? read_cgroup_max(cg_max_bytes) : limit_kind::INVALID;
    const limit_kind high_kind = cgroup_version_ > 0 ? read_cgroup_high(cg_high_bytes) : limit_kind::INVALID;
    (void) high_kind;
    if (cgroup_current_ok) {
        telemetry_.cgroup_current_bytes = cg_current_bytes;
        telemetry_.cgroup_current_kb = cg_current_bytes / 1024;
    }
    if (max_kind != limit_kind::INVALID) {
        telemetry_.cgroup_max_bytes = cg_max_bytes;
        telemetry_.cgroup_max_kb = cg_max_bytes / 1024;
    }
    if (high_kind != limit_kind::INVALID) {
        telemetry_.cgroup_high_kb = cg_high_bytes / 1024;
    }

    // With no explicit absolute source, a valid runtime finite/max change
    // immediately reselects ratio vs telemetry-only. A malformed max keeps the
    // existing source and makes a ratio sample invalid.
    if (max_kind != limit_kind::INVALID) {
        const bool unlimited = max_kind == limit_kind::UNLIMITED;
        if (unlimited != cgroup_max_is_max_) {
            cgroup_max_is_max_ = unlimited;
            if (config_valid_ && config_.pressure_rss_kb == 0 && config_.pressure_cgroup_kb == 0) {
                determine_source();
            }
        }
    }

    // 3. Read system PSI from /proc/pressure/memory
    uint32_t psi_some_avg10 = 0;
    uint32_t psi_full_avg10 = 0;
    uint64_t psi_some_total = 0;
    uint64_t psi_full_total = 0;
    bool psi_ok = read_psi((paths_.proc_root + "/pressure/memory").c_str(),
                            psi_some_avg10, psi_full_avg10,
                            psi_some_total, psi_full_total);
    if (psi_ok) {
        telemetry_.psi_some_avg10 = psi_some_avg10;
        telemetry_.psi_full_avg10 = psi_full_avg10;
    } else {
        telemetry_.psi_some_avg10 = 0;
        telemetry_.psi_full_avg10 = 0;
    }

    // Also try cgroup-level PSI (v2 only)
    if (cgroup_version_ == 2 && !cgroup_mem_path_.empty()) {
        std::string cg_psi_path = cgroup_mem_path_ + "/memory.pressure";
        uint32_t cg_some = 0, cg_full = 0;
        uint64_t cg_some_tot = 0, cg_full_tot = 0;
        if (read_psi(cg_psi_path.c_str(), cg_some, cg_full, cg_some_tot, cg_full_tot)) {
            // Use max(system_psi, cgroup_psi) for the upgrade signal
            if (cg_some > telemetry_.psi_some_avg10) {
                telemetry_.psi_some_avg10 = cg_some;
            }
            if (cg_full > telemetry_.psi_full_avg10) {
                telemetry_.psi_full_avg10 = cg_full;
            }
        }
    }

    switch (telemetry_.source) {
        case kv_pressure_source::CGROUP_RATIO:
            telemetry_.sample_valid = cgroup_current_ok && max_kind == limit_kind::FINITE && cg_max_bytes > 0;
            break;
        case kv_pressure_source::RSS_ABSOLUTE:
            telemetry_.sample_valid = rss_ok;
            break;
        case kv_pressure_source::CGROUP_ABSOLUTE:
            telemetry_.sample_valid = cgroup_current_ok;
            break;
        case kv_pressure_source::NONE:
            if (config_valid_ && cgroup_version_ > 0 && config_.pressure_rss_kb == 0 &&
                config_.pressure_cgroup_kb == 0) {
                // memory.max remains necessary in telemetry-only mode because a
                // valid finite value must reactivate the ratio source.
                telemetry_.sample_valid = cgroup_current_ok && max_kind != limit_kind::INVALID;
            } else {
                telemetry_.sample_valid = rss_ok || cgroup_current_ok;
            }
            break;
    }
    last_sample_ns_ = get_time_ns();
    telemetry_.sample_latency_ns = last_sample_ns_ - t_start;

    // 4. Evaluate state machine
    evaluate_transition();

    return telemetry_.state;
}

// ── synthetic sample (for testing) ──────────────────────────────────────────

kv_pressure_state kv_pressure_sampler::sample_synthetic(
        const kv_pressure_telemetry & synthetic) {
    if (!config_.enabled) {
        return kv_pressure_state::NORMAL;
    }

    uint64_t t_start = get_time_ns();

    // Copy synthetic values into telemetry
    telemetry_.enabled          = synthetic.enabled;
    telemetry_.source           = synthetic.source;
    telemetry_.sample_valid     = synthetic.sample_valid;
    telemetry_.stale            = synthetic.stale;   // may be overridden below
    telemetry_.rss_kb           = synthetic.rss_kb;
    telemetry_.cgroup_current_bytes = synthetic.cgroup_current_bytes;
    telemetry_.cgroup_max_bytes = synthetic.cgroup_max_bytes;
    telemetry_.cgroup_current_kb = synthetic.cgroup_current_kb;
    telemetry_.cgroup_max_kb    = synthetic.cgroup_max_kb;
    telemetry_.cgroup_high_kb   = synthetic.cgroup_high_kb;
    telemetry_.psi_some_avg10   = synthetic.psi_some_avg10;
    telemetry_.psi_full_avg10   = synthetic.psi_full_avg10;
    telemetry_.sample_latency_ns = synthetic.sample_latency_ns;

    last_sample_ns_ = get_time_ns();
    if (telemetry_.sample_latency_ns == 0) {
        telemetry_.sample_latency_ns = last_sample_ns_ - t_start;
    }

    // Override source if the test provided one
    if (synthetic.source != kv_pressure_source::NONE) {
        telemetry_.source = synthetic.source;
    }

    // Evaluate state machine
    evaluate_transition();

    return telemetry_.state;
}

// ── get_time_ns ─────────────────────────────────────────────────────────────

uint64_t kv_pressure_sampler::get_time_ns() {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}
