#include "server-cache-log.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace {

// Column widths for the fixed-width cache-operations table. Chosen up front
// since this is an append-only, ever-growing file -- unlike the per-request
// performance footer, earlier rows can never be re-flowed once written. See
// REQUEST_LOGGING_SPEC.md's cache-operations design for the target format.
constexpr int W_TS      = 30; // "2026-08-15T10:23:41.221Z" (25 chars) + gap
constexpr int W_OP      = 18;
constexpr int W_LOC     = 16;
constexpr int W_TOKENS  = 10;
constexpr int W_BYTES   = 12;
constexpr int W_DURATION = 13;

std::string iso8601_utc_millis(std::chrono::system_clock::time_point tp) {
    const auto since_epoch = tp.time_since_epoch();
    const auto secs        = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
    const auto millis      = std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch) - secs;

    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[24];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_utc);

    std::ostringstream oss;
    oss << buf << '.' << std::setfill('0') << std::setw(3) << millis.count() << 'Z';
    return oss.str();
}

struct cache_log_state {
    std::mutex   mutex;
    std::ofstream ofs;
    bool         attempted = false; // ensure_open() was called at least once
    bool         enabled   = false; // request-logging-dir was non-empty at init
    std::string  dir;
};

cache_log_state & state() {
    static cache_log_state s;
    return s;
}

// caller must hold state().mutex
void ensure_open(cache_log_state & s) {
    if (s.attempted || !s.enabled) {
        return;
    }
    s.attempted = true;

    std::error_code ec;
    std::filesystem::create_directories(s.dir, ec);

    const std::string path = (std::filesystem::path(s.dir) / "cache-operations.log").string();
    const bool file_existed = std::filesystem::exists(path);

    // append mode, never truncate -- this is a running table for the whole
    // lifetime of the server process, potentially across restarts.
    s.ofs.open(path, std::ios::out | std::ios::app);
    if (!s.ofs.is_open()) {
        fprintf(stderr, "warning: cache-logging: failed to open '%s' for writing\n", path.c_str());
        return;
    }

    if (!file_existed) {
        s.ofs << server_cache_log_header_row(/* with_timestamp_and_detail */ true) << "\n";
        s.ofs.flush();
    }
}

} // namespace

std::string fmt_fixed(double v, int prec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

std::string format_bytes_human(std::optional<int64_t> bytes) {
    if (!bytes) {
        return "n/a";
    }
    double       v      = static_cast<double>(*bytes);
    const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    int          unit   = 0;
    while (v >= 1024.0 && unit < 4) {
        v /= 1024.0;
        unit++;
    }
    return fmt_fixed(v, 1) + " " + units[unit];
}

void server_cache_log_init(const std::string & request_logging_dir) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.attempted) {
        // already initialized (or a previous init already ran) -- ignore.
        return;
    }
    s.dir     = request_logging_dir;
    s.enabled = !request_logging_dir.empty();
    // actual file open is deferred to the first note() call (lazy open, per
    // the design: "lazily open (on first cache event, append mode, never
    // truncate on restart)").
}

void server_cache_log_note(const server_cache_event & ev) {
    try {
        auto & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.enabled) {
            return;
        }
        ensure_open(s);
        if (!s.ofs.is_open()) {
            return;
        }
        s.ofs << server_cache_log_format_row(ev, /* with_timestamp_and_detail */ true) << "\n";
        s.ofs.flush();
    } catch (...) {
        // best-effort: never let cache-operation logging break cache functionality
    }
}

std::string server_cache_log_header_row(bool with_timestamp_and_detail) {
    std::ostringstream oss;
    if (with_timestamp_and_detail) {
        oss << std::left << std::setw(W_TS) << "TIMESTAMP";
    }
    oss << std::left << std::setw(W_OP)       << "OPERATION"
        << std::left << std::setw(W_LOC)      << "LOCATION"
        << std::left << std::setw(W_TOKENS)   << "TOKENS"
        << std::left << std::setw(W_BYTES)    << "BYTES"
        << std::left << std::setw(W_DURATION) << "DURATION";
    if (with_timestamp_and_detail) {
        oss << "DETAIL";
    }
    // trim trailing spaces from the last padded column when there's no DETAIL
    std::string out = oss.str();
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

std::string server_cache_log_format_row(const server_cache_event & ev, bool with_timestamp_and_detail) {
    std::ostringstream oss;
    if (with_timestamp_and_detail) {
        oss << std::left << std::setw(W_TS) << iso8601_utc_millis(std::chrono::system_clock::now());
    }

    const std::string tokens_str = ev.tokens ? std::to_string(*ev.tokens) : "n/a";
    const std::string duration_str = fmt_fixed(ev.duration_ms, 3) + " ms";

    oss << std::left << std::setw(W_OP)       << ev.operation
        << std::left << std::setw(W_LOC)      << ev.location
        << std::left << std::setw(W_TOKENS)   << tokens_str
        << std::left << std::setw(W_BYTES)    << format_bytes_human(ev.bytes)
        << std::left << std::setw(W_DURATION) << duration_str;

    if (with_timestamp_and_detail) {
        oss << ev.detail;
    }

    std::string out = oss.str();
    if (!with_timestamp_and_detail) {
        while (!out.empty() && out.back() == ' ') {
            out.pop_back();
        }
    }
    return out;
}
