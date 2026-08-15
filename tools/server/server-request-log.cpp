#include "server-request-log.h"
#include "server-task.h"   // server_task_result, server_task_result_cmpl_partial/_final, result_timings
#include "server-common.h" // SRV_ERR, json alias

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

namespace {

std::string fmt_fixed(double v, int prec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

std::string iso8601_utc(std::chrono::system_clock::time_point tp, bool filesystem_safe) {
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), filesystem_safe ? "%Y-%m-%dT%H-%M-%SZ" : "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

// first n_words words of `text`, sanitized+joined with '_', capped at max_len
std::string first_n_words_stub(const std::string & text, int n_words, size_t max_len) {
    std::istringstream iss(text);
    std::string word;
    std::string out;
    int count = 0;
    while (count < n_words && (iss >> word)) {
        if (out.size() >= max_len) {
            break;
        }
        const std::string san = server_request_log_sanitize(word, max_len - out.size());
        if (!san.empty()) {
            if (!out.empty()) {
                out.push_back('_');
            }
            out += san;
        }
        count++;
    }
    if (out.size() > max_len) {
        out.resize(max_len);
    }
    if (out.empty()) {
        out = "prompt";
    }
    return out;
}

} // namespace

std::string server_request_log_sanitize(const std::string & text, size_t max_len) {
    std::string out;
    out.reserve(std::min(text.size(), max_len));
    bool last_was_sep = false;
    for (unsigned char c : text) {
        if (out.size() >= max_len) {
            break;
        }
        if (std::isalnum(c) || c == '_' || c == '-') {
            out.push_back(static_cast<char>(c));
            last_was_sep = false;
        } else if (!last_was_sep && !out.empty()) {
            out.push_back('_');
            last_was_sep = true;
        }
    }
    while (!out.empty() && (out.back() == '_' || out.back() == '-')) {
        out.pop_back();
    }
    return out;
}

std::string server_request_log_build_path(
        const std::string & requests_dir,
        const std::string & request_id,
        const std::string & prompt_text) {
    std::filesystem::path dir(requests_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    // if create_directories fails, we still try to build a path; the actual
    // ofstream::open() in the writer will fail loudly (best-effort) instead

    const std::string ts = iso8601_utc(std::chrono::system_clock::now(), /* filesystem_safe */ true);

    std::string id_part = server_request_log_sanitize(request_id, 40);
    if (id_part.empty()) {
        id_part = "req";
    }

    const std::string stub = first_n_words_stub(prompt_text, 10, 80);

    const std::string base = ts + "_" + id_part + "_" + stub;
    std::string candidate = base + ".log";
    int n = 2;
    while (std::filesystem::exists(dir / candidate)) {
        candidate = base + "-" + std::to_string(n) + ".log";
        n++;
    }
    return (dir / candidate).string();
}

std::string server_request_log_extract_request_id(
        const std::map<std::string, std::string> & headers,
        const std::string & fallback_id) {
    for (const auto & [k, v] : headers) {
        std::string lower = k;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                        [](unsigned char c) { return std::tolower(c); });
        if ((lower == "x-request-id" || lower == "request-id") && !v.empty()) {
            return v;
        }
    }
    return fallback_id;
}

std::string server_request_log_render_chat_messages(const json & messages) {
    std::ostringstream oss;
    if (!messages.is_array()) {
        return "";
    }
    for (const auto & msg : messages) {
        std::string role = msg.value("role", std::string("unknown"));
        std::string tag = role.empty() ? "UNKNOWN" : role;
        std::transform(tag.begin(), tag.end(), tag.begin(), [](unsigned char c) { return std::toupper(c); });

        oss << tag << ":\n";

        if (msg.contains("content") && !msg.at("content").is_null()) {
            const json & content = msg.at("content");
            if (content.is_string()) {
                oss << content.get<std::string>() << "\n";
            } else if (content.is_array()) {
                for (const auto & part : content) {
                    const std::string type = part.value("type", std::string());
                    if (type == "text") {
                        oss << part.value("text", std::string()) << "\n";
                    } else if (type == "image_url") {
                        oss << "[image attached]\n";
                    } else if (type == "input_audio") {
                        oss << "[audio attached]\n";
                    } else if (type == "input_video") {
                        oss << "[video attached]\n";
                    } else if (type == "media_marker") {
                        oss << "[attachment]\n";
                    } else if (!type.empty()) {
                        oss << "[attachment: " << type << "]\n";
                    }
                }
            } else {
                oss << content.dump() << "\n";
            }
        }

        if (msg.contains("tool_calls") && msg.at("tool_calls").is_array()) {
            for (const auto & tc : msg.at("tool_calls")) {
                std::string name;
                std::string args;
                if (tc.contains("function")) {
                    const json & fn = tc.at("function");
                    name = fn.value("name", std::string());
                    if (fn.contains("arguments")) {
                        args = fn.at("arguments").is_string()
                            ? fn.at("arguments").get<std::string>()
                            : fn.at("arguments").dump();
                    }
                }
                oss << "[tool_call: " << name << "(" << args << ")]\n";
            }
        }

        oss << "\n";
    }
    return oss.str();
}

namespace {

// every method a no-op; used when --request-logging-dir is not set
class null_writer : public server_request_log_writer {
public:
    void write_header(
        const std::string &, const std::string &, const std::string &, const std::string &,
        const std::string &, const std::string &, const std::map<std::string, std::string> &,
        const json &, const json &, const std::string &, const std::string &) override {}
    void set_expected_results(size_t) override {}
    void on_result(const std::unique_ptr<server_task_result> &) override {}
    void write_error(const std::string &, const std::string &, std::optional<int>) override {}
    void close() override {}
};

struct footer_data {
    std::optional<int32_t> prompt_tokens;
    std::optional<int32_t> completion_tokens;
    std::optional<int32_t> cached_tokens;
    std::optional<double>  pp_tokens_per_sec;
    std::optional<double>  tg_tokens_per_sec;
    std::optional<double>  ttft_sec;
    std::optional<double>  prefill_duration_sec;
    std::optional<double>  generation_duration_sec;
    std::optional<double>  total_duration_sec;
    std::optional<double>  queue_wait_sec;
    // concurrent_requests_at_prefill_start / concurrent_requests_at_decode_start /
    // ran_prefill_in_parallel / ran_decode_in_parallel are intentionally not
    // tracked yet: there is no cheap thread-safe way to read "how many other
    // slots are busy right now" from the HTTP handler thread today (the only
    // existing mechanism, SERVER_TASK_TYPE_METRICS, requires a queue
    // round-trip and is too heavyweight to do on every request). Adding a
    // small atomic counter alongside the existing cumulative busy-slot
    // metric (server-context.cpp, near metrics.on_decoded()) -- mirroring
    // the get_active_user_count() pattern -- would unlock these fields; see
    // REQUEST_LOGGING_SPEC.md Part 3 "Concurrency count" for the suggested
    // approach. Deferred to keep this port's scope contained.
};

class real_writer : public server_request_log_writer {
public:
    explicit real_writer(std::string path) : path_(std::move(path)) {}

    ~real_writer() override {
        // RAII safety net: if this writer gets destroyed (client disconnect,
        // cancellation, or any code path that never reached a terminal
        // on_result()/write_error()) without ever finalizing, do it now
        // using whatever partial data we tracked. This is what makes
        // disconnects/cancellations first-class instead of a silent drop --
        // see REQUEST_LOGGING_SPEC.md Part 1 gotcha #1 and Part 3
        // "Cancellation handling: RAII does most of the work for you".
        try {
            finalize_cancelled_if_needed();
        } catch (...) {
            // never let a logging failure escape a destructor
        }
    }

    void write_header(
            const std::string & request_id,
            const std::string & client_addr,
            const std::string & method,
            const std::string & path,
            const std::string & model,
            const std::string & resolved_model,
            const std::map<std::string, std::string> & headers,
            const json & sampling_params,
            const json & model_settings,
            const std::string & raw_request_body,
            const std::string & prompt_text) override {
        try {
            ensure_open();
            if (!ofs_.is_open()) {
                return;
            }

            t_start_ = std::chrono::steady_clock::now();

            ofs_ << "=== CACHYLLAMA REQUEST LOG ===\n";
            ofs_ << "timestamp: "    << iso8601_utc(std::chrono::system_clock::now(), false) << "\n";
            ofs_ << "request_id: "   << request_id << "\n";
            ofs_ << "client: "       << (client_addr.empty() ? "unknown" : client_addr) << "\n";
            ofs_ << "method_path: "  << method << " " << path << "\n";
            ofs_ << "model: "        << model << "\n";
            ofs_ << "resolved_model: " << resolved_model << "\n";

            ofs_ << "\n=== HEADERS ===\n";
            for (const auto & [k, v] : headers) {
                ofs_ << k << ": " << v << "\n";
            }

            ofs_ << "\n=== SAMPLING PARAMS ===\n";
            ofs_ << sampling_params.dump(2) << "\n";

            ofs_ << "\n=== MODEL SETTINGS ===\n";
            ofs_ << model_settings.dump(2) << "\n";

            ofs_ << "\n=== REQUEST ===\n";
            try {
                ofs_ << json::parse(raw_request_body).dump(2) << "\n";
            } catch (const std::exception &) {
                ofs_ << raw_request_body << "\n";
            }

            ofs_ << "\n=== PROMPT ===\n";
            ofs_ << prompt_text << "\n";

            ofs_.flush();
        } catch (...) {
            // best-effort: never let logging break request handling
        }
    }

    void set_expected_results(size_t n) override {
        expected_results_ = n > 0 ? n : 1;
    }

    void on_result(const std::unique_ptr<server_task_result> & result) override {
        if (!result) {
            return;
        }
        try {
            ensure_open();
            if (!ofs_.is_open() || finalized_) {
                return;
            }

            if (result->is_error()) {
                const json err_json = result->to_json();
                const std::string message = err_json.value("message", std::string("unknown error"));
                const std::string type    = err_json.value("type", std::string("server_error"));
                const int code             = err_json.value("code", 0);
                finalize_error("=== ERROR ===", type, message, code > 0 ? std::optional<int>(code) : std::nullopt);
                return;
            }

            if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                note_progress(partial->n_decoded, partial->n_prompt_tokens, partial->n_prompt_tokens_cache,
                               partial->timings, partial->queue_wait_us);
                if (!partial->is_begin && !partial->is_progress && !partial->content.empty()) {
                    append_chunk(partial->content);
                }
                return;
            }

            if (auto * final_res = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
                note_progress(final_res->n_decoded, final_res->n_prompt_tokens, final_res->n_prompt_tokens_cache,
                               final_res->timings, final_res->queue_wait_us);
                if (!final_res->content.empty()) {
                    // content is only non-empty here for the non-streaming
                    // case; in streaming mode the text was already sent via
                    // partial chunks (see send_final_response()).
                    append_chunk(final_res->content);
                }
                finals_seen_++;
                if (finals_seen_ >= expected_results_) {
                    finalize_success();
                }
            }
        } catch (...) {
            // best-effort: never let logging break request handling
        }
    }

    void write_error(const std::string & exception_type, const std::string & message,
                      std::optional<int> http_status) override {
        try {
            ensure_open();
            if (!ofs_.is_open() || finalized_) {
                return;
            }
            finalize_error("=== ERROR ===", exception_type, message, http_status);
        } catch (...) {
        }
    }

    void close() override {
        try {
            if (!finalized_ && ofs_.is_open()) {
                finalize_success();
            }
            if (ofs_.is_open()) {
                ofs_.flush();
            }
        } catch (...) {
        }
    }

private:
    void ensure_open() {
        if (opened_) {
            return;
        }
        opened_ = true;
        ofs_.open(path_, std::ios::out | std::ios::trunc);
        if (!ofs_.is_open()) {
            fprintf(stderr, "warning: request-logging: failed to open '%s' for writing\n", path_.c_str());
        }
    }

    void append_chunk(const std::string & text) {
        if (!chunk_header_written_) {
            ofs_ << "\n=== RESPONSE (streaming) ===\n";
            chunk_header_written_ = true;
        }
        ofs_ << text;
        ofs_.flush();
    }

    void note_progress(int32_t n_decoded, int32_t n_prompt_tokens, int32_t n_prompt_tokens_cache,
                        const result_timings & timings, int64_t queue_wait_us) {
        last_n_decoded_             = n_decoded;
        last_n_prompt_tokens_       = n_prompt_tokens;
        last_n_prompt_tokens_cache_ = n_prompt_tokens_cache;
        // result_timings defaults prompt_n/predicted_n to -1 when unset (see
        // send_partial_response(): timings are only populated on the final
        // chunk or when timings_per_token is on) -- only adopt them once
        // they're actually present, so an earlier "no timings yet" chunk
        // never clobbers a later real value.
        if (timings.prompt_n >= 0 || timings.predicted_n >= 0) {
            last_timings_  = timings;
            have_timings_  = true;
        }
        if (queue_wait_us >= 0) {
            last_queue_wait_sec_ = queue_wait_us / 1e6;
        }
    }

    footer_data build_footer() const {
        footer_data f;
        if (last_n_prompt_tokens_ >= 0)       { f.prompt_tokens = last_n_prompt_tokens_; }
        if (last_n_decoded_ >= 0)             { f.completion_tokens = last_n_decoded_; }
        if (last_n_prompt_tokens_cache_ >= 0) { f.cached_tokens = last_n_prompt_tokens_cache_; }
        if (have_timings_) {
            if (last_timings_.prompt_n > 0 && last_timings_.prompt_ms > 0.0) {
                f.pp_tokens_per_sec    = last_timings_.prompt_per_second;
                f.prefill_duration_sec = last_timings_.prompt_ms / 1000.0;
                // TTFT: on this codebase the first generated token follows
                // immediately after prompt processing finishes (no separate
                // "first token" timestamp is tracked beyond t_prompt_processing,
                // which is set exactly at n_decoded==1 -- see server-context.cpp
                // pre_decode()/the sampling loop), so prefill duration doubles
                // as TTFT.
                f.ttft_sec = f.prefill_duration_sec;
            }
            if (last_timings_.predicted_n > 0 && last_timings_.predicted_ms > 0.0) {
                f.tg_tokens_per_sec        = last_timings_.predicted_per_second;
                f.generation_duration_sec  = last_timings_.predicted_ms / 1000.0;
            }
        }
        if (last_queue_wait_sec_) {
            f.queue_wait_sec = last_queue_wait_sec_;
        }
        f.total_duration_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start_).count();
        return f;
    }

    // Renders an aligned (label, value-with-unit) table -- see
    // REQUEST_LOGGING_SPEC.md Part 1 "File format": the label column is
    // left-justified to the width of the longest *included* label in this
    // particular footer (recomputed per call, since which optional rows are
    // present varies between the success footer and the partial-on-error
    // footer), values always carry their unit inline. Rows for uncomputable
    // fields are simply not appended -- never fake a value that wasn't
    // actually measured.
    void write_footer_block(const std::string & label, const footer_data & f) {
        std::vector<std::pair<std::string, std::string>> rows;

        if (f.prompt_tokens)     { rows.emplace_back("Prompt tokens", std::to_string(*f.prompt_tokens)); }
        if (f.completion_tokens) { rows.emplace_back("Completion tokens", std::to_string(*f.completion_tokens)); }
        if (f.cached_tokens) {
            std::string value = std::to_string(*f.cached_tokens);
            if (f.prompt_tokens && *f.prompt_tokens > 0) {
                const double ratio = 100.0 * static_cast<double>(*f.cached_tokens) / static_cast<double>(*f.prompt_tokens);
                value += " (" + fmt_fixed(ratio, 1) + "%)";
            }
            rows.emplace_back("Cached tokens", value);
        }
        if (f.pp_tokens_per_sec)       { rows.emplace_back("Prefill speed (PP)",    fmt_fixed(*f.pp_tokens_per_sec, 2) + " tok/s"); }
        if (f.tg_tokens_per_sec)       { rows.emplace_back("Generation speed (TG)", fmt_fixed(*f.tg_tokens_per_sec, 2) + " tok/s"); }
        if (f.ttft_sec)                { rows.emplace_back("Time to first token",   fmt_fixed(*f.ttft_sec, 3) + " s"); }
        if (f.prefill_duration_sec)    { rows.emplace_back("Prefill duration",      fmt_fixed(*f.prefill_duration_sec, 3) + " s"); }
        if (f.generation_duration_sec) { rows.emplace_back("Generation duration",   fmt_fixed(*f.generation_duration_sec, 3) + " s"); }
        if (f.total_duration_sec)      { rows.emplace_back("Total duration",        fmt_fixed(*f.total_duration_sec, 3) + " s"); }
        if (f.queue_wait_sec)          { rows.emplace_back("Queue wait",            fmt_fixed(*f.queue_wait_sec, 3) + " s"); }

        ofs_ << "\n=== " << label << " ===\n";
        if (!rows.empty()) {
            size_t width = 0;
            for (const auto & row : rows) {
                width = std::max(width, row.first.size());
            }
            for (const auto & row : rows) {
                ofs_ << std::left << std::setw(static_cast<int>(width)) << row.first << " : " << row.second << "\n";
            }
        }
        ofs_.flush();
    }

    void finalize_success() {
        if (finalized_) {
            return;
        }
        finalized_ = true;
        write_footer_block("PERFORMANCE", build_footer());
    }

    void finalize_error(const std::string & label, const std::string & exception_type,
                         const std::string & message, std::optional<int> http_status) {
        if (finalized_) {
            return;
        }
        finalized_ = true;
        ofs_ << "\n" << label << "\n";
        ofs_ << "exception_type: " << (exception_type.empty() ? "server_error" : exception_type) << "\n";
        ofs_ << "message: " << message << "\n";
        if (http_status) {
            ofs_ << "http_status: " << *http_status << "\n";
        }
        write_footer_block("PERFORMANCE (partial)", build_footer());
    }

    void finalize_cancelled_if_needed() {
        if (finalized_ || !opened_ || !ofs_.is_open()) {
            return;
        }
        finalized_ = true;
        ofs_ << "\n=== CANCELLED ===\n";
        ofs_ << "message: request was destroyed before completion (client disconnect or server-side cancellation)\n";
        write_footer_block("PERFORMANCE (partial)", build_footer());
        ofs_.flush();
        ofs_.close();
    }

    std::string path_;
    std::ofstream ofs_;
    bool opened_    = false;
    bool finalized_ = false;
    bool chunk_header_written_ = false;

    size_t expected_results_ = 1;
    size_t finals_seen_      = 0;

    int32_t last_n_decoded_             = -1;
    int32_t last_n_prompt_tokens_       = -1;
    int32_t last_n_prompt_tokens_cache_ = -1;
    bool have_timings_ = false;
    result_timings last_timings_;
    std::optional<double> last_queue_wait_sec_;

    std::chrono::steady_clock::time_point t_start_ = std::chrono::steady_clock::now();
};

} // namespace

std::unique_ptr<server_request_log_writer> server_request_log_create(
        const std::string & requests_dir,
        const std::string & request_id,
        const std::string & prompt_text) {
    try {
        std::string path = server_request_log_build_path(requests_dir, request_id, prompt_text);
        return std::make_unique<real_writer>(std::move(path));
    } catch (const std::exception & e) {
        fprintf(stderr, "warning: request-logging: failed to initialize writer: %s\n", e.what());
        return server_request_log_create_null();
    }
}

std::unique_ptr<server_request_log_writer> server_request_log_create_null() {
    return std::make_unique<null_writer>();
}
