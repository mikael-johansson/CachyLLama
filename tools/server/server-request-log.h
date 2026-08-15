#pragma once

// Full per-request disk logging ("request logging").
//
// Opt-in via --request-logging-dir. One human-readable .log file per
// request, written incrementally as the response is generated, so that a
// request that hangs, gets cancelled, or errors out mid-generation still
// leaves a readable trace of what actually happened. This is a port of the
// oMLX request_logging.py feature (see REQUEST_LOGGING_SPEC.md in the omlx
// repo for the full portable spec + rationale); this module mirrors that
// module's shape: a path builder, a writer interface, and a null writer for
// the disabled case so call sites never have to branch on the setting.
//
// File format (see spec Part 1 "File format" for the authoritative version):
//
//   === CACHYLLAMA REQUEST LOG ===
//   timestamp / request_id / client / method_path / model / resolved_model
//
//   === HEADERS ===
//   (verbatim, no redaction -- see spec's "Configuration decisions")
//
//   === SAMPLING PARAMS ===
//   (effective/resolved sampling params, pretty JSON)
//
//   === MODEL SETTINGS ===
//   (small subset of server-side model metadata, pretty JSON)
//
//   === REQUEST ===
//   (raw request body, pretty JSON)
//
//   === PROMPT ===
//   (readable rendering: SYSTEM:/USER:/ASSISTANT:/TOOL: blocks for chat
//   requests, or the resolved prompt text otherwise)
//
//   === RESPONSE (streaming) ===
//   (raw generated text, appended as it is produced)
//
// followed by exactly one of:
//
//   === PERFORMANCE ===             (success)
//   === ERROR ===  + === PERFORMANCE (partial) ===   (explicit failure)
//   === CANCELLED === + === PERFORMANCE (partial) === (client disconnect /
//                                                       destroyed without
//                                                       ever finalizing)
//
// Fields that can't be computed are omitted, never faked as 0 -- see the
// spec's "fields that can't be computed" note.

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

struct server_task_result;

// Build the on-disk path for a request's log file inside `requests_dir`.
// `prompt_text` is used to derive a short, sanitized filename stub (first
// ~10 words). Creates `requests_dir` if it doesn't exist. Handles filename
// collisions (two requests landing in the same second) by appending
// -2, -3, ... rather than silently overwriting a previous log.
std::string server_request_log_build_path(
    const std::string & requests_dir,
    const std::string & request_id,
    const std::string & prompt_text);

// Sanitize a string down to the [A-Za-z0-9_-] alphabet used for filename
// stubs / request-id components, collapsing runs of other characters (incl.
// whitespace) to a single '_', truncated to max_len.
std::string server_request_log_sanitize(const std::string & text, size_t max_len);

// Render an OAI-style `messages` JSON array into a readable
// SYSTEM:/USER:/ASSISTANT:/TOOL: transcript for the "=== PROMPT ===" section,
// with attachments rendered as [image attached] / [audio attached] / etc.
// and tool calls as [tool_call: name(args)].
std::string server_request_log_render_chat_messages(const nlohmann::ordered_json & messages);

// Best-effort extraction of a client-supplied request-id header (checked
// case-insensitively: x-request-id, request-id), capped/sanitized. Falls
// back to `fallback_id` (typically the server-generated completion id) if
// no usable header is present.
std::string server_request_log_extract_request_id(
    const std::map<std::string, std::string> & headers,
    const std::string & fallback_id);

struct server_request_log_writer {
    virtual ~server_request_log_writer() = default;

    // Write the header sections. Called once per request, after the request
    // has been parsed/validated and the effective sampling params are known,
    // but before any task is posted to the queue.
    //
    // `thinking_forced_open`: true when this request's chat template appends
    // an opening reasoning-block tag (e.g. "<think>") to the end of the
    // rendered prompt, forcing the model straight into its reasoning phase.
    // In that case the model's own generated tokens never include the
    // opening tag (only the matching close), so the writer synthesizes it as
    // the first line of "=== RESPONSE (streaming) ===" -- otherwise the log
    // shows a lone closing tag with no opener, which reads as truncation.
    // See REQUEST_LOGGING_SPEC.md Part 1 "Reasoning-model <think> prefix".
    virtual void write_header(
        const std::string & request_id,
        const std::string & client_addr,
        const std::string & method,
        const std::string & path,
        const std::string & model,
        const std::string & resolved_model,
        const std::map<std::string, std::string> & headers,
        const nlohmann::ordered_json & sampling_params,
        const nlohmann::ordered_json & model_settings,
        const std::string & raw_request_body,
        const std::string & prompt_text,
        bool thinking_forced_open = false) = 0;

    // Number of terminal (final or error) results expected before the
    // request as a whole is considered done. 1 for a normal request, >1 only
    // when n_cmpl > 1 (parallel sampling / child tasks). Must be set before
    // the first call to on_result(), typically right after write_header().
    virtual void set_expected_results(size_t n) = 0;

    // Feed every result server_response_reader::next() produces for this
    // request, in arrival order. This is the single hook point that covers
    // both streaming and non-streaming HTTP modes (both flow through
    // server_response_reader::next() internally -- see server-queue.cpp).
    // Handles response-chunk appending and writes the terminal
    // PERFORMANCE / ERROR+PERFORMANCE(partial) block once the expected
    // number of terminal results has been seen. Never throws.
    virtual void on_result(const std::unique_ptr<server_task_result> & result) = 0;

    // Explicitly record a failure that happened before any task was ever
    // posted to the queue (request validation, per-user rate limiting,
    // exceptions inside the streaming SSE callback, ...). Idempotent with
    // on_result()'s error handling and the destructor fallback: only the
    // first call actually writes anything. Never throws.
    virtual void write_error(
        const std::string & exception_type,
        const std::string & message,
        std::optional<int> http_status) = 0;

    // Explicit idempotent finalize/flush. Not required for correctness (the
    // destructor finalizes automatically if this was never reached -- see
    // server-request-log.cpp), but available for callers that want to force
    // a flush without waiting for the writer to be destroyed.
    virtual void close() = 0;
};

// Real writer (opens/writes a file under requests_dir) or a null writer
// (every method a no-op), matching oMLX's get_request_log_writer() factory
// pattern: callers call this once per request and never branch on the
// enabled/disabled setting themselves.
std::unique_ptr<server_request_log_writer> server_request_log_create(
    const std::string & requests_dir,
    const std::string & request_id,
    const std::string & prompt_text);

std::unique_ptr<server_request_log_writer> server_request_log_create_null();
