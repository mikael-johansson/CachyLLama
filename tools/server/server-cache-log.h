#pragma once

// Shared, server-lifetime cache-operations log ("cache logging").
//
// Companion to server-request-log.{h,cpp} (the per-request feature), reusing
// the same opt-in flag: when --request-logging-dir is set, every cache
// operation the server performs -- slot save/restore/erase, context shift,
// context checkpoint create/evict, SSD tiered-cache store/load/promote/
// demote/evict, system-prompt-cache store/hit, and model load -- gets one
// row appended to <request-logging-dir>/cache-operations.log for the
// lifetime of the process, in addition to (for cache events that happen
// synchronously while processing a specific completion request) a
// "=== CACHE ===" section in that request's own log file (see
// server_request_log_writer::note_cache_event()).
//
// Unlike the per-request log (one file per request, closed at the end of
// that request), this is a single running table opened once and appended to
// forever -- see REQUEST_LOGGING_SPEC.md's cache-operations design ("route
// by scope") for the full rationale.
//
// Best-effort throughout: a failure to open/write the file never throws and
// never affects cache functionality -- callers can fire-and-forget.

#include <cstdint>
#include <optional>
#include <string>

// Fixed-precision decimal formatting, e.g. fmt_fixed(12.3456, 3) == "12.346".
// Shared with server-request-log.cpp so both the performance footer and the
// cache-operations tables use one formatting convention.
std::string fmt_fixed(double v, int prec);

// Render `bytes` as a human-readable size ("4.2 MB", "512.0 KB", "1.3 GB"),
// binary-scaled (1024-based). Returns "n/a" when `bytes` is unset -- see
// REQUEST_LOGGING_SPEC.md Part 1: omit uncomputable fields, never fake them.
std::string format_bytes_human(std::optional<int64_t> bytes);

// One row of the cache-operations table.
struct server_cache_event {
    // Short verb-ish operation name, e.g. "slot_save", "slot_restore",
    // "slot_erase", "ctx_shift", "checkpoint_create", "checkpoint_evict",
    // "ssd_store", "ssd_restore", "ssd_maintenance", "sys_cache_store",
    // "sys_cache_hit", "model_load".
    std::string operation;
    // Where the operation happened / what it touched, e.g. "disk", "ram",
    // "hot->warm", "warm->cold", "cold (disk)".
    std::string location;
    std::optional<int64_t> tokens;
    std::optional<int64_t> bytes;
    double duration_ms = 0.0;
    // Free-form "key=value key2=value2" detail, e.g. "slot=0 file=abc.bin".
    // Only rendered in the shared cache-operations.log; the per-request
    // "=== CACHE ===" section omits it as redundant with that file's own
    // surrounding request context (see server-request-log.h).
    std::string detail;
};

// Lazily opens (append mode, never truncated on restart) `<dir>/cache-operations.log`
// on the first call to server_cache_log_note() after this. A no-op (and
// every later server_cache_log_note() call becomes a silent no-op too) when
// `dir` is empty. Call once at server startup, e.g. right after model load;
// safe to call more than once (subsequent calls are ignored). Never throws.
void server_cache_log_init(const std::string & request_logging_dir);

// Append one row to the shared cache-operations.log. Best-effort: swallows
// any failure (disk full, permissions, ...) rather than throwing or
// affecting cache functionality. No-op if server_cache_log_init() was never
// called or was called with an empty directory. Safe to call from any
// thread (internally mutex-guarded).
void server_cache_log_note(const server_cache_event & ev);

// Render one row (no trailing newline) of the fixed-width cache-operations
// table. `with_timestamp_and_detail` selects the full 6-column layout
// (TIMESTAMP, OPERATION, LOCATION, TOKENS, BYTES, DURATION, DETAIL) used by
// the shared cache-operations.log vs. the 5-column layout (OPERATION,
// LOCATION, TOKENS, BYTES, DURATION) used by the per-request "=== CACHE ==="
// section.
std::string server_cache_log_format_row(const server_cache_event & ev, bool with_timestamp_and_detail);
std::string server_cache_log_header_row(bool with_timestamp_and_detail);
