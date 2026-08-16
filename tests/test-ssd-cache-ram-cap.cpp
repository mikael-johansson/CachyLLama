// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Regression test for the global hot/warm RAM cap on server_context_page_manager.
//
// Bug: each conversation's kv_ssd_cache independently auto-sized (or, with
// explicit --cache-ssd-hot-ram/--cache-ssd-warm-ram, independently applied)
// its OWN hot/warm RAM budget, once per conversation, in kv_ssd_init()
// (common/kv-ssd-cache.cpp). With N tracked conversations there was no
// shared ceiling: the sum of N independent per-conversation budgets could
// vastly exceed the intended total, which caused a real OOM-kill in
// production (confirmed via dmesg on a 32GB machine: anon-rss ~29.6GB,
// swap exhausted, killed by the kernel OOM killer).
//
// Fix (tools/server/server-context-page-manager.h/.cpp): the page manager
// now computes hot_max_size_bytes/warm_max_size_bytes ONCE, in its
// constructor, and enforces them GLOBALLY across every conversation via
// compute_hot_total_bytes_locked()/compute_warm_total_bytes_locked() and
// evict_conversations_for_ram_locked(), hooked right after every checkpoint
// store (mirroring the existing cold-tier global disk cap).
//
// This test cannot follow tests/test-ssd-cache-caps.cpp's pattern exactly:
// that test (and the two other tests/test-ssd-cache-*.cpp files) only
// exercise the low-level kv-ssd-cache.h API (kv_ssd_init/kv_ssd_store/
// kv_ssd_load) directly, because server_context_page_manager's public API
// (store_checkpoint_with_tokens/load_checkpoint) requires a real
// llama_context to serialize/restore KV state, which a standalone test
// binary with no loaded model cannot provide. The new methods under test
// here (compute_hot_total_bytes_locked, compute_warm_total_bytes_locked,
// evict_conversations_for_ram_locked) are also private by design (per the
// spec this was implemented against). To reach them without a model or
// context, this test uses a small friend-struct hook,
// llama::server_context_page_manager_test_hook (declared as a friend in
// server-context-page-manager.h, defined only in this file), which:
//   - drives the same kv_ssd_store()/kv_ssd_load() C-level calls the real
//     store/load path uses internally (server_ssd_cache::store just
//     serializes ctx state and forwards to kv_ssd_store -- see
//     tools/server/server-context-ssd-cache.cpp), so this test exercises
//     the exact same storage/eviction machinery with synthetic byte
//     buffers instead of real KV state, and
//   - calls the private accounting/eviction methods directly, holding
//     mutex_ exactly as store_checkpoint_with_tokens() does around its own
//     evict_conversations_for_size_locked()/evict_conversations_for_ram_locked()
//     calls.
//
// Because the methods under test are brand new (this test can't even
// compile against pre-fix code, unlike a pure behavior-change regression
// test), "confirm it fails pre-fix" is demonstrated in situ instead of via
// git-stash-and-rebuild: test_global_cap_enforced_across_conversations()
// runs the identical store pattern twice against two independent
// manager instances -- once WITHOUT calling evict_conversations_for_ram_locked()
// after each conversation's stores (reproducing the old bug: nothing ever
// enforces a global ceiling, so per-conversation local caps alone let the
// summed total badly exceed the intended budget) and once WITH it (the
// fix). See that test's comments for the exact assertions.
#undef NDEBUG

#include "server-context-page-manager.h"
#include "kv-ssd-cache.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace llama {

// Friend-struct test hook -- see file header comment. Declared as a friend
// of server_context_page_manager in server-context-page-manager.h; defined
// only here. Every method takes the mutex_ lock itself, exactly like the
// real call sites (store_checkpoint_with_tokens etc.) do, since the
// *_locked() methods it forwards to require the caller to already hold it.
struct server_context_page_manager_test_hook {
    // Get-or-create a conversation's raw kv_ssd_cache*, bypassing the
    // server_ssd_cache/llama_context-requiring public store() path.
    static kv_ssd_cache* get_or_create_conv_cache(server_context_page_manager& m, uint64_t conv_hash) {
        std::unique_lock<std::shared_mutex> lock(m.mutex_);
        server_ssd_cache* wrapper = m.get_or_create_cache(conv_hash);
        if (!wrapper) return nullptr;
        auto it = m.conv_caches_.find(conv_hash);
        return it == m.conv_caches_.end() ? nullptr : it->second.get();
    }

    static size_t hot_total(server_context_page_manager& m) {
        std::unique_lock<std::shared_mutex> lock(m.mutex_);
        return m.compute_hot_total_bytes_locked();
    }

    static size_t warm_total(server_context_page_manager& m) {
        std::unique_lock<std::shared_mutex> lock(m.mutex_);
        return m.compute_warm_total_bytes_locked();
    }

    // Simulates the hook added right after evict_conversations_for_size_locked()
    // in store_checkpoint_with_tokens().
    static void evict_ram(server_context_page_manager& m) {
        std::unique_lock<std::shared_mutex> lock(m.mutex_);
        m.evict_conversations_for_ram_locked();
    }

    static kv_ssd_cache* conv_cache_raw(server_context_page_manager& m, uint64_t conv_hash) {
        std::shared_lock<std::shared_mutex> lock(m.mutex_);
        auto it = m.conv_caches_.find(conv_hash);
        return it == m.conv_caches_.end() ? nullptr : it->second.get();
    }
};

} // namespace llama

using llama::server_context_page_manager;
using llama::server_context_page_manager_test_hook;
using llama::kv_eviction_config;

static int tests_run    = 0;
static int tests_failed = 0;

#define RUN(name) do {                              \
    printf("  %-60s ", #name);                      \
    fflush(stdout);                                 \
    tests_run++;                                    \
    try {                                           \
        test_##name();                              \
        printf("OK\n");                             \
    } catch (const std::exception & e) {            \
        printf("FAIL: %s\n", e.what());             \
        tests_failed++;                             \
    }                                                \
} while (0)

static fs::path make_scratch(const std::string & name) {
    fs::path p = fs::temp_directory_path() / ("ssd_ram_cap_test_" + name);
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p, ec);
    if (ec) {
        throw std::runtime_error("could not create scratch dir " + p.string() + ": " + ec.message());
    }
    return p;
}

// Fills a buffer with a pattern that identifies (conv, chunk) so a
// load-back can be verified byte-for-byte, not just by size.
static std::vector<uint8_t> make_pattern(uint64_t conv, uint32_t chunk, size_t size) {
    std::vector<uint8_t> data(size);
    uint64_t seed = conv * 1000003ULL + chunk;
    for (size_t i = 0; i < size; i++) {
        data[i] = (uint8_t)((seed + i * 2654435761ULL) & 0xFF);
    }
    return data;
}

static const size_t MIB = 1024ULL * 1024;
static const size_t HOT_BYTES  = 4 * MIB;
static const size_t WARM_BYTES = 2 * MIB;
static const size_t CHUNK_BYTES = 1 * MIB;
static const int CHUNKS_PER_CONV = 6;   // 6 MiB per conversation == HOT_BYTES + WARM_BYTES
static const int N_CONVERSATIONS = 5;   // 5 * 6 MiB = 30 MiB >> 6 MiB global cap if unenforced

// Stores CHUNKS_PER_CONV checkpoints of CHUNK_BYTES each into conv_hash's
// cache via the raw kv_ssd_store() API (no llama_context needed -- see file
// header comment). Returns the last checkpoint id stored and its data (for
// later byte-for-byte load verification).
static void store_conversation(server_context_page_manager& mgr, uint64_t conv_hash,
                                uint64_t& out_last_id, std::vector<uint8_t>& out_last_data) {
    kv_ssd_cache* cache = server_context_page_manager_test_hook::get_or_create_conv_cache(mgr, conv_hash);
    assert(cache != nullptr);

    for (int i = 0; i < CHUNKS_PER_CONV; i++) {
        std::vector<uint8_t> data = make_pattern(conv_hash, (uint32_t)i, CHUNK_BYTES);
        uint64_t id = kv_ssd_store(cache, /*slot_id=*/(uint32_t)i,
                                    data.data(), data.size(),
                                    /*pos_min=*/0, /*pos_max=*/999,
                                    /*n_tokens=*/1000, /*turn_id=*/(uint32_t)(i + 1),
                                    /*tokens=*/nullptr, /*tokens_size=*/0);
        assert(id != 0);
        out_last_id = id;
        out_last_data = data;
    }
}

// Test 1: constructor computes the GLOBAL budget from explicit
// --cache-ssd-hot-ram/--cache-ssd-warm-ram-equivalent config (cfg.auto_size
// = false, cfg.max_hot_bytes/max_warm_bytes set), not per-conversation.
static void test_explicit_global_budget_set_once() {
    kv_eviction_config cfg;
    cfg.auto_size      = false;
    cfg.max_hot_bytes  = HOT_BYTES;
    cfg.max_warm_bytes = WARM_BYTES;

    fs::path scratch = make_scratch("explicit_budget");
    server_context_page_manager mgr(scratch.string().c_str(), &cfg, /*n_tokens_total=*/0,
                                     /*max_cross_slot_checkpoints=*/1000);

    assert(mgr.hot_max_size_bytes  == HOT_BYTES);
    assert(mgr.warm_max_size_bytes == WARM_BYTES);

    fs::remove_all(scratch);
}

// Test 2: the actual bug/fix contrast. Two independent manager instances,
// identical store pattern (5 conversations x 6 MiB each = 30 MiB, vs a 6
// MiB combined global cap): one never calls the new eviction hook (bug:
// this is exactly what pre-fix code did -- nothing ever capped the sum
// across conversations), one calls it after every conversation's stores
// (fix, mirroring the real hook in store_checkpoint_with_tokens()).
static void test_global_cap_enforced_across_conversations() {
    kv_eviction_config cfg;
    cfg.auto_size      = false;
    cfg.max_hot_bytes  = HOT_BYTES;
    cfg.max_warm_bytes = WARM_BYTES;

    // --- Bug repro: no RAM-eviction hook called after stores ---
    {
        fs::path scratch = make_scratch("bug_repro");
        server_context_page_manager mgr(scratch.string().c_str(), &cfg, 0, 1000);

        for (int c = 0; c < N_CONVERSATIONS; c++) {
            uint64_t last_id; std::vector<uint8_t> last_data;
            store_conversation(mgr, /*conv_hash=*/0x9000ULL + (uint64_t)c, last_id, last_data);
            // Deliberately NOT calling evict_ram() here -- this is the bug.
        }

        size_t hot  = server_context_page_manager_test_hook::hot_total(mgr);
        size_t warm = server_context_page_manager_test_hook::warm_total(mgr);
        // Each conversation independently self-caps at its own local
        // hot+warm budget (HOT_BYTES+WARM_BYTES, unchanged existing
        // per-conversation make_room_hot()/demote_*() behavior), so without
        // a global eviction pass N conversations can together hold up to
        // N*(HOT_BYTES+WARM_BYTES). Assert this reproduces (>2x the cap is
        // an unambiguous reproduction, well short of the full Nx to avoid
        // test flakiness around exact demotion boundaries).
        if (hot + warm <= 2 * (HOT_BYTES + WARM_BYTES)) {
            throw std::runtime_error("expected bug to reproduce (total " +
                std::to_string((hot + warm) / MIB) + " MiB should exceed 2x cap of " +
                std::to_string(2 * (HOT_BYTES + WARM_BYTES) / MIB) + " MiB when the RAM eviction "
                "hook is never called)");
        }

        fs::remove_all(scratch);
    }

    // --- Fix: evict_conversations_for_ram_locked() called after every
    //     conversation's stores, exactly like store_checkpoint_with_tokens()
    //     does for every real checkpoint store. ---
    {
        fs::path scratch = make_scratch("fix_applied");
        server_context_page_manager mgr(scratch.string().c_str(), &cfg, 0, 1000);

        uint64_t first_conv_hash = 0xA000ULL;
        uint64_t first_conv_last_id = 0;
        std::vector<uint8_t> first_conv_last_data;

        for (int c = 0; c < N_CONVERSATIONS; c++) {
            uint64_t conv_hash = 0xA000ULL + (uint64_t)c;
            uint64_t last_id; std::vector<uint8_t> last_data;
            store_conversation(mgr, conv_hash, last_id, last_data);
            server_context_page_manager_test_hook::evict_ram(mgr);

            if (c == 0) {
                first_conv_last_id   = last_id;
                first_conv_last_data = last_data;
            }
        }

        size_t hot  = server_context_page_manager_test_hook::hot_total(mgr);
        size_t warm = server_context_page_manager_test_hook::warm_total(mgr);

        // Slack: up to ~2 checkpoints' worth, since the newest conversation's
        // own local cap equals the global cap exactly (every per-conversation
        // kv_ssd_config now carries the same honestly-shared figures), so
        // after eviction runs the resident total should sit at or just under
        // HOT_BYTES+WARM_BYTES, not accumulate across conversations.
        const size_t slack = 2 * CHUNK_BYTES;
        if (hot + warm > HOT_BYTES + WARM_BYTES + slack) {
            throw std::runtime_error("global RAM cap not enforced: total " +
                std::to_string((hot + warm) / MIB) + " MiB exceeds cap " +
                std::to_string((HOT_BYTES + WARM_BYTES) / MIB) + " MiB by more than " +
                std::to_string(slack / MIB) + " MiB slack");
        }
        if (hot > HOT_BYTES + slack) {
            throw std::runtime_error("hot tier alone exceeds its cap by more than slack");
        }
        if (warm > WARM_BYTES + slack) {
            throw std::runtime_error("warm tier alone exceeds its cap by more than slack");
        }

        // The first conversation stored must have been evicted from RAM
        // (oldest-active-first eviction order) since 5 conversations can't
        // all fit in a 6 MiB combined budget when each alone can occupy up
        // to 6 MiB.
        kv_ssd_cache* first_cache =
            server_context_page_manager_test_hook::conv_cache_raw(mgr, first_conv_hash);
        assert(first_cache != nullptr);
        {
            std::lock_guard<std::mutex> l(first_cache->mutex);
            assert(first_cache->hot_bytes == 0);
            assert(first_cache->warm_bytes == 0);
            assert(first_cache->hot_cache.empty());
            assert(first_cache->warm_cache.empty());
            for (const auto& [id, ckpt] : first_cache->index) {
                assert(ckpt.tier == KV_TIER_COLD);
            }
        }

        // Data-loss check: the evicted conversation's data must still load
        // correctly (and byte-identically) from disk via the normal
        // kv_ssd_load() path -- proving the coarse hot+warm -> cold RAM
        // demotion lost no data, because it was already durably fsync'd to
        // disk in kv_ssd_store() before ever being cached in RAM.
        std::vector<uint8_t> loaded;
        double io_ms = -1.0;
        bool ok = kv_ssd_load(first_cache, first_conv_last_id, loaded, nullptr, nullptr, &io_ms);
        assert(ok);
        assert(io_ms > 0.0); // proves this was actually a disk read, not a RAM hit
        if (loaded != first_conv_last_data) {
            throw std::runtime_error("loaded checkpoint data does not match what was stored "
                                      "(data loss after RAM eviction)");
        }

        fs::remove_all(scratch);
    }
}

// Test 3: hot_max_size_bytes == warm_max_size_bytes == 0 (unlimited) is a
// true no-op, matching cold_max_size_bytes's existing convention.
static void test_zero_cap_is_unlimited() {
    kv_eviction_config cfg;
    cfg.auto_size      = false;
    cfg.max_hot_bytes  = HOT_BYTES;
    cfg.max_warm_bytes = WARM_BYTES;

    fs::path scratch = make_scratch("zero_cap");
    server_context_page_manager mgr(scratch.string().c_str(), &cfg, 0, 1000);
    mgr.hot_max_size_bytes  = 0;
    mgr.warm_max_size_bytes = 0;

    for (int c = 0; c < N_CONVERSATIONS; c++) {
        uint64_t last_id; std::vector<uint8_t> last_data;
        store_conversation(mgr, 0xB000ULL + (uint64_t)c, last_id, last_data);
        server_context_page_manager_test_hook::evict_ram(mgr);
    }

    size_t hot  = server_context_page_manager_test_hook::hot_total(mgr);
    size_t warm = server_context_page_manager_test_hook::warm_total(mgr);
    // With both caps at 0 (unlimited), the global eviction pass must be a
    // no-op: every conversation's own local per-conversation cap still
    // applies (unchanged existing behavior), so the total should still
    // reach roughly N * (HOT_BYTES + WARM_BYTES), not be trimmed down to
    // one conversation's worth the way test 2's "fix_applied" case is.
    if (hot + warm <= 2 * (HOT_BYTES + WARM_BYTES)) {
        throw std::runtime_error("expected hot_max_size_bytes=0/warm_max_size_bytes=0 to be a "
                                  "no-op (global eviction should not have trimmed the total)");
    }

    fs::remove_all(scratch);
}

int main(void) {
    printf("test-ssd-cache-ram-cap: regression suite for the global hot/warm RAM cap\n");
    printf("==========================================================================\n\n");

    RUN(explicit_global_budget_set_once);
    RUN(global_cap_enforced_across_conversations);
    RUN(zero_cap_is_unlimited);

    printf("\n==========================================================================\n");
    printf("Ran %d tests, %d failed\n", tests_run, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
