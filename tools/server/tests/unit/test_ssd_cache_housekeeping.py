import os

import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0


def hex16_dirs(base: str) -> set[str]:
    """Top-level conversation/user-cache directory names directly under
    `base` -- i.e. entries that are exactly 16 hex chars and are directories.
    Mirrors scan_on_disk_conversations_locked()'s naming filter in
    server-context-page-manager.cpp: this excludes "u" (the user-cache
    namespace), "lost+found", and sys-<hash>.bin (a *file*, not a directory,
    used by the separate system-prompt cache)."""
    if not os.path.isdir(base):
        return set()
    out = set()
    for name in os.listdir(base):
        if len(name) == 16 and all(c in "0123456789abcdefABCDEF" for c in name):
            if os.path.isdir(os.path.join(base, name)):
                out.add(name)
    return out


def dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def run_conversation(srv: ServerProcess, marker: str, n_predict: int = 4):
    # The SSD cache's deferred final checkpoint only fires for prompts >= 64
    # tokens (server-context.cpp: deferred_create_final_checkpoint returns
    # early below that). Repeating a filler sentence gets comfortably past
    # that threshold while staying under the tiny test model's small n_ctx
    # (512 total / 2 slots = 256 tokens per slot).
    text = f"{marker} " + ("The quick brown fox jumps over the lazy dog. " * 8)
    res = srv.make_request("POST", "/completion", data={
        "prompt": text,
        "n_predict": n_predict,
    })
    assert res.status_code == 200
    return res


def test_cache_ssd_max_conversations_survives_restart(tmp_path):
    # Regression test for the SSD-cache housekeeping bug: get_or_create_cache()
    # enforced --cache-ssd-max-conversations by checking conv_caches_.size(),
    # an in-memory map populated only for conversations touched during the
    # *current process's* lifetime. A server restart starts that map empty,
    # so every directory created by a previous process instance became
    # permanently invisible to the cap -- it was never counted, and never
    # considered for eviction, so it stayed on disk forever.
    cache_dir = str(tmp_path / "cache")

    global server
    server.cache_ssd_path = cache_dir
    server.cache_ssd_max_conversations = 2
    server.cache_ssd_no_fsync = True
    server.start()

    for i in range(3):
        run_conversation(server, f"run1-conv-{i}")

    dirs_after_run1 = hex16_dirs(cache_dir)
    assert len(dirs_after_run1) <= 2, (
        f"cap violated within a single run (no restart involved yet): {dirs_after_run1}"
    )

    server.stop()

    # Simulate a server restart: a brand-new process, pointed at the same
    # cache directory, with the same cap. Under the bug, the new process's
    # conv_caches_ starts empty, so get_or_create_cache() has no idea the
    # cap is already "full" on disk -- it creates new conversation
    # directories without ever evicting the ones left behind by run 1.
    server2 = ServerPreset.tinyllama2()
    server2.temperature = 0.0
    server2.cache_ssd_path = cache_dir
    server2.cache_ssd_max_conversations = 2
    server2.cache_ssd_no_fsync = True
    server2.start()

    # A single new conversation is enough to expose the bug: disk already
    # has 2 (== cap) directories from run 1, so this one must trigger an
    # eviction of one of them to stay within the cap.
    run_conversation(server2, "run2-conv-0")

    dirs_after_first_run2_conv = hex16_dirs(cache_dir)
    assert len(dirs_after_first_run2_conv) <= 2, (
        f"on-disk conversation count exceeded --cache-ssd-max-conversations "
        f"right after restart: {dirs_after_first_run2_conv}"
    )
    assert not dirs_after_run1.issubset(dirs_after_first_run2_conv), (
        "none of run 1's on-disk conversation directories were evicted -- "
        "the new process never discovered them, meaning the cap is being "
        "enforced against an empty in-memory map instead of the real "
        f"on-disk state (run1={dirs_after_run1}, "
        f"after_restart={dirs_after_first_run2_conv})"
    )

    # Drive a few more new conversations through the restarted instance and
    # confirm the cap keeps holding steady turn after turn.
    for i in range(1, 4):
        run_conversation(server2, f"run2-conv-{i}")
        dirs_now = hex16_dirs(cache_dir)
        assert len(dirs_now) <= 2, (
            f"on-disk conversation count exceeded cap after restart: {dirs_now}"
        )

    server2.stop()


def test_cache_ssd_cold_maxsize_survives_restart(tmp_path):
    # Regression test for the same underlying bug on the byte-cap side:
    # compute_cold_total_bytes_locked() (backing --cache-ssd-cold-maxsize)
    # summed bytes from conv_caches_/user_caches_' in-memory indexes, which
    # are empty for conversations created by a previous process instance.
    # A restart made those directories invisible to the byte cap too, so
    # total on-disk usage could grow without bound across restarts even
    # with the cap configured.
    cache_dir = str(tmp_path / "cache")

    global server
    server.cache_ssd_path = cache_dir
    # Count cap disabled (effectively) for run 1 -- this run just
    # accumulates conversation directories on disk unconstrained, standing
    # in for "state left behind by a prior server instance/restart".
    server.cache_ssd_max_conversations = 1000
    server.cache_ssd_no_fsync = True
    server.start()

    for i in range(6):
        run_conversation(server, f"run1-conv-{i}")

    dirs_after_run1 = hex16_dirs(cache_dir)
    bytes_after_run1 = dir_bytes(cache_dir)
    assert len(dirs_after_run1) == 6
    assert bytes_after_run1 > 0

    server.stop()

    # New process, same cache dir. --cache-ssd-cold-maxsize is specified in
    # MiB; 1 MiB is comfortably above a single tiny-model checkpoint
    # (~170 KiB, measured empirically) but well below run 1's 6-conversation
    # total, so the very first store in run 2 must evict some of run 1's
    # directories to get under the cap.
    cap_mib = 1
    cap_bytes = cap_mib * 1024 * 1024

    server2 = ServerPreset.tinyllama2()
    server2.temperature = 0.0
    server2.cache_ssd_path = cache_dir
    server2.cache_ssd_max_conversations = 1000
    server2.cache_ssd_cold_maxsize = cap_mib
    server2.cache_ssd_no_fsync = True
    server2.start()

    run_conversation(server2, "run2-conv-0")

    dirs_after_first_run2_conv = hex16_dirs(cache_dir)
    assert not dirs_after_run1.issubset(dirs_after_first_run2_conv), (
        "none of run 1's on-disk conversation directories were evicted -- "
        "the byte cap never discovered them, meaning "
        "compute_cold_total_bytes_locked() is still summing only the "
        f"in-memory index instead of real on-disk usage (run1="
        f"{dirs_after_run1}, after_restart={dirs_after_first_run2_conv})"
    )

    # Drive a few more conversations and confirm on-disk usage settles down
    # near the cap instead of growing without bound.
    for i in range(1, 4):
        run_conversation(server2, f"run2-conv-{i}")

    server2.stop()

    bytes_after_run2 = dir_bytes(cache_dir)
    # Eviction runs *after* each store (see evict_conversations_for_size_locked's
    # "brief overshoot" comment), so the true total can exceed the cap by
    # roughly one conversation's worth at any instant. A generous multiple
    # of the cap is still far below what unbounded growth across the two
    # runs (10 conversations x ~170 KiB never evicted) would leave behind.
    assert bytes_after_run2 < cap_bytes * 4, (
        f"on-disk cold-tier bytes ({bytes_after_run2}) grew far beyond the "
        f"configured cap ({cap_bytes}) after a restart -- run 1's "
        f"directories were not correctly discovered/evicted"
    )
