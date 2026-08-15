import glob
import os
import tempfile
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


SHORT_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
""".strip()


# A prompt long enough to clear deferred_create_final_checkpoint()'s 64-token
# floor (server-context.cpp) so a completion reliably produces a
# checkpoint_create + ssd_store pair, not just the response itself. Same
# repeated-sentence pattern as test_ssd_cache_housekeeping.py's
# run_conversation(), sized to stay comfortably under the tiny test model's
# n_ctx once split across a few slots.
LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 8


# Mirrors test_kv_keep_only_active.py's LogReader: reads the live server
# stdout/stderr log (server.log_path, captured to a temp file -- see
# ServerProcess.start() in utils.py) incrementally, for asserting on the
# narrative "[Conversation ...]" / "[Slot ...]" / "[Server]" lines this
# change adds alongside the structured cache-operations.log rows.
class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)


def cache_ops_log_path(request_logging_dir: str) -> str:
    return os.path.join(request_logging_dir, "cache-operations.log")


def wait_for_file(path: str, timeout: float = 10.0):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.isfile(path):
            return
        time.sleep(0.1)
    raise TimeoutError(f"{path} never appeared within {timeout}s")


def wait_for_marker_in_file(path: str, marker: str, timeout: float = 10.0) -> str:
    start = time.time()
    content = ""
    while time.time() - start < timeout:
        if os.path.isfile(path):
            with open(path, "r") as f:
                content = f.read()
            if marker in content:
                return content
        time.sleep(0.1)
    raise TimeoutError(f"'{marker}' never appeared in {path} (last content:\n{content})")


def request_log_files(directory: str):
    return sorted(glob.glob(os.path.join(directory, "*.log")))


def test_cache_operations_log_disabled_by_default(tmp_path):
    # no --request-logging-dir flag passed -> no cache-operations.log, no
    # matter what cache operations happen. Same flag also gates the
    # narrative "[Slot N] ..." live-log lines (note_cache_event() in
    # server-context.cpp) -- verify those are silent too, not just the
    # structured file.
    global server
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200

    would_be_path = cache_ops_log_path(str(tmp_path / "requests"))
    assert not os.path.isfile(would_be_path)
    assert "[Slot 1]" not in log.drain()


def test_cache_operations_log_slot_save_erase(tmp_path):
    # NOTE: slot restore is intentionally exercised in a separate test
    # (test_cache_operations_log_slot_restore, below) against the tinygemma3
    # preset rather than here. Restoring a saved tinyllama2/stories260K slot
    # crashes the server outright -- reproduced identically on an unmodified
    # checkout of this branch (confirmed via `git stash`), and
    # tools/server/tests/unit/test_slot_save.py::test_slot_save_restore hits
    # the exact same pre-existing crash in this environment. Unrelated to
    # cache-operations logging; not something this change should paper over.
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    log_path = cache_ops_log_path(req_dir)
    content = wait_for_marker_in_file(log_path, "slot_erase")

    # header row present, with the documented fixed-width columns
    header_line = content.splitlines()[0]
    assert "TIMESTAMP" in header_line
    assert "OPERATION" in header_line
    assert "LOCATION" in header_line
    assert "TOKENS" in header_line
    assert "BYTES" in header_line
    assert "DURATION" in header_line
    assert "DETAIL" in header_line

    assert "slot_save" in content
    assert "slot_erase" in content
    # detail column carries the slot id / filename
    assert "slot=1" in content
    assert "file=slot1.bin" in content
    # duration values are rendered with their unit, never bare numbers
    assert " ms" in content

    # narrative counterpart in the live server log (console/stdout) --
    # slot save/restore/erase aren't tied to an in-flight completion request
    # (note_cache_event(nullptr, ...)), so they fall back to "[Slot N]"
    # rather than "[Conversation ...]" (see server_cache_event::slot_id).
    log_text = LogReader(server.log_path).drain()
    assert "[Slot 1] Saving 84 tokens to disk file slot1.bin" in log_text
    assert "[Slot 1] Erased 84 tokens from cache" in log_text


@pytest.fixture
def mmproj_server():
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    fd, mm_server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    return mm_server


def test_cache_operations_log_slot_restore(tmp_path, mmproj_server):
    # tinygemma3 restore round-trip is proven to work in
    # test_slot_save.py::test_slot_save_restore_text_only_on_multimodal;
    # mirrored here (pure-text prompt) to verify the slot_restore row without
    # the pre-existing tinyllama2/stories260K restore crash noted above.
    req_dir = str(tmp_path / "requests")
    mmproj_server.request_logging_dir = req_dir
    mmproj_server.start()

    res = mmproj_server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = mmproj_server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0

    res = mmproj_server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    log_path = cache_ops_log_path(req_dir)
    content = wait_for_marker_in_file(log_path, "slot_restore")
    assert "slot=0" in content
    assert "file=mm_slot1.bin" in content

    log_text = LogReader(mmproj_server.log_path).drain()
    assert f"[Slot 0] Loading {n_saved} tokens from disk file mm_slot1.bin" in log_text


def test_cache_operations_log_ctx_shift(tmp_path):
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.n_ctx = 512
    server.n_slots = 2
    server.n_predict = 128
    server.enable_ctx_shift = True
    server.start()

    res = server.make_request("POST", "/completion", data={
        "n_predict": 96,
        "prompt": SHORT_TEXT,
    }, headers={"X-Request-Id": "test-ctx-shift-cache-log"})
    assert res.status_code == 200
    assert res.body["truncated"] is True

    # shared, server-lifetime cache-operations.log gets a ctx_shift row
    log_path = cache_ops_log_path(req_dir)
    content = wait_for_marker_in_file(log_path, "ctx_shift")
    assert "ram" in content
    assert "kept=" in content and "discarded=" in content

    # narrative counterpart in the live server log: tied to a slot currently
    # processing a request, so scoped by conv_hash ("[Conversation ...]"),
    # not "[Slot N]" -- see narrate_cache_event() in server-context.cpp.
    log_text = LogReader(server.log_path).drain()
    assert "[Conversation " in log_text
    assert "Context shift: dropping " in log_text and "keeping " in log_text

    # the same event also lands in this request's own per-request log, in a
    # "=== CACHE ===" section positioned after the response and before the
    # performance footer (see server-request-log.h's note_cache_event()).
    files = request_log_files(req_dir)
    req_log_files = [f for f in files if os.path.basename(f) != "cache-operations.log"]
    assert len(req_log_files) == 1
    req_content = wait_for_marker_in_file(req_log_files[0], "=== PERFORMANCE ===")

    assert "test-ctx-shift-cache-log" in req_content
    assert "=== CACHE ===" in req_content
    cache_section = req_content.split("=== CACHE ===", 1)[1].split("=== PERFORMANCE ===", 1)[0]
    assert "ctx_shift" in cache_section
    # per-request section omits TIMESTAMP/DETAIL columns (redundant with the
    # file's own surrounding context) -- header should not contain them
    cache_header = cache_section.strip().splitlines()[0]
    assert "TIMESTAMP" not in cache_header
    assert "DETAIL" not in cache_header
    assert "OPERATION" in cache_header

    # response/cache/performance ordering
    assert req_content.index("=== RESPONSE (streaming) ===") < req_content.index("=== CACHE ===")
    assert req_content.index("=== CACHE ===") < req_content.index("=== PERFORMANCE ===")


def test_cache_operations_log_model_load(tmp_path):
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.start()

    log_path = cache_ops_log_path(req_dir)
    content = wait_for_marker_in_file(log_path, "model_load")
    assert "model_load" in content

    # narrative counterpart: logged directly in server.cpp (this event fires
    # before request handling -- and hence note_cache_event()'s slot-based
    # dispatch -- exists at all), always scoped "[Server]".
    log_text = wait_for_marker_in_file(server.log_path, "[Server] Loaded model ")
    assert "[Server] Loaded model " in log_text and " ms)" in log_text


def hex16_conv_dirs(cache_dir: str):
    """Top-level per-conversation SSD cache directories directly under
    `cache_dir` -- entries named with exactly 16 hex chars, mirroring
    server-context-page-manager.cpp's on-disk naming convention (see
    test_ssd_cache_housekeeping.py's identical hex16_dirs() helper)."""
    if not os.path.isdir(cache_dir):
        return []
    out = []
    for name in os.listdir(cache_dir):
        if len(name) == 16 and all(c in "0123456789abcdefABCDEF" for c in name):
            full = os.path.join(cache_dir, name)
            if os.path.isdir(full):
                out.append(full)
    return out


def test_cache_operations_log_narrative_ssd_store_restore(tmp_path):
    # Narrative counterpart for the per-conversation SSD-backed cache: verify
    # both "Saving ... on DISK" (ssd_store) and "Loading ... [in RAM]"
    # (ssd_restore -- served from the hot tier immediately after the store,
    # see kv_ssd_load()'s hot_cache check in common/kv-ssd-cache.cpp) appear
    # in the live server log, scoped by conv_hash the same way the RAM
    # checkpoint narrative is.
    global server
    req_dir = str(tmp_path / "requests")
    cache_dir = str(tmp_path / "ssd-cache")
    server.request_logging_dir = req_dir
    server.cache_ssd_path = cache_dir
    server.n_ctx = 2048
    server.n_slots = 2
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    log_path = cache_ops_log_path(req_dir)
    wait_for_marker_in_file(log_path, "ssd_store")

    # same conversation content, different (previously untouched) slot ->
    # cold-start SSD restore path (find_and_load_checkpoint() in
    # server-context.cpp).
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    wait_for_marker_in_file(log_path, "ssd_restore")

    log_text = LogReader(server.log_path).drain()
    assert "[Conversation " in log_text
    assert "on DISK" in log_text
    assert "Loading " in log_text and "from snapshot " in log_text


def test_cache_operations_log_ssd_restore_fail(tmp_path):
    # Regression test for the "silent restore failure" gap: previously, when
    # find_and_load_checkpoint()'s SSD restore was *attempted* (a candidate
    # checkpoint existed in the index) but failed (missing/corrupted file,
    # I/O error), there was no log line at all -- only the success path
    # (`if (ssd_restore_ok) { note_cache_event(...) }`) was instrumented.
    # Now a failed-but-attempted restore logs an SLT_WRN narrative line and
    # an "ssd_restore_fail" cache-operations.log row (see
    # server-context.cpp's find_and_load_checkpoint() call site).
    #
    # Reproduction: store a checkpoint, then advance the SSD turn counter
    # (server-context.cpp hardcodes hot_turns=2, warm_turns=4 via
    # cfg.turn_inactivity_threshold = 2) well past both thresholds with
    # filler completions on an unrelated slot, so the checkpoint demotes
    # hot->warm->cold (cold = disk-only, no in-process RAM fallback -- see
    # kv_ssd_load() in common/kv-ssd-cache.cpp). Deleting the on-disk file
    # only causes a load failure once the checkpoint is cold; while it's
    # still hot/warm, kv_ssd_load() serves it straight from the in-process
    # hot_cache/warm_cache maps regardless of what's on disk (confirmed by
    # hand while developing this test).
    global server
    req_dir = str(tmp_path / "requests")
    cache_dir = str(tmp_path / "ssd-cache")
    server.request_logging_dir = req_dir
    server.cache_ssd_path = cache_dir
    server.n_ctx = 2048
    server.n_slots = 3
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    log_path = cache_ops_log_path(req_dir)
    wait_for_marker_in_file(log_path, "ssd_store")

    # Advance turns well past hot_turns(2) + warm_turns(4) via short filler
    # completions (< 64 tokens, so they don't create checkpoints of their
    # own -- server-context.cpp's deferred_create_final_checkpoint() floor)
    # on a different slot.
    for i in range(8):
        res = server.make_request("POST", "/completion", data={
            "prompt": f"filler prompt {i} to advance the ssd turn counter",
            "id_slot": 1,
            "cache_prompt": False,
        })
        assert res.status_code == 200

    conv_dirs = hex16_conv_dirs(cache_dir)
    target_dir = None
    for d in conv_dirs:
        if glob.glob(os.path.join(d, "ckpt-*.bin")):
            target_dir = d
            break
    assert target_dir is not None, f"no checkpoint file found under {conv_dirs}"

    ckpts = sorted(glob.glob(os.path.join(target_dir, "ckpt-*.bin")))
    assert len(ckpts) >= 1
    os.remove(ckpts[-1])  # delete the (now cold, disk-only) checkpoint data

    # fresh slot, same conversation content -> cold-start restore attempt
    # against the now-missing file.
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT,
        "id_slot": 2,
        "cache_prompt": True,
    })
    # the request itself must still succeed -- a failed restore falls back
    # to a full re-prefill, it does not fail the request.
    assert res.status_code == 200

    content = wait_for_marker_in_file(log_path, "ssd_restore_fail")
    assert "ssd_restore_fail" in content

    log_text = LogReader(server.log_path).drain()
    assert "SSD cache restore attempted but failed" in log_text
    assert "Failed to load snapshot" in log_text
    assert "falling back to full re-prefill" in log_text
