import glob
import os
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


SHORT_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
""".strip()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


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
    # matter what cache operations happen.
    global server
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

    would_be_path = cache_ops_log_path(str(tmp_path / "requests"))
    assert not os.path.isfile(would_be_path)


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


@pytest.fixture
def mmproj_server():
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
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
