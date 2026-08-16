import os
import tempfile
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)


# Mirrors test_cache_operations_logging.py's LogReader: reads the live
# server stdout/stderr log (server.log_path, captured to a temp file -- see
# ServerProcess.start() in utils.py) so tests can assert on the narrative
# "SSD cache: evicted conversation ... from RAM" line added by
# evict_conversations_for_ram_locked() (tools/server/server-context-page-manager.cpp),
# not just infer it indirectly.
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


def cache_ops_log_path(request_logging_dir: str) -> str:
    return os.path.join(request_logging_dir, "cache-operations.log")


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


# Repeats of a filler sentence long enough to produce a checkpoint of
# roughly 1+ MiB with the tiny stories260K test model (empirically ~620
# bytes/token for this model's KV state -- see test_ssd_cache_housekeeping.py's
# ~170 KiB/246-token measurement). This is deliberately large: the fixed
# (non-configurable) hot_turns=2/warm_turns=4 turn-based tier-aging in
# common/kv-ssd-cache.cpp would otherwise demote every conversation to cold
# "naturally" within a handful of turns regardless of the RAM cap under
# test here (the shared ssd_turn_counter advances once per request across
# *all* conversations), making it impossible to tell natural aging apart
# from the new global-cap eviction with small, ~150 KiB checkpoints. With
# ~1.2 MiB checkpoints, two conversations already exceed the ~2 MiB
# combined hot+warm cap configured below while the turn counter is still
# only at 2 -- well before hot_turns=2 would even make the first
# conversation eligible for its own local demotion -- so an eviction
# happening that early can only be the new global cap.
_REPEATS = 70


def run_conversation(srv: ServerProcess, marker: str, n_predict: int = 4):
    # The SSD cache's deferred final checkpoint only fires for prompts >= 64
    # tokens (server-context.cpp: deferred_create_final_checkpoint returns
    # early below that threshold); _REPEATS clears that floor by a wide
    # margin.
    text = f"{marker} " + ("The quick brown fox jumps over the lazy dog. " * _REPEATS)
    res = srv.make_request("POST", "/completion", data={
        "prompt": text,
        "n_predict": n_predict,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res


def _configure_ram_cap_server(srv: ServerProcess, cache_dir: str, req_dir: str) -> None:
    srv.cache_ssd_path = cache_dir
    srv.cache_ssd_max_conversations = 1000  # not under test here
    srv.cache_ssd_no_fsync = True
    # Deliberately tight, and paired with the large _REPEATS prompts above:
    # ~1.2 MiB/checkpoint means 2 conversations alone already exceed a
    # combined 1 MiB hot + 1 MiB warm budget.
    srv.cache_ssd_hot_ram = 1
    srv.cache_ssd_warm_ram = 1
    # n_slots=8 (> the 6 conversations driven below) so that, after a
    # process restart, every repeated conversation can be assigned its own
    # never-before-touched slot instead of reusing one that already holds a
    # *different* conversation's state -- find_and_load_checkpoint()'s
    # cold-start SSD path (server-context.cpp) is only even attempted when
    # a slot has literally no prior prompt tokens at all, so slot reuse
    # across distinct conversations would silently skip it for everything
    # but the first couple of repeats. n_ctx=24576 gives each of the 8
    # slots ~3000 tokens, comfortably above the ~2032-token _REPEATS
    # prompts; the tinyllama2 preset's default 512/2=256-token slots are
    # far too small for that.
    srv.n_ctx = 24576
    srv.n_slots = 8
    srv.request_logging_dir = req_dir


def test_ssd_cache_ram_cap_evicts_and_stays_retrievable(tmp_path):
    # Regression test for the global hot/warm RAM cap
    # (server_context_page_manager::hot_max_size_bytes/warm_max_size_bytes,
    # enforced by evict_conversations_for_ram_locked()). Before this fix,
    # each conversation's kv_ssd_cache independently auto-sized (or, with
    # explicit --cache-ssd-hot-ram/--cache-ssd-warm-ram, independently
    # applied) its own hot/warm RAM budget -- there was no shared ceiling,
    # so N tracked conversations could together hold up to N times the
    # intended RAM budget. This end-to-end test drives several real
    # conversations through a live server with a deliberately tight
    # combined RAM budget and confirms (a) the new RAM-tier eviction
    # narrative log line fires, and (b) every conversation -- including
    # ones evicted from RAM -- remains correctly retrievable afterward via
    # a disk restore, not an error or silent full re-prefill miss.
    #
    # Verifying (b) needs a genuine cold-start SSD restore, not a live-slot
    # cache hit: find_and_load_checkpoint() (server-context.cpp) is only
    # even attempted when a slot has *never* held any prior state
    # (slot.prompt.n_tokens() == 0), which with only 2 live model slots and
    # 6 distinct conversations can't be guaranteed request-by-request within
    # a single running process. A real process restart (mirroring
    # test_ssd_cache_housekeeping.py's pattern) sidesteps this cleanly:
    # every slot in the second process starts genuinely empty, so every
    # repeat request is forced through the cold-start SSD path, and RAM
    # state never survives a restart anyway -- so every successful repeat
    # is unambiguously a real disk read, not a RAM hit left over from
    # server 1's eviction bookkeeping.
    cache_dir = str(tmp_path / "cache")
    req_dir = str(tmp_path / "requests")

    global server
    _configure_ram_cap_server(server, cache_dir, req_dir)
    server.start()

    log = LogReader(server.log_path)

    # A handful of conversations is enough: at ~1.2 MiB/checkpoint each,
    # this comfortably exceeds the ~2 MiB combined hot+warm budget above.
    n_conversations = 6
    markers = [f"ram-cap-conv-{i}" for i in range(n_conversations)]
    for marker in markers:
        run_conversation(server, marker)

    # The new global RAM-tier eviction must have fired at least once: this
    # many conversations' worth of hot-tier data (created one at a time,
    # each immediately stored to SSD and cached hot) cannot all fit under a
    # combined 2 MiB hot+warm budget.
    log_text = log.drain()
    assert "from RAM" in log_text and "evicted conversation" in log_text, (
        "expected the new global RAM-cap eviction narrative line "
        "('SSD cache: evicted conversation ... from RAM (hot+warm -> cold, ...)') "
        f"to fire with only a 1 MiB hot + 1 MiB warm budget across "
        f"{n_conversations} conversations; server log:\n{log_text}"
    )

    server.stop()

    # Fresh process, same on-disk cache dir -- see the docstring above for
    # why this is needed to get a deterministic, provable disk restore for
    # every conversation instead of an ambiguous mix of live-slot hits.
    server2 = ServerPreset.tinyllama2()
    server2.temperature = 0.0
    fd, server2.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    _configure_ram_cap_server(server2, cache_dir, req_dir)
    server2.start()

    # Every conversation, including whichever were evicted from RAM before
    # the restart, must still be retrievable: repeat the exact same prompt
    # (same leading tokens -> same conv_hash -> same on-disk conversation
    # directory) for every conversation and confirm a correct, non-error
    # response.
    for marker in markers:
        res = run_conversation(server2, marker)
        assert res.status_code == 200
        assert isinstance(res.body.get("content"), str)
        assert len(res.body["content"]) > 0

    # Confirm these repeats were actually served via a real SSD disk
    # restore (cache-consistent, not a silent full re-prefill that merely
    # looks like a normal response): cache-operations.log gets an
    # "ssd_restore" row whenever find_and_load_checkpoint() succeeds, and
    # the narrative live-log line distinguishes a disk hit ("on DISK") from
    # a RAM hit ("in RAM") -- see server-context.cpp's ssd_restore
    # narrative branch. Every repeat above must have taken this path (fresh
    # process => every slot starts empty => the cold-start branch always
    # runs), so require one ssd_restore per conversation, not just one.
    ops_log = cache_ops_log_path(req_dir)
    wait_for_marker_in_file(ops_log, "ssd_restore")
    with open(ops_log) as f:
        ops_content = f.read()
    n_restores = ops_content.count("ssd_restore")
    assert n_restores >= n_conversations, (
        f"expected >= {n_conversations} ssd_restore rows in {ops_log} (one per "
        f"repeated conversation after the restart), found {n_restores}:\n{ops_content}"
    )

    log2_text = LogReader(server2.log_path).drain()
    assert "on DISK" in log2_text, (
        "expected every post-restart restore to be served from disk (narrative "
        "'... [on DISK] ...' line) since RAM state never survives a process "
        f"restart; server log:\n{log2_text}"
    )

    server2.stop()
