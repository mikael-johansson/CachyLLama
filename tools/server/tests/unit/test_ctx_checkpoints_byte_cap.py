import os
import re
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


# Mirrors test_ssd_cache_ram_cap.py's LogReader: reads the live server
# stdout/stderr log (server.log_path, captured to a temp file -- see
# ServerProcess.start() in utils.py) so tests can assert on the narrative
# "erasing context checkpoint over byte cap" line added by
# server_context_impl::enforce_checkpoint_byte_cap() (tools/server/server-context.cpp),
# not just infer eviction indirectly.
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


# Repeats of a filler sentence long enough to produce a checkpoint of
# roughly 1+ MiB with the tiny stories260K test model (empirically ~620
# bytes/token for this model's KV state -- see test_ssd_cache_ram_cap.py's
# derivation of the same constant). deferred_create_final_checkpoint() only
# fires for prompts >= 64 tokens, so this clears that floor by a wide
# margin.
_REPEATS = 70


def run_turn(srv: ServerProcess, id_slot: int, n_predict: int = 4):
    text = "The quick brown fox jumps over the lazy dog. " * _REPEATS
    res = srv.make_request("POST", "/completion", data={
        "prompt": text,
        "id_slot": id_slot,
        "n_predict": n_predict,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res


def test_ctx_checkpoints_byte_cap_evicts_and_slot_stays_usable():
    # Regression test for the new per-slot byte cap on
    # slot.prompt.checkpoints (server_context_impl::enforce_checkpoint_byte_cap(),
    # tools/server/server-context.cpp), gated by --ctx-checkpoints-max-size
    # (common_params::n_ctx_checkpoints_max_size_mib). Before this fix, the
    # checkpoint ring was bounded only by count (--ctx-checkpoints,
    # default 32) -- never by bytes -- so a single busy slot's ring could
    # grow to many GiB on long conversations, since each checkpoint holds
    # real serialized KV state (data_tgt/data_dft/data_spec).
    #
    # Pin every request to the same slot (id_slot=0) and repeat the same
    # long prompt several times: each completion >= 64 tokens produces a
    # deferred final checkpoint (~1+ MiB at ~620 bytes/token for this tiny
    # model), and repeats accumulate in that one slot's ring with no
    # dedup -- so a handful of repeats comfortably exceeds a tight,
    # deliberately small combined byte budget.
    global server
    server.n_ctx = 8192
    server.n_slots = 1
    server.n_ctx_checkpoints_max_size_mib = 2  # tight: ~1 checkpoint's worth
    server.start()

    log = LogReader(server.log_path)

    n_turns = 5
    for _ in range(n_turns):
        run_turn(server, id_slot=0)

    log_text = log.drain()
    assert "erasing context checkpoint over byte cap" in log_text, (
        "expected the new byte-cap eviction narrative line to fire after "
        f"{n_turns} ~1+ MiB checkpoints on one slot with a 2 MiB cap; "
        f"server log:\n{log_text}"
    )
    assert "reason=bytecap" not in log_text, (
        "reason=bytecap is a cache-operations.log detail field, not part of "
        "the live-log narrative line -- if this fires, the assertion above "
        "is matching the wrong string"
    )

    # The slot must still be usable for a follow-up turn after eviction --
    # eviction must never corrupt or wedge the ring.
    res = run_turn(server, id_slot=0, n_predict=8)
    assert res.status_code == 200
    assert isinstance(res.body.get("content"), str)
    assert len(res.body["content"]) > 0

    server.stop()


def test_ctx_checkpoints_default_unlimited_never_evicts_on_bytes():
    # Confirms the default (n_ctx_checkpoints_max_size_mib = 0, i.e. the
    # --ctx-checkpoints-max-size flag left unset) preserves today's exact
    # behavior: no byte-cap eviction message ever appears, even when the
    # ring is large enough to trigger the pre-existing count-based cap.
    #
    # Force the count-based cap to actually do work for comparison (so this
    # test proves the byte-cap code path is inert, not just untriggered):
    # set --ctx-checkpoints very low (3) and drive more turns than that on
    # one slot, then confirm the "created final context checkpoint N of 3"
    # index never exceeds 3 -- proof the pre-existing count-based eviction
    # in deferred_create_final_checkpoint() kept firing (note: that loop,
    # unlike create_checkpoint()'s, doesn't emit its own log line on evict --
    # only the "N of M" index in the "created" line is observable here).
    # The new byte-cap eviction ("erasing context checkpoint over byte cap")
    # must never appear.
    global server
    server.n_ctx = 8192
    server.n_slots = 1
    server.n_ctx_checkpoints = 3
    # n_ctx_checkpoints_max_size_mib intentionally left unset (default: 0 = unlimited)
    server.start()

    log = LogReader(server.log_path)

    n_turns = 6
    for _ in range(n_turns):
        run_turn(server, id_slot=0)

    log_text = log.drain()
    indices = [int(n) for n in re.findall(r"context checkpoint (\d+) of 3", log_text)]
    assert indices, f"expected checkpoint-creation log lines; server log:\n{log_text}"
    assert max(indices) <= 3, (
        f"expected the pre-existing count-based cap (--ctx-checkpoints=3) to "
        f"keep the ring at <= 3 checkpoints across {n_turns} turns on one "
        f"slot, but saw index {max(indices)}; server log:\n{log_text}"
    )
    assert "erasing context checkpoint over byte cap" not in log_text, (
        "byte-cap eviction must never fire when --ctx-checkpoints-max-size "
        f"is left at its default (0 = unlimited); server log:\n{log_text}"
    )

    server.stop()
