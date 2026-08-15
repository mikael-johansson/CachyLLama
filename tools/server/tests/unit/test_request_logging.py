import glob
import os
import time

import pytest
import requests

from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def log_files(directory: str):
    # cache-operations.log (see test_cache_operations_logging.py) lives in
    # this same --request-logging-dir directory by design (a shared,
    # server-lifetime file, not a per-request one) -- exclude it so callers
    # here only ever see per-request log files.
    files = glob.glob(os.path.join(directory, "*.log"))
    files = [f for f in files if os.path.basename(f) != "cache-operations.log"]
    return sorted(files)


def wait_for_log_file(directory: str, timeout: float = 10.0):
    """Poll for a *.log file to appear (writer finalization can lag a bit
    behind the HTTP response, e.g. for the destructor-driven cancelled path).
    """
    start = time.time()
    while time.time() - start < timeout:
        files = log_files(directory)
        if files:
            return files[-1]
        time.sleep(0.1)
    raise TimeoutError(f"no *.log file appeared in {directory} within {timeout}s")


def wait_for_marker(path: str, marker: str, timeout: float = 10.0):
    start = time.time()
    content = ""
    while time.time() - start < timeout:
        with open(path, "r") as f:
            content = f.read()
        if marker in content:
            return content
        time.sleep(0.1)
    raise TimeoutError(f"'{marker}' never appeared in {path} (last content:\n{content})")


def parse_footer_rows(footer_text: str) -> dict[str, str]:
    """Parse the aligned "Label : value" table written by write_footer_block()
    into a {label: value} dict. Labels are stripped of the alignment padding;
    matching is done by exact label text, so callers don't need to know (or
    re-derive) the column width, which varies per call depending on which
    optional rows are present.
    """
    rows = {}
    for line in footer_text.splitlines():
        if " : " in line:
            label, _, value = line.partition(" : ")
            rows[label.strip()] = value.strip()
    return rows


def test_request_logging_disabled_by_default(tmp_path):
    # request_logging_dir is NOT set -- no --request-logging-dir flag is
    # passed to the server, so no directory / files should ever appear.
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
    })
    assert res.status_code == 200
    would_be_dir = str(tmp_path / "requests")
    assert not os.path.isdir(would_be_dir)


def test_request_logging_non_streaming_completion(tmp_path):
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
    }, headers={"X-Request-Id": "test-non-streaming-001"})
    assert res.status_code == 200

    path = wait_for_log_file(req_dir)
    assert "test-non-streaming-001" in os.path.basename(path)

    content = wait_for_marker(path, "=== PERFORMANCE ===")

    assert "=== CACHYLLAMA REQUEST LOG ===" in content
    assert "request_id: test-non-streaming-001" in content
    assert "method_path: POST /completion" in content
    assert "=== HEADERS ===" in content
    assert "x-request-id: test-non-streaming-001" in content.lower()
    assert "=== SAMPLING PARAMS ===" in content
    assert "=== MODEL SETTINGS ===" in content
    assert "=== REQUEST ===" in content
    assert "=== PROMPT ===" in content
    assert "I believe the meaning of life is" in content
    # the non-streaming HTTP path must still produce a "(streaming)" response
    # section -- this is the proof that it flows through the same
    # server_response_reader::next() hook as the streaming path, not a
    # separate atomic code path (see spec Part 3 "The good news").
    assert "=== RESPONSE (streaming) ===" in content

    # performance footer: aligned "Label : value-with-unit" table (see
    # REQUEST_LOGGING_SPEC.md Part 1 "File format"), with plausible,
    # non-zero values
    footer = content.split("=== PERFORMANCE ===", 1)[1]
    rows = parse_footer_rows(footer)
    assert "Prompt tokens" in rows
    assert rows["Completion tokens"] == "8"
    assert "tok/s" in rows.get("Prefill speed (PP)", "") or "tok/s" in rows.get("Generation speed (TG)", "")
    assert rows["Total duration"].endswith(" s")


def test_request_logging_streaming_completion(tmp_path):
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.start()

    streamed_text = ""
    for chunk in server.make_stream_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
        "stream": True,
    }, headers={"X-Request-Id": "test-streaming-002"}):
        streamed_text += chunk.get("content", "")

    assert len(streamed_text) > 0

    path = wait_for_log_file(req_dir)
    content = wait_for_marker(path, "=== PERFORMANCE ===")

    assert "request_id: test-streaming-002" in content
    assert "=== RESPONSE (streaming) ===" in content
    # what landed in the log should match what the client actually received
    response_section = content.split("=== RESPONSE (streaming) ===\n", 1)[1]
    response_section = response_section.split("\n=== PERFORMANCE", 1)[0]
    assert response_section == streamed_text


def test_request_logging_chat_completion_renders_transcript(tmp_path):
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.start()

    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello there"},
        ],
    }, headers={"X-Request-Id": "test-chat-003"})
    assert res.status_code == 200

    path = wait_for_log_file(req_dir)
    content = wait_for_marker(path, "=== PERFORMANCE ===")

    # readable SYSTEM:/USER: transcript, not raw escaped JSON
    prompt_section = content.split("=== PROMPT ===\n", 1)[1].split("\n\n=== RESPONSE", 1)[0]
    assert "SYSTEM:" in prompt_section
    assert "You are a helpful assistant." in prompt_section
    assert "USER:" in prompt_section
    assert "Hello there" in prompt_section


def test_request_logging_forced_thinking_shows_think_opener(tmp_path):
    # Reasoning-model chat templates often append an opening "<think>" to the
    # end of the rendered prompt, forcing the model straight into its
    # reasoning phase -- the model's own generated tokens then never include
    # that opening tag (only the matching close), which would otherwise make
    # the log read like a truncated response (see
    # REQUEST_LOGGING_SPEC.md's "Reasoning-model <think> prefix" section).
    #
    # DeepSeek-R1-Distill's real chat template is a genuine example of this:
    # test_template.py's test_reasoning() asserts its rendered prompt ends in
    # "<think>\n" with reasoning="on"/"auto". Reused here (against tinyllama2,
    # which can't produce coherent reasoning but exercises the exact same
    # write_header()/generation_prompt detection path) so this doesn't need
    # the full reasoning-capable model to verify the logging behavior.
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.jinja = True
    server.reasoning = "on"
    server.chat_template_file = '../../../models/templates/deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja'
    server.start()

    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "Hello there"},
        ],
    }, headers={"X-Request-Id": "test-thinking-005"})
    assert res.status_code == 200

    path = wait_for_log_file(req_dir)
    content = wait_for_marker(path, "=== PERFORMANCE ===")

    response_section = content.split("=== RESPONSE (streaming) ===\n", 1)[1]
    response_section = response_section.split("\n=== PERFORMANCE", 1)[0]
    assert response_section.startswith("<think>\n"), \
        f"expected response section to start with a synthetic '<think>' opener, got:\n{response_section[:200]!r}"
    # exactly one opener -- never duplicated with anything the model itself emits
    assert response_section.count("<think>") == 1


def test_request_logging_no_forced_thinking_no_think_opener(tmp_path):
    # Control for the test above: a plain (non-reasoning) request must NOT
    # get a synthetic "<think>" prefix.
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.start()

    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "Hello there"},
        ],
    }, headers={"X-Request-Id": "test-no-thinking-006"})
    assert res.status_code == 200

    path = wait_for_log_file(req_dir)
    content = wait_for_marker(path, "=== PERFORMANCE ===")

    response_section = content.split("=== RESPONSE (streaming) ===\n", 1)[1]
    response_section = response_section.split("\n=== PERFORMANCE", 1)[0]
    assert not response_section.startswith("<think>\n")


def test_request_logging_disconnect_leaves_partial_footer(tmp_path):
    # This is the regression test for the ordering concern called out in the
    # porting spec ("must verify before shipping"): when a request is
    # cancelled mid-generation (client disconnect here), the resulting log
    # must show real, non-zero partial token counts -- not the request
    # silently vanishing, and not a zeroed-out counter clobbering the real
    # progress that was already made and already written to disk.
    global server
    req_dir = str(tmp_path / "requests")
    server.request_logging_dir = req_dir
    server.n_predict = -1
    server.n_ctx = 1024
    server.server_slots = True
    server.start()

    url = f"http://{server.server_host}:{server.server_port}/completion"
    resp = requests.post(url, json={
        "prompt": "Once upon a time",
        "n_predict": 256,  # long enough that we can reliably interrupt it
        "stream": True,
    }, headers={"X-Request-Id": "test-disconnect-004"}, stream=True, timeout=30)
    assert resp.status_code == 200

    received = 0
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: ") and "[DONE]" not in decoded:
            received += 1
            if received >= 2:
                break
    assert received >= 2, "expected to receive at least 2 chunks before disconnecting"
    resp.close()  # simulate a client disconnect mid-generation

    path = wait_for_log_file(req_dir)
    content = wait_for_marker(path, "=== PERFORMANCE (partial) ===", timeout=15.0)

    assert "=== CANCELLED ===" in content or "=== ERROR ===" in content

    footer = content.split("=== PERFORMANCE (partial) ===", 1)[1]
    rows = parse_footer_rows(footer)
    assert "Completion tokens" in rows, f"no 'Completion tokens' row in partial footer:\n{footer}"
    # the critical assertion: real partial progress, not zeroed out by the
    # terminal cancellation event
    assert int(rows["Completion tokens"]) > 0

    # the response section should also contain real generated text, not be
    # empty (independent corroboration of the same ordering property)
    response_section = content.split("=== RESPONSE (streaming) ===\n", 1)[1]
    response_section = response_section.split("\n=== CANCELLED", 1)[0].split("\n=== ERROR", 1)[0]
    assert len(response_section.strip()) > 0

    # the slot should be freed shortly after the disconnect is detected
    time.sleep(1)
    res = server.make_request("GET", "/slots")
    assert all(not slot["is_processing"] for slot in res.body)
