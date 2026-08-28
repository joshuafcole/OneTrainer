"""stdout carries the payload and nothing else, even when C writes to fd 1.

Driven through real subprocesses rather than by monkeypatching ``sys.stdout``:
the whole point of the module is that it works at the *descriptor* level, and a
test that only exercised the Python-level object would pass just as happily
against the bug it exists to catch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_UTIL = Path(__file__).resolve().parent.parent / "scripts" / "util"


def _run(body: str) -> subprocess.CompletedProcess[str]:
    program = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(_UTIL)!r})\n"
        "import result_channel\n" + body
    )
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )


class TestTheBannerGoesToTheLog:
    def test_a_write_straight_to_fd_1_lands_on_stderr(self):
        # `os.write(1, ...)` is what a C library does. `print` and
        # `contextlib.redirect_stdout` do not see it, which is exactly why the
        # nvCOMP banner reached the agent as the first bytes of the payload.
        proc = _run(
            "result_channel.claim()\n"
            "os.write(1, b'nvCOMP: low-level batched C API is deprecated\\n')\n"
            "result_channel.emit_json({'ok': True})\n"
        )
        assert json.loads(proc.stdout) == {"ok": True}
        assert "nvCOMP" in proc.stderr

    def test_a_stray_print_lands_on_stderr_too(self):
        proc = _run(
            "result_channel.claim()\n"
            "print('a debugging print somebody left in')\n"
            "result_channel.emit_json([1, 2, 3])\n"
        )
        assert json.loads(proc.stdout) == [1, 2, 3]
        assert "debugging print" in proc.stderr

    def test_noise_written_before_the_claim_is_not_caught(self):
        # Stated so the ordering requirement is a tested fact rather than a
        # comment: the claim has to precede the noisy import, and a script that
        # imports torch first is not protected by having imported this module.
        proc = _run(
            "os.write(1, b'too early\\n')\n"
            "result_channel.claim()\n"
            "result_channel.emit_json({'ok': True})\n"
        )
        assert proc.stdout.startswith("too early")


class TestThePayloadItself:
    def test_one_json_document_and_a_newline(self):
        proc = _run("result_channel.claim()\nresult_channel.emit_json({'a': 1})\n")
        assert proc.stdout == '{"a": 1}\n'

    def test_a_payload_larger_than_a_pipe_buffer_arrives_whole(self):
        # A pipe takes 64 KiB before it blocks, and `os.write` is allowed to
        # return short. A Gram over a large selection is comfortably past that.
        proc = _run(
            "result_channel.claim()\n"
            "result_channel.emit_json({'layers': list(range(200000))})\n"
        )
        assert json.loads(proc.stdout)["layers"][-1] == 199999

    def test_unclaimed_is_the_old_behaviour_not_a_failure(self):
        proc = _run("result_channel.emit_json({'ok': True})\n")
        assert json.loads(proc.stdout) == {"ok": True}

    def test_claim_is_idempotent(self):
        proc = _run(
            "result_channel.claim()\n"
            "result_channel.claim()\n"
            "os.write(1, b'noise\\n')\n"
            "result_channel.emit_json({'ok': True})\n"
        )
        assert json.loads(proc.stdout) == {"ok": True}
        assert "noise" in proc.stderr


class TestEveryScriptThatPrintsJsonClaimsTheChannel:
    """The rule, enforced over the family rather than remembered per script.

    A seventh ``block_*`` script that emits JSON and forgets the claim would
    reproduce the original defect exactly, and would do it on the box, where the
    banner is — never here, where nvCOMP is not installed.
    """

    def test_no_block_script_writes_its_result_to_sys_stdout(self):
        offenders = [
            p.name
            for p in sorted(_UTIL.glob("block_*.py"))
            if "sys.stdout" in p.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_every_block_script_that_emits_claims_first(self):
        for path in sorted(_UTIL.glob("block_*.py")):
            text = path.read_text(encoding="utf-8")
            if "result_channel.emit_json" not in text:
                continue
            claim = text.index("result_channel.claim()")
            assert claim < text.index("import torch"), (
                f"{path.name}: claims the channel after importing torch, so the "
                "banner it exists to catch is already on fd 1"
            )
