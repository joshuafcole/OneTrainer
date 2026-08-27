"""Tests for scripts/extract_bundled_embeddings.py.

Pins the behavior of `extract()`, including the two bugs fixed after the
script's initial cut:
  - Windows-safe filenames: a bracketed placeholder like "<token>" (the
    default placeholder value) must produce a writeable "token.safetensors",
    not "<token>.safetensors" (`<>` are illegal on Windows).
  - Dotted-placeholder grouping: a placeholder containing a dot (e.g. "v1.0")
    must stay whole; splitting the bundle key from the left mis-parses it as
    "v1" + "0.qwen".

Run with ``python -m pytest tests/test_extract_bundled_embeddings.py``.
"""

import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from extract_bundled_embeddings import extract  # noqa: E402


def _write_lora(path, tensors: dict[str, torch.Tensor]):
    save_file(tensors, str(path))


def _read_keys(path) -> set[str]:
    with safe_open(str(path), framework="pt") as f:
        return set(f.keys())


def test_simple_placeholder_round_trips(tmp_path):
    lora = tmp_path / "lora.safetensors"
    vec = torch.randn(4)
    _write_lora(lora, {"bundle_emb.mychar.qwen": vec})

    written = extract(lora, tmp_path / "out")

    assert [p.name for p in written] == ["mychar.safetensors"]
    with safe_open(str(written[0]), framework="pt") as f:
        assert set(f.keys()) == {"qwen"}
        assert torch.allclose(f.get_tensor("qwen"), vec)


def test_bracketed_placeholder_produces_windows_safe_filename(tmp_path):
    lora = tmp_path / "lora.safetensors"
    _write_lora(lora, {"bundle_emb.<token>.qwen": torch.randn(4)})

    written = extract(lora, tmp_path / "out")

    names = {p.name for p in written}
    assert names == {"token.safetensors"}
    assert not (tmp_path / "out" / "<token>.safetensors").exists()


def test_dotted_placeholder_groups_correctly(tmp_path):
    lora = tmp_path / "lora.safetensors"
    _write_lora(
        lora,
        {
            "bundle_emb.v1.0.qwen": torch.randn(4),
            "bundle_emb.v1.0.qwen_out": torch.randn(4),
        },
    )

    written = extract(lora, tmp_path / "out")

    assert len(written) == 1
    out_path = written[0]
    assert out_path.name == "v1.0.safetensors"
    assert _read_keys(out_path) == {"qwen", "qwen_out"}


def test_multiple_placeholders_are_grouped_separately(tmp_path):
    lora = tmp_path / "lora.safetensors"
    _write_lora(
        lora,
        {
            "bundle_emb.alpha.qwen": torch.randn(4),
            "bundle_emb.beta.clip_l": torch.randn(4),
            "bundle_emb.beta.clip_l_out": torch.randn(4),
        },
    )

    written = extract(lora, tmp_path / "out")

    names = {p.name for p in written}
    assert names == {"alpha.safetensors", "beta.safetensors"}
    by_name = {p.name: p for p in written}
    assert _read_keys(by_name["alpha.safetensors"]) == {"qwen"}
    assert _read_keys(by_name["beta.safetensors"]) == {"clip_l", "clip_l_out"}


def test_non_bundle_keys_are_ignored(tmp_path):
    lora = tmp_path / "lora.safetensors"
    _write_lora(
        lora,
        {
            "transformer.some_lora.weight": torch.randn(4, 4),
            "bundle_emb.alpha.qwen": torch.randn(4),
        },
    )

    written = extract(lora, tmp_path / "out")

    assert [p.name for p in written] == ["alpha.safetensors"]


def test_malformed_bundle_key_with_trailing_dot_is_skipped(tmp_path):
    lora = tmp_path / "lora.safetensors"
    _write_lora(
        lora,
        {
            # Trailing dot -- rpartition finds it but the segment after it is
            # empty, so encoder_key comes back empty and the key must be
            # skipped rather than crash or silently misfile.
            "bundle_emb.alpha.": torch.randn(4),
            "bundle_emb.alpha.qwen": torch.randn(4),
        },
    )

    written = extract(lora, tmp_path / "out")

    assert [p.name for p in written] == ["alpha.safetensors"]
    assert _read_keys(written[0]) == {"qwen"}


def test_bundle_key_with_no_dot_at_all_is_skipped(tmp_path):
    # Regression guard for the half-swapped guard that rpartition invites.
    # rpartition('.') on a dot-less remainder returns ('', '', 'orphan') -- so
    # the placeholder is empty and the encoder key is truthy, exactly inverted
    # from what left-partition() produced. Guarding `encoder_key` alone
    # therefore accepts the malformed key and writes a nameless, hidden
    # ".safetensors". Every *LoRASaver writes placeholder + "." + encoder_key,
    # so a dot-less key is malformed by construction and must be skipped.
    lora = tmp_path / "lora.safetensors"
    _write_lora(lora, {"bundle_emb.orphan": torch.randn(4)})

    written = extract(lora, tmp_path / "out")

    assert written == []
    assert not (tmp_path / "out").exists(), "a malformed-only file must not create an out dir"


def test_bundle_key_with_empty_placeholder_is_skipped(tmp_path):
    # The other spelling of the same defect: an explicit dot with nothing
    # before it. rpartition gives ('', '.', 'qwen') -- encoder_key is fine,
    # the placeholder is not.
    lora = tmp_path / "lora.safetensors"
    _write_lora(lora, {"bundle_emb..qwen": torch.randn(4)})

    written = extract(lora, tmp_path / "out")

    assert written == []


def test_no_bundled_embeddings_returns_empty_list(tmp_path):
    lora = tmp_path / "lora.safetensors"
    _write_lora(lora, {"transformer.some_lora.weight": torch.randn(4, 4)})

    written = extract(lora, tmp_path / "out")

    assert written == []
    assert not (tmp_path / "out").exists()
