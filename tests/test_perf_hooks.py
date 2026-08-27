"""OneTrainer's OT_PERF call sites: guarded, correct when on, free when off.

The reviewer's first question about instrumentation left permanently in a hot
loop is what it costs the people who never switch it on. This file answers it
three ways, in increasing strength:

1. **Statically** -- every ``perf.<method>(...)`` in ``modules/`` is lexically
   inside an ``if perf.enabled:``. Python evaluates arguments eagerly, so an
   early return inside the probe is too late: the call and its arguments are
   already paid for. This audit is what stops the next hook from being added
   unguarded.
2. **Behaviourally** -- with the probe off, the real production functions are
   driven and the probe records *zero* calls, not "calls that did nothing".
3. **Numerically** -- the disabled path is timed against the unguarded shape the
   guard exists to avoid, and the numbers are printed.

The on-path tests drive the real entry points and let production compute what it
computes: the token count comes out of a real latent tensor via the trainer's own
helper, and the offload counters are read back out of a real ``PerfProbe``'s JSONL
after the real (name-mangled, private) scheduling method has run.

Run with ``python -m pytest tests/test_perf_hooks.py``.
"""

import ast
import json
import os
import pathlib
import sys
import timeit
import types
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import modules.trainer.GenericTrainer as generic_trainer  # noqa: E402
import modules.util.LayerOffloadConductor as offload  # noqa: E402
from modules.trainer.GenericTrainer import _latent_tokens, _profile_enabled  # noqa: E402

from mgds.perf_probe import PerfProbe  # noqa: E402

import torch  # noqa: E402

_MODULES_ROOT = pathlib.Path(__file__).resolve().parent.parent / "modules"


# --------------------------------------------------------------------------- helpers

def _probe(tmp_path, **env):
    """A real PerfProbe built from a real environment, writing where we can read it."""
    previous = {k: os.environ.get(k) for k in ("OT_PERF", "OT_PERF_OUT", "OT_PROFILE_STEP", "OT_PROFILE_MIN_TOKENS")}
    os.environ["OT_PERF"] = "1"
    os.environ["OT_PERF_OUT"] = str(tmp_path / "ot_perf.jsonl")
    for key in ("OT_PROFILE_STEP", "OT_PROFILE_MIN_TOKENS"):
        os.environ.pop(key, None)
    os.environ.update(env)
    try:
        return PerfProbe()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _SpyProbe:
    """A probe that is off, and remembers every method call it should never receive."""

    def __init__(self):
        self.enabled = False
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


def _batch(h_lat, w_lat):
    # 5D (B, C, T, H, W), the shape the video-capable setups actually hand over.
    return {"latent_image": torch.zeros(1, 16, 1, h_lat, w_lat)}


# --------------------------------------------------------------------------- 1. static audit

def _perf_calls(tree):
    """(lineno, method, guarded) for every ``perf.<method>(...)`` in a parsed module."""
    found = []

    def visit(node, guarded):
        if isinstance(node, ast.If) and _tests_perf_enabled(node.test):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, guarded)
            return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "perf"):
            found.append((node.lineno, node.func.attr, guarded))
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(tree, False)
    return found


def _tests_perf_enabled(test):
    return (isinstance(test, ast.Attribute) and test.attr == "enabled"
            and isinstance(test.value, ast.Name) and test.value.id == "perf")


def _audit_modules():
    sites = []
    for path in sorted(_MODULES_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "perf." not in source:
            continue
        for lineno, method, guarded in _perf_calls(ast.parse(source)):
            sites.append((f"{path.relative_to(_MODULES_ROOT.parent)}:{lineno}", method, guarded))
    return sites


class TestEveryCallSiteIsGuarded:
    def test_no_perf_call_in_modules_runs_outside_an_if_perf_enabled(self):
        unguarded = [f"{where} -> perf.{method}(...)" for where, method, guarded in _audit_modules() if not guarded]
        assert not unguarded, (
            "these perf calls are paid for on every run with OT_PERF unset, because Python "
            "evaluates their arguments before the probe can return:\n  " + "\n  ".join(unguarded)
        )

    def test_the_audit_actually_found_the_call_sites_it_claims_to_police(self):
        # Without this, deleting every hook (or the import) makes the audit above vacuous.
        sites = _audit_modules()
        assert len(sites) >= 8, f"expected the trainer and offload hooks; found {sites}"
        assert {method for _, method, _ in sites} >= {"step_begin", "step_end", "tic", "toc", "note", "incr"}


def _region_labels(path):
    """The literal region names passed to perf.tic()/perf.toc() in one module."""
    tics, tocs = Counter(), Counter()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "perf"
                and node.func.attr in ("tic", "toc") and node.args
                and isinstance(node.args[0], ast.Constant)):
            (tics if node.func.attr == "tic" else tocs)[node.args[0].value] += 1
    return tics, tocs


class TestTheTrainerStillTimesTheRegionsItClaimsTo:
    """The regions live inside ``train()``, which no unit test can drive.

    What is checkable without a model is that they are all still there and still
    balanced -- and balance matters more than it looks: ``PerfProbe.step_end``
    silently skips a ``tic`` with no matching ``toc``, so an unbalanced pair does
    not error, it just quietly drops the region from every row.
    """

    _PATH = _MODULES_ROOT / "trainer" / "GenericTrainer.py"

    def test_every_region_the_trainer_opens_it_also_closes(self):
        tics, tocs = _region_labels(self._PATH)
        assert tics == tocs, f"unbalanced perf regions: opened {tics}, closed {tocs}"

    def test_the_step_is_still_broken_into_the_regions_that_scale_differently(self):
        tics, _ = _region_labels(self._PATH)
        assert set(tics) == {"predict", "prior_predict", "backward", "optimizer"}


def _region_spans(path):
    """Every (label, first_line, last_line) region, paired in source order.

    ``_region_labels`` counts the pairs; this one locates them. Balance says the
    hooks exist, which is not the same as saying they are on the right side of the
    call they name -- a ``toc`` above its ``tic`` still balances, and still reports
    a duration measured over the wrong statements.
    """
    marks = [
        (node.lineno, node.args[0].value, node.func.attr)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "perf"
            and node.func.attr in ("tic", "toc") and node.args
            and isinstance(node.args[0], ast.Constant))
    ]
    open_at, spans = {}, []
    for lineno, label, method in sorted(marks):
        if method == "tic":
            assert label not in open_at, f"perf.tic({label!r}) reopened at line {lineno} before its toc"
            open_at[label] = lineno
        else:
            assert label in open_at, f"perf.toc({label!r}) at line {lineno} closes a region never opened"
            spans.append((label, open_at.pop(label), lineno))
    assert not open_at, f"regions opened and never closed: {open_at}"
    return spans


def _perf_note_arguments(path):
    """``{note_name: source of the value expression}`` for every perf.note() call."""
    notes = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "perf"
                and node.func.attr == "note" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)):
            notes[node.args[0].value] = ast.unparse(node.args[1])
    return notes


class TestTheRegionsBracketTheWorkTheyName:
    """Balanced is not the same as correct.

    ``train()`` cannot be driven without a model, an optimizer and a dataloader, so
    the statement *ordering* inside it is unreachable behaviourally. It is still
    reachable structurally: a region has to open before it closes, and the source
    between the two has to contain the call the label claims to be timing. This is
    what catches a ``tic``/``toc`` pair written on the wrong sides of its call --
    which balances, runs, and reports a number measured over the wrong work.
    """

    _PATH = _MODULES_ROOT / "trainer" / "GenericTrainer.py"

    #: The call each region must enclose, and (for the two that both wrap
    #: ``model_setup.predict``) the one it must *not*, so a swapped label is caught.
    _MUST_ENCLOSE = {
        "predict": ("self.model_setup.predict(", "self.model_setup.prior_model("),
        "prior_predict": ("self.model_setup.prior_model(", None),
        "backward": (".backward()", None),
        "optimizer": ("self.model.optimizer.step()", None),
    }

    def test_a_region_opens_before_it_closes(self):
        for label, first, last in _region_spans(self._PATH):
            assert first < last, (
                f"perf.toc({label!r}) is at line {last}, above its tic at {first} -- the region "
                "is inverted and times everything except the call it names"
            )

    def test_each_region_encloses_the_call_it_is_named_after(self):
        lines = self._PATH.read_text(encoding="utf-8").splitlines()
        seen = set()
        for label, first, last in _region_spans(self._PATH):
            required, forbidden = self._MUST_ENCLOSE[label]
            body = "\n".join(lines[first:last - 1])  # strictly between the two hooks
            assert required in body, (
                f"the {label!r} region (lines {first}-{last}) does not contain {required!r}; "
                "it is bracketing something other than the work it reports"
            )
            if forbidden is not None:
                assert forbidden not in body, (
                    f"the {label!r} region (lines {first}-{last}) contains {forbidden!r} -- "
                    "the two predict regions look swapped"
                )
            seen.add(label)
        assert seen == set(self._MUST_ENCLOSE), f"unchecked regions: {set(self._MUST_ENCLOSE) - seen}"


class TestTheStepNotesReportWhatTheyAreNamed:
    """A note wired to the wrong field is a wrong number, not a missing one.

    Every row is bucketed by these, so ``batch_size`` reading
    ``gradient_accumulation_steps`` would not fail anything -- it would silently
    make every comparison between rows meaningless. Pinned as source, because the
    call site is inside ``train()``.
    """

    _PATH = _MODULES_ROOT / "trainer" / "GenericTrainer.py"

    def test_each_note_reads_the_field_it_claims_to(self):
        assert _perf_note_arguments(self._PATH) == {
            "latent_tokens": "_latent_tokens(batch)",
            "batch_size": "self.config.batch_size",
            "rank": "multi.rank()",
        }


# --------------------------------------------------------------------------- 2. the off path

class TestTheOffPathCallsNothing:
    def test_the_trainer_profile_trigger_never_touches_the_probe(self, monkeypatch):
        spy = _SpyProbe()
        monkeypatch.setattr(generic_trainer, "perf", spy)
        monkeypatch.setattr(generic_trainer, "_PROFILE_STEPS", ())
        assert _profile_enabled(7, _batch(64, 64)) is False
        assert spy.calls == []

    def test_the_offload_hook_never_touches_the_probe(self, monkeypatch):
        spy = _SpyProbe()
        monkeypatch.setattr(offload, "perf", spy)
        _run_schedule_layer_to(to_train_device=True)
        assert spy.calls == []


# --------------------------------------------------------------------------- 3. the on path

class TestTheProfileTrigger:
    def test_the_upstream_fixed_step_list_still_fires_with_the_probe_off(self, monkeypatch):
        # OT_DEBUG_PROFILES is upstream's mechanism and OT_PERF must not have replaced it.
        monkeypatch.setattr(generic_trainer, "perf", _SpyProbe())
        monkeypatch.setattr(generic_trainer, "_PROFILE_STEPS", (10, 11))
        assert _profile_enabled(10, _batch(64, 64)) is True
        assert _profile_enabled(12, _batch(64, 64)) is False

    def test_an_exact_step_index_fires_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr(generic_trainer, "perf", _probe(tmp_path, OT_PROFILE_STEP="5"))
        monkeypatch.setattr(generic_trainer, "_PROFILE_STEPS", ())
        assert _profile_enabled(4, _batch(64, 64)) is False
        assert _profile_enabled(5, _batch(64, 64)) is True
        assert _profile_enabled(5, _batch(64, 64)) is False, "the trigger latches; one trace, not one per step"

    def test_the_token_threshold_fires_on_the_first_big_enough_step(self, tmp_path, monkeypatch):
        # 96*96 = 9216 tokens clears the threshold; 64*64 = 4096 does not. The test never
        # passes a token count -- the trainer derives it from the latent it was handed.
        monkeypatch.setattr(generic_trainer, "perf", _probe(tmp_path, OT_PROFILE_MIN_TOKENS="8192"))
        monkeypatch.setattr(generic_trainer, "_PROFILE_STEPS", ())
        assert _profile_enabled(0, _batch(64, 64)) is False, "a small bucket is not the step we are waiting for"
        assert _profile_enabled(1, _batch(96, 96)) is True
        assert _profile_enabled(2, _batch(96, 96)) is False, "latched"

    def test_the_step_index_acts_as_a_warmup_floor_under_the_token_threshold(self, tmp_path, monkeypatch):
        # Otherwise the first trace is of step 0, which is all torch.compile.
        monkeypatch.setattr(
            generic_trainer, "perf", _probe(tmp_path, OT_PROFILE_STEP="10", OT_PROFILE_MIN_TOKENS="8192"))
        monkeypatch.setattr(generic_trainer, "_PROFILE_STEPS", ())
        assert _profile_enabled(9, _batch(96, 96)) is False, "big enough, but still inside warmup"
        assert _profile_enabled(10, _batch(96, 96)) is True

    def test_a_batch_with_no_latent_does_not_crash_the_step(self, tmp_path, monkeypatch):
        monkeypatch.setattr(generic_trainer, "perf", _probe(tmp_path, OT_PROFILE_MIN_TOKENS="1"))
        monkeypatch.setattr(generic_trainer, "_PROFILE_STEPS", ())
        assert _profile_enabled(0, {}) is False


class TestLatentTokens:
    def test_counts_the_two_trailing_spatial_dims_of_a_5d_latent(self):
        assert _latent_tokens(_batch(96, 64)) == 96 * 64

    def test_counts_the_two_trailing_spatial_dims_of_a_4d_latent(self):
        assert _latent_tokens({"latent_image": torch.zeros(2, 16, 32, 48)}) == 32 * 48

    @pytest.mark.parametrize("batch", [{}, {"latent_image": None}, "not a dict"])
    def test_an_unrecognisable_batch_is_none_rather_than_an_exception(self, batch):
        assert _latent_tokens(batch) is None


# --------------------------------------------------------------------------- the offload counters

def _run_schedule_layer_to(*, to_train_device: bool):
    """Drive the real (private, name-mangled) LayerOffloadConductor scheduling method.

    Only the state the method reads before the counters is supplied; the allocator
    lookup immediately after them raises, which stops the call without needing a
    model, a CUDA context or a transfer stream. That keeps this a test of the real
    call site rather than of a copy of it.
    """
    class _Stop(Exception):
        pass

    class _Allocator:
        def get_allocator(self, layer_index, is_forward):
            raise _Stop

    train_device, temp_device = torch.device("cuda:0"), torch.device("cpu")
    target = train_device if to_train_device else temp_device
    conductor = types.SimpleNamespace(_LayerOffloadConductor__layer_device_map=[temp_device if to_train_device else train_device], _LayerOffloadConductor__train_device=train_device, _LayerOffloadConductor__temp_device=temp_device, _LayerOffloadConductor__train_device_layer_allocator=_Allocator(), _LayerOffloadConductor__temp_device_layer_allocator=_Allocator())
    schedule = offload.LayerOffloadConductor._LayerOffloadConductor__schedule_layer_to
    with pytest.raises(_Stop):
        schedule(conductor, 0, target, True)


def _skip_schedule_layer_to():
    """The same method, on a layer already where it is being sent -- the skipped case."""
    device = torch.device("cpu")
    conductor = types.SimpleNamespace(_LayerOffloadConductor__layer_device_map=[device])
    offload.LayerOffloadConductor._LayerOffloadConductor__schedule_layer_to(conductor, 0, device, True)


class TestOffloadCounters:
    def _row(self, probe, tmp_path):
        probe.step_end()
        lines = (tmp_path / "ot_perf.jsonl").read_text().splitlines()
        assert len(lines) == 1, lines
        return json.loads(lines[0])

    def test_an_onload_counts_as_both_a_transfer_and_an_onload(self, tmp_path, monkeypatch):
        probe = _probe(tmp_path)
        monkeypatch.setattr(offload, "perf", probe)
        probe.step_begin(0)
        _run_schedule_layer_to(to_train_device=True)
        row = self._row(probe, tmp_path)
        assert row["offload_xfers"] == 1
        assert row["offload_onload"] == 1

    def test_an_offload_counts_as_a_transfer_only(self, tmp_path, monkeypatch):
        probe = _probe(tmp_path)
        monkeypatch.setattr(offload, "perf", probe)
        probe.step_begin(0)
        _run_schedule_layer_to(to_train_device=False)
        row = self._row(probe, tmp_path)
        assert row["offload_xfers"] == 1
        assert "offload_onload" not in row, "a CPU-bound layer is not an onload"

    def test_a_layer_already_on_the_target_device_is_not_counted(self, tmp_path, monkeypatch):
        # The counter measures traffic, so the early-return case must stay uncounted --
        # otherwise every step reports a constant transfer count and the metric is noise.
        probe = _probe(tmp_path)
        monkeypatch.setattr(offload, "perf", probe)
        probe.step_begin(0)
        _skip_schedule_layer_to()
        row = self._row(probe, tmp_path)
        assert "offload_xfers" not in row


# --------------------------------------------------------------------------- 4. the number

class TestTheDisabledPathIsFree:
    """The guard's whole justification, measured rather than asserted in prose."""

    @staticmethod
    def _ns(statement, probe):
        latent = torch.zeros(1, 16, 1, 96, 96)
        loops = 200_000
        best = min(timeit.repeat(statement, repeat=5, number=loops,
                                 globals={"perf": probe, "latent": latent}))
        return best / loops * 1e9

    def test_guarding_at_the_call_site_beats_returning_early_inside_the_probe(self, capsys):
        probe = PerfProbe()  # OT_PERF is unset in the test environment
        assert probe.enabled is False

        # The offload site: hottest, once per moved layer per step, constant argument.
        hot_unguarded = self._ns('perf.incr("offload_xfers")', probe)
        hot_guarded = self._ns('if perf.enabled: perf.incr("offload_xfers")', probe)

        # The note site: the shape the mgds half measured, where the *arguments* are the
        # cost. Python evaluates them before the probe can decline them.
        note_unguarded = self._ns(
            'perf.note("latent_tokens", int(latent.shape[-2]) * int(latent.shape[-1]))', probe)
        note_guarded = self._ns(
            'if perf.enabled: perf.note("latent_tokens", int(latent.shape[-2]) * int(latent.shape[-1]))', probe)

        with capsys.disabled():
            print(f"\n  OT_PERF unset, per call site, ns:"
                  f"\n    perf.incr(const)          unguarded {hot_unguarded:6.1f}  guarded {hot_guarded:6.1f}"
                  f"\n    perf.note(computed args)  unguarded {note_unguarded:6.1f}  guarded {note_guarded:6.1f}")

        assert hot_guarded < hot_unguarded * 0.7
        assert note_guarded < note_unguarded / 2
        # An attribute load and a branch. Generous enough not to flake on a loaded box,
        # tight enough that a guard silently becoming a call would break it.
        assert hot_guarded < 60.0
