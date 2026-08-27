"""Tests for the end-of-training artifact emission added to GenericTrainer.

A run whose step count does not land on a save/sample interval boundary used
to end with no checkpoint and no sample at its final state -- the weights the
user actually wants are the ones nothing captured. `save_on_train_end` /
`sample_on_train_end` fix that, and `saved_train_progresses` /
`sampled_train_progresses` make the fix idempotent: if the final step happens
to *also* land on a normal save/sample interval, the run must not emit the
same artifact twice.

The idempotency guard is the entire reviewable claim of this change, so these
tests exercise the real bookkeeping-writing code in `__save` and
`__sample_during_training` -- not just the five-line
`__emit_final_workspace_artifacts` helper in isolation. A test that only
drives that helper with hand-set membership in `saved_train_progresses` /
`sampled_train_progresses` would still pass with the writes into those sets
deleted; see the mutation checks in this project's report.

`GenericTrainer.__init__` does heavy work (tensorboard, torch.compile setup),
so no test constructs a real trainer. `_make_trainer` below builds the
smallest stand-in that the methods under test actually read: a fake config,
a fake model (with a real `TrainProgress`), and fake `model_saver` /
`model_sampler` / `model_setup` / `callbacks` collaborators that record calls
instead of touching disk, a GPU, or the network. Name-mangled privates are
reached directly (`trainer._GenericTrainer__save(...)`), which is expected
and normal for testing this kind of private wiring.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.enum.EMAMode import EMAMode
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress


class _FakeSampleConfig:
    """Stands in for `modules.util.config.SampleConfig.SampleConfig` --
    only the attributes `__sample_loop` / provenance-building actually read."""

    def __init__(self, enabled: bool = True, prompt: str = "a cat"):
        self.enabled = enabled
        self.prompt = prompt
        self.random_seed = False
        self.seed = 42

    def from_train_config(self, train_config):
        pass

    def to_dict(self):
        return {"prompt": self.prompt, "enabled": self.enabled}


class _FakeOptimizerConfig:
    def __init__(self, is_schedule_free: bool = False):
        self.is_schedule_free = is_schedule_free


class _FakeOptimizerWrapper:
    def __init__(self, is_schedule_free: bool = False):
        self.optimizer = _FakeOptimizerConfig(is_schedule_free)


class _FakeOutputModelFormat:
    def file_extension(self) -> str:
        return ".safetensors"

    def is_single_file(self) -> bool:
        return True


class _FakeOutputDtype:
    def torch_dtype(self):
        return torch.float32


class _FakeConfig:
    """Stands in for `TrainConfig` -- only the fields `__save` /
    `__sample_during_training` / `__sample_loop` actually read."""

    def __init__(
        self,
        *,
        save_on_train_end: bool = True,
        sample_on_train_end: bool = True,
        samples: list | None = None,
        ema: EMAMode = EMAMode.OFF,
        non_ema_sampling: bool = True,
        samples_to_tensorboard: bool = False,
    ):
        self.save_on_train_end = save_on_train_end
        self.sample_on_train_end = sample_on_train_end
        self.samples = samples if samples is not None else []
        self.ema = ema
        self.non_ema_sampling = non_ema_sampling
        self.samples_to_tensorboard = samples_to_tensorboard
        self.workspace_dir = "fake-workspace"
        self.save_filename_prefix = ""
        self.output_model_format = _FakeOutputModelFormat()
        self.output_dtype = _FakeOutputDtype()
        self.optimizer = _FakeOptimizerWrapper(False)
        self.sample_image_format = None
        self.sample_video_format = None
        self.sample_audio_format = None
        self.sample_definition_file_name = "unused-samples.json"
        self.model_type = None


class _FakeOptimizer:
    def eval(self):
        pass

    def train(self):
        pass


class _FakeModel:
    def __init__(self, train_progress: TrainProgress | None = None, ema=None):
        self.train_progress = train_progress if train_progress is not None else TrainProgress()
        self.ema = ema
        self.optimizer = _FakeOptimizer()

    def eval(self):
        pass


class _FakeModelSaver:
    def __init__(self):
        self.save_calls = []

    def save(self, **kwargs):
        self.save_calls.append(kwargs)


class _FakeModelSampler:
    def __init__(self):
        self.sample_calls = 0
        self.provenance = None

    def set_provenance(self, provenance):
        self.provenance = provenance

    def sample(self, **kwargs):
        self.sample_calls += 1


class _FakeModelSetup:
    def setup_train_device(self, model, config):
        pass


class _FakeCallbacks:
    def on_update_status(self, status):
        pass

    def on_sample_default(self, output):
        pass

    def on_sample_custom(self, output):
        pass

    def on_update_sample_default_progress(self, *args, **kwargs):
        pass

    def on_update_sample_custom_progress(self, *args, **kwargs):
        pass


def _make_trainer(config: _FakeConfig, model: _FakeModel) -> GenericTrainer:
    # object.__new__ bypasses __init__ -- real __init__ starts tensorboard
    # and torch.compile, neither of which any of these tests need.
    trainer = object.__new__(GenericTrainer)
    trainer.config = config
    trainer.model = model
    trainer.model_setup = _FakeModelSetup()
    trainer.model_saver = _FakeModelSaver()
    trainer.model_sampler = _FakeModelSampler()
    trainer.callbacks = _FakeCallbacks()
    trainer.train_device = torch.device("cpu")
    trainer.parameters = []
    trainer.one_step_trained = True
    trainer.train_exited_cleanly = True
    trainer.sampled_train_progresses = set()
    trainer.saved_train_progresses = set()
    trainer.last_save_filename = None
    return trainer


# ---------------------------------------------------------------------------
# 1. Off-boundary final step: exactly one save and one sample.
# ---------------------------------------------------------------------------

def test_off_boundary_final_step_emits_exactly_one_save_and_one_sample():
    train_progress = TrainProgress(global_step=7, epoch=0, epoch_step=7)
    config = _FakeConfig(samples=[_FakeSampleConfig(enabled=True)])
    trainer = _make_trainer(config, _FakeModel(train_progress=train_progress))

    assert trainer._GenericTrainer__should_emit_final_workspace_artifacts() is True
    trainer._GenericTrainer__emit_final_workspace_artifacts(train_progress)

    assert len(trainer.model_saver.save_calls) == 1
    assert trainer.model_sampler.sample_calls == 1
    key = train_progress.filename_string()
    assert trainer.saved_train_progresses == {key}
    assert trainer.sampled_train_progresses == {key}


# ---------------------------------------------------------------------------
# 2. On-boundary final step: the real bookkeeping writes from a normal
#    periodic save/sample suppress the final emission -- zero additional
#    saves or samples. This is the whole point of the slice, and it drives
#    the REAL __save / __sample_during_training bookkeeping, not a
#    hand-populated set.
# ---------------------------------------------------------------------------

def test_on_boundary_final_step_emits_nothing_additional():
    train_progress = TrainProgress(global_step=10, epoch=1, epoch_step=0)
    config = _FakeConfig(samples=[_FakeSampleConfig(enabled=True)])
    trainer = _make_trainer(config, _FakeModel(train_progress=train_progress))

    # A normal periodic save + sample lands on the exact same step as the
    # final one -- via the real bookkeeping-writing methods, not the sets
    # populated by hand.
    trainer._GenericTrainer__save(train_progress)
    trainer._GenericTrainer__sample_during_training(train_progress, trainer.train_device)

    key = train_progress.filename_string()
    assert trainer.saved_train_progresses == {key}
    assert trainer.sampled_train_progresses == {key}
    save_calls_before = len(trainer.model_saver.save_calls)
    sample_calls_before = trainer.model_sampler.sample_calls

    assert trainer._GenericTrainer__should_emit_final_workspace_artifacts() is True
    trainer._GenericTrainer__emit_final_workspace_artifacts(train_progress)

    assert len(trainer.model_saver.save_calls) == save_calls_before
    assert trainer.model_sampler.sample_calls == sample_calls_before


# ---------------------------------------------------------------------------
# 3. Both flags off: nothing emitted.
# ---------------------------------------------------------------------------

def test_both_flags_disabled_emits_nothing():
    train_progress = TrainProgress(global_step=7, epoch=0, epoch_step=7)
    config = _FakeConfig(
        save_on_train_end=False,
        sample_on_train_end=False,
        samples=[_FakeSampleConfig(enabled=True)],
    )
    trainer = _make_trainer(config, _FakeModel(train_progress=train_progress))

    trainer._GenericTrainer__emit_final_workspace_artifacts(train_progress)

    assert trainer.model_saver.save_calls == []
    assert trainer.model_sampler.sample_calls == 0
    assert trainer.saved_train_progresses == set()
    assert trainer.sampled_train_progresses == set()


# ---------------------------------------------------------------------------
# 4. __should_emit_final_workspace_artifacts requires BOTH one_step_trained
#    and a clean exit.
# ---------------------------------------------------------------------------

def test_should_emit_is_false_when_nothing_was_trained():
    trainer = _make_trainer(_FakeConfig(), _FakeModel())
    trainer.one_step_trained = False
    trainer.train_exited_cleanly = True
    assert trainer._GenericTrainer__should_emit_final_workspace_artifacts() is False


def test_should_emit_is_false_when_the_run_did_not_exit_cleanly():
    trainer = _make_trainer(_FakeConfig(), _FakeModel())
    trainer.one_step_trained = True
    trainer.train_exited_cleanly = False
    assert trainer._GenericTrainer__should_emit_final_workspace_artifacts() is False


def test_should_emit_is_true_when_trained_and_clean():
    trainer = _make_trainer(_FakeConfig(), _FakeModel())
    trainer.one_step_trained = True
    trainer.train_exited_cleanly = True
    assert trainer._GenericTrainer__should_emit_final_workspace_artifacts() is True


# ---------------------------------------------------------------------------
# 5. __has_enabled_samples, and the real sample bookkeeping is not written
#    when every sample config is disabled.
# ---------------------------------------------------------------------------

def test_has_enabled_samples_is_false_when_every_config_is_disabled():
    trainer = _make_trainer(_FakeConfig(), _FakeModel())
    assert trainer._GenericTrainer__has_enabled_samples([]) is False
    assert trainer._GenericTrainer__has_enabled_samples([_FakeSampleConfig(enabled=False)]) is False
    assert trainer._GenericTrainer__has_enabled_samples(
        [_FakeSampleConfig(enabled=False), _FakeSampleConfig(enabled=True)]
    ) is True


def test_all_disabled_samples_leave_bookkeeping_unwritten():
    train_progress = TrainProgress(global_step=3, epoch=0, epoch_step=3)
    config = _FakeConfig(samples=[_FakeSampleConfig(enabled=False)])
    trainer = _make_trainer(config, _FakeModel(train_progress=train_progress))

    trainer._GenericTrainer__sample_during_training(train_progress, trainer.train_device)

    assert trainer.model_sampler.sample_calls == 0
    assert trainer.sampled_train_progresses == set()


# ---------------------------------------------------------------------------
# 6. The shipped defaults are True, and are pinned so a later decision to
#    flip them is visible as a one-line change against a failing test.
# ---------------------------------------------------------------------------

def test_shipped_defaults_split_sample_from_save():
    """The two knobs ship differently, and the asymmetry is the whole argument.

    `end()` already writes the final weights to `output_model_destination` on
    every run that trained a step, so `save_on_train_end` does not rescue a lost
    model -- it adds a *second* copy in the workspace. That is a few MB for a
    LoRA and gigabytes for a full finetune, paid by every run, to duplicate
    bytes the user already has. It ships OFF.

    Sampling is the half that was genuinely missing: nothing captures the final
    state's images when the step count is not a multiple of the interval, and
    the cost is zero for a user with no enabled sample configs, since
    __sample_loop iterates only over those. It ships ON.

    Pinned so revisiting either is a one-line change against a failing
    assertion, not a silent drift.
    """
    config = TrainConfig.default_values()
    assert config.sample_on_train_end is True
    assert config.save_on_train_end is False


# ---------------------------------------------------------------------------
# 7. A custom (user-triggered) sample must not suppress the final one --
#    it must not populate sampled_train_progresses even though it samples.
# ---------------------------------------------------------------------------

def test_custom_sample_does_not_populate_sampled_progresses():
    train_progress = TrainProgress(global_step=5, epoch=0, epoch_step=5)
    config = _FakeConfig(samples=[_FakeSampleConfig(enabled=True)])
    trainer = _make_trainer(config, _FakeModel(train_progress=train_progress))

    # A non-empty sample_params_list argument is exactly how a user-triggered
    # ("sample now") custom sample reaches __sample_during_training.
    trainer._GenericTrainer__sample_during_training(
        train_progress, trainer.train_device, [_FakeSampleConfig(enabled=True)]
    )

    assert trainer.model_sampler.sample_calls == 1
    assert trainer.sampled_train_progresses == set()

    # So a real final emission right after must still take its own sample.
    assert trainer._GenericTrainer__should_emit_final_workspace_artifacts() is True
    trainer._GenericTrainer__emit_final_workspace_artifacts(train_progress)
    assert trainer.model_sampler.sample_calls == 2
    assert trainer.sampled_train_progresses == {train_progress.filename_string()}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
