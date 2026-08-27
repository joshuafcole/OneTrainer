"""Tests that gradient-aligned (GA) init only ever touches the *trainable*
population of a LoRAModuleWrapper.

Layer-filtered resume (see test_lora_resume_filter.py) splits a wrapper into
three populations: trainable (lora_modules), frozen inherited
(frozen_lora_modules, loaded from a resumed checkpoint but not selected by
the current filter), and dummy (dummy_lora_modules, foreign/stale checkpoint
keys with no real target module). GA init did not originally have to reason
about this split -- it predates layer-filtered resume -- so there is no fork
commit this is ported from; it is new coverage for the current tree's
GenericTrainer.__run_peft_gradient_init and
LoRAModuleWrapper.init_from_gradients/init_from_factors.

Re-initializing a frozen module would silently discard weights the user
explicitly resumed from. A dummy module has no real weights (and no orig
Linear to align a gradient against) at all -- touching one would crash, since
DummyLoRAModule methods other than load_state_dict/state_dict raise
NotImplementedError by construction (see PeftBase.make_dummy).

Run with ``python -m pytest tests/test_peft_ga_init_populations.py``.
"""

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.module.LoRAModule import LoRAModuleWrapper  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.ModelType import PeftType  # noqa: E402

IN, OUT, RANK = 32, 32, 4


class _ThreeBlockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Linear(IN, OUT, bias=False)
        self.block1 = nn.Linear(IN, OUT, bias=False)


class _RaisingDict(dict):
    """A dict standing in for dummy_lora_modules that fails the test loudly if
    GA init ever iterates or looks into it."""

    def __getitem__(self, key):
        raise AssertionError("GA init must never index into dummy_lora_modules")

    def __iter__(self):
        raise AssertionError("GA init must never iterate dummy_lora_modules")

    def values(self):
        raise AssertionError("GA init must never read dummy_lora_modules.values()")

    def items(self):
        raise AssertionError("GA init must never read dummy_lora_modules.items()")


def _config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.peft_type = PeftType.LORA
    config.lora_rank = RANK
    config.lokr_dim = RANK
    config.lora_alpha = 1.0
    config.train_device = "cpu"
    return config


def _wrapper(model: nn.Module, module_filter: list[str]) -> LoRAModuleWrapper:
    return LoRAModuleWrapper(model, "lora", _config(), module_filter)


def _split_wrapper():
    """Build a checkpoint under filter {block0, block1}, add a foreign key, then
    resume under the narrower filter {block0} -- giving trainable=block0,
    frozen=block1, dummy=ghost. Returns (wrapper, block1_down_before)."""
    model = _ThreeBlockNet()
    wrapper_a = _wrapper(model, ["block0", "block1"])
    block1_down_before = wrapper_a.lora_modules["block1"].lora_down.weight.detach().clone()
    state_dict = wrapper_a.state_dict()

    state_dict["lora.ghost.lora_down.weight"] = torch.zeros(RANK, IN)
    state_dict["lora.ghost.lora_up.weight"] = torch.zeros(OUT, RANK)
    state_dict["lora.ghost.alpha"] = torch.tensor(1.0)

    wrapper_b = _wrapper(model, ["block0"])
    wrapper_b.load_state_dict(state_dict)

    assert set(wrapper_b.lora_modules) == {"block0"}
    assert set(wrapper_b.frozen_lora_modules) == {"block1"}
    assert len(wrapper_b.dummy_lora_modules) > 0

    return wrapper_b, block1_down_before


def _fake_grad():
    torch.manual_seed(7)
    return torch.randn(OUT, IN)


def test_init_from_gradients_applies_only_to_trainable():
    wrapper, _block1_before = _split_wrapper()
    grads = {"block0": _fake_grad(), "block1": _fake_grad()}

    applied, skipped, factors = wrapper.init_from_gradients(grads, gain=1.0)

    assert applied == 1 and skipped == 0
    assert set(factors) == {"block0"}


def test_init_from_gradients_leaves_frozen_module_bit_identical():
    wrapper, block1_down_before = _split_wrapper()
    # A gradient keyed to the frozen module's short name -- if the frozen
    # population were ever visited, this would change it.
    grads = {"block0": _fake_grad(), "block1": _fake_grad()}

    wrapper.init_from_gradients(grads, gain=5.0)

    frozen = wrapper.frozen_lora_modules["block1"]
    for param in frozen.parameters():
        assert not param.requires_grad
    assert torch.equal(frozen.lora_down.weight, block1_down_before)


def test_init_from_gradients_never_touches_dummy_population():
    wrapper, _block1_before = _split_wrapper()
    real_dummies = wrapper.dummy_lora_modules
    wrapper.dummy_lora_modules = _RaisingDict(real_dummies)

    grads = {"block0": _fake_grad(), "block1": _fake_grad()}
    applied, skipped, _factors = wrapper.init_from_gradients(grads, gain=1.0)

    assert applied == 1  # got this far without the _RaisingDict tripping


def test_init_from_factors_replay_also_respects_the_split():
    wrapper, block1_down_before = _split_wrapper()
    real_dummies = wrapper.dummy_lora_modules
    wrapper.dummy_lora_modules = _RaisingDict(real_dummies)

    trainable_before = wrapper.lora_modules["block0"].lora_down.weight.detach().clone()
    factors = {
        "block0": (torch.randn(RANK, IN),),
        "block1": (torch.randn(RANK, IN),),
    }

    applied, skipped = wrapper.init_from_factors(factors, gain=1.0)

    assert applied == 1 and skipped == 0
    assert not torch.equal(wrapper.lora_modules["block0"].lora_down.weight, trainable_before)
    assert torch.equal(wrapper.frozen_lora_modules["block1"].lora_down.weight, block1_down_before)


def test_dora_is_skipped_not_reinitialized():
    # DoRA subclasses LoRAModule and inherits its init_from_gradient/
    # init_from_factors unmodified -- if the wrapper's isinstance(DoRAModule)
    # skip were ever dropped, those inherited methods would happily rewrite
    # lora_down and silently break DoRA's identity-at-init property (DoRA's
    # dora_scale is computed once, at construction, off the *original*
    # lora_down/lora_up; changing lora_down afterwards moves the forward
    # output away from the base model, unlike LoRA/LoKr where lora_up staying
    # zero keeps the delta at zero regardless of what lora_down holds).
    model = _ThreeBlockNet()
    config = _config()
    config.lora_decompose = True
    wrapper = LoRAModuleWrapper(model, "lora", config, ["block0", "block1"])
    wrapper.hook_to_module()

    x = torch.randn(3, IN)
    before = {name: module.forward(x).clone() for name, module in wrapper.lora_modules.items()}

    grads = {"block0": _fake_grad(), "block1": _fake_grad()}
    applied, skipped, factors = wrapper.init_from_gradients(grads, gain=5.0)

    assert applied == 0 and skipped == len(wrapper.lora_modules)
    assert factors == {}
    for name, module in wrapper.lora_modules.items():
        assert torch.allclose(module.forward(x), before[name], atol=1e-6)


if __name__ == "__main__":
    test_init_from_gradients_applies_only_to_trainable()
    test_init_from_gradients_leaves_frozen_module_bit_identical()
    test_init_from_gradients_never_touches_dummy_population()
    test_init_from_factors_replay_also_respects_the_split()
    test_dora_is_skipped_not_reinitialized()
    print("all peft_ga_init_populations tests passed")
