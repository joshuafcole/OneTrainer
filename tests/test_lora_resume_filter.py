"""Tests for narrower/wider LoRA layer-filter resume.

Resuming a LoRA whose module filter differs from the checkpoint's needs three
populations distinguished, not one:

- trainable (`lora_modules`): the current filter selects it, and the
  checkpoint has weights for it. Trained, hooked, saved.
- frozen (`frozen_lora_modules`): the checkpoint has weights for it, the
  current filter does not select it, but it is still a real target module in
  the base model. Loaded and hooked -- its contribution stays active in the
  forward pass -- but never trained.
- dummy (`dummy_lora_modules`): the checkpoint has a key with no
  corresponding real target module at all (a stale/foreign key). Preserved
  only so it round-trips through state_dict(); never hooked, never trained.

Before this, a narrower filter on resume either dropped the out-of-filter
weights or crashed trying to hook a dummy placeholder that had no real
module to hook into. See docs/LayerFilteredLoRAResume.md for the design
write-up.

Run with ``python -m pytest tests/test_lora_resume_filter.py`` from the repo
root.
"""

import torch
from torch import nn

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType

IN, OUT, RANK = 8, 8, 4


class _ThreeBlockNet(nn.Module):
    """Three independent Linear layers: enough to give trainable, frozen,
    and (via an injected foreign key) dummy each a distinct real target."""

    def __init__(self):
        super().__init__()
        self.block0 = nn.Linear(IN, OUT, bias=False)
        self.block1 = nn.Linear(IN, OUT, bias=False)
        self.block2 = nn.Linear(IN, OUT, bias=False)


class _NestedNet(nn.Module):
    """`outer` and `outer.inner` are both real Linear target modules, and the
    string "outer" is a literal prefix of "outer.inner" -- the ambiguity
    `__find_target_prefix`'s longest-match exists to resolve correctly."""

    def __init__(self):
        super().__init__()
        self.outer = nn.Linear(IN, OUT, bias=False)
        self.outer.inner = nn.Linear(IN, OUT, bias=False)
        self.sibling = nn.Linear(IN, OUT, bias=False)


def _config(regex: bool = False) -> TrainConfig:
    config = TrainConfig.default_values()
    config.peft_type = PeftType.LORA
    config.lora_rank = RANK
    config.lokr_dim = RANK
    config.lora_alpha = 1.0
    config.train_device = "cpu"
    config.layer_filter_regex = regex
    return config


def _wrapper(model: nn.Module, module_filter: list[str], regex: bool = False) -> LoRAModuleWrapper:
    return LoRAModuleWrapper(model, "lora", _config(regex=regex), module_filter)


def _randomize_up(module) -> torch.Tensor:
    """lora_up is zero-initialized (LoRA's delta-starts-at-zero convention), so
    a freshly re-initialized module would have a nonzero lora_down but a zero
    lora_up. Randomizing lora_up gives a value a fresh init could never
    reproduce, which is what "loaded, not reinitialized" needs to prove.
    """
    torch.manual_seed(3)
    nn.init.normal_(module.lora_up.weight, std=0.5)
    return module.lora_up.weight.detach().clone()


def _narrower_resume():
    """Build a checkpoint under filter {block0, block1}, add a foreign key
    with no corresponding target module, then resume under the narrower
    filter {block0}. Returns (wrapper_b, saved_state_dict, block1_up_before).
    """
    model = _ThreeBlockNet()
    wrapper_a = _wrapper(model, ["block0", "block1"])
    block1_up_before = _randomize_up(wrapper_a.lora_modules["block1"])
    state_dict = wrapper_a.state_dict()

    # A foreign/stale key: no target module in the base model has this name.
    state_dict["lora.ghost.lora_down.weight"] = torch.zeros(RANK, IN)
    state_dict["lora.ghost.lora_up.weight"] = torch.zeros(OUT, RANK)
    state_dict["lora.ghost.alpha"] = torch.tensor(1.0)

    wrapper_b = _wrapper(model, ["block0"])
    wrapper_b.load_state_dict(state_dict)
    return wrapper_b, state_dict, block1_up_before


# ---------------------------------------------------------------------------
# The core scenario
# ---------------------------------------------------------------------------


def test_narrower_resume_splits_into_three_populations():
    wrapper_b, _state_dict, _up = _narrower_resume()

    assert set(wrapper_b.lora_modules.keys()) == {"block0"}
    assert set(wrapper_b.frozen_lora_modules.keys()) == {"block1"}
    assert len(wrapper_b.dummy_lora_modules) > 0


def test_frozen_modules_are_not_trainable_but_hold_loaded_weights():
    wrapper_b, _state_dict, block1_up_before = _narrower_resume()

    frozen = wrapper_b.frozen_lora_modules["block1"]
    for param in frozen.parameters():
        assert not param.requires_grad

    assert torch.allclose(frozen.lora_up.weight, block1_up_before)
    assert frozen.lora_up.weight.abs().sum() > 0


def test_dummy_modules_preserve_the_foreign_keys_verbatim():
    wrapper_b, state_dict, _up = _narrower_resume()

    dummy_state = {}
    for module in wrapper_b.dummy_lora_modules.values():
        dummy_state |= module.state_dict()

    for key in ("lora.ghost.lora_down.weight", "lora.ghost.lora_up.weight", "lora.ghost.alpha"):
        assert key in dummy_state
        assert torch.equal(dummy_state[key], state_dict[key])


def test_state_dict_round_trips_every_loaded_key():
    """The one that catches a wrong population in state_dict(): every key
    that went into load_state_dict must come back out -- trainable, frozen,
    and dummy alike -- or a subsequent save silently drops weights."""
    wrapper_b, state_dict, _up = _narrower_resume()

    round_tripped = wrapper_b.state_dict()
    assert set(round_tripped.keys()) == set(state_dict.keys())
    for key, value in state_dict.items():
        assert torch.equal(round_tripped[key], value)


# ---------------------------------------------------------------------------
# Per-consumer coverage: which population each method iterates
# ---------------------------------------------------------------------------


def test_parameters_are_trainable_only():
    wrapper_b, _state_dict, _up = _narrower_resume()

    wrapper_param_ids = {id(p) for p in wrapper_b.parameters()}
    trainable_ids = {id(p) for p in wrapper_b.lora_modules["block0"].parameters()}
    frozen_ids = {id(p) for p in wrapper_b.frozen_lora_modules["block1"].parameters()}

    assert trainable_ids <= wrapper_param_ids
    assert not (frozen_ids & wrapper_param_ids)


def test_requires_grad_leaves_frozen_modules_frozen():
    wrapper_b, _state_dict, _up = _narrower_resume()

    wrapper_b.requires_grad_(True)
    for param in wrapper_b.lora_modules["block0"].parameters():
        assert param.requires_grad
    for param in wrapper_b.frozen_lora_modules["block1"].parameters():
        assert not param.requires_grad


def test_modules_includes_frozen_but_excludes_dummy():
    wrapper_b, _state_dict, _up = _narrower_resume()

    expected = 0
    for module in wrapper_b.lora_modules.values():
        expected += len(list(module.modules()))
    for module in wrapper_b.frozen_lora_modules.values():
        expected += len(list(module.modules()))

    assert len(wrapper_b.modules()) == expected


def test_hook_to_module_covers_real_and_never_touches_dummy():
    wrapper_b, _state_dict, _up = _narrower_resume()

    wrapper_b.hook_to_module()  # would raise NotImplementedError if a dummy got hooked
    assert wrapper_b.lora_modules["block0"].is_applied
    assert wrapper_b.frozen_lora_modules["block1"].is_applied

    wrapper_b.remove_hook_from_module()
    assert not wrapper_b.lora_modules["block0"].is_applied
    assert not wrapper_b.frozen_lora_modules["block1"].is_applied


def test_to_moves_frozen_modules_too():
    wrapper_b, _state_dict, _up = _narrower_resume()

    wrapper_b.to(dtype=torch.float64)
    assert wrapper_b.lora_modules["block0"].lora_up.weight.dtype == torch.float64
    assert wrapper_b.frozen_lora_modules["block1"].lora_up.weight.dtype == torch.float64


def test_set_dropout_only_touches_trainable():
    wrapper_b, _state_dict, _up = _narrower_resume()

    wrapper_b.set_dropout(0.5)
    assert wrapper_b.lora_modules["block0"].dropout.p == 0.5
    assert wrapper_b.frozen_lora_modules["block1"].dropout.p == 0.0


def test_set_multiplier_scales_trainable_and_frozen_alike():
    """A frozen module is hooked and contributes to the forward pass exactly
    like a trainable one, so the strength slider must scale both uniformly --
    otherwise "0.0 disables the adapter" would leave inherited layers active,
    and a partial-strength preview would visibly desync trainable vs.
    inherited layers. Dummies are never hooked, so a multiplier there is an
    inert no-op either way; the wrapper simply never touches them.
    """
    wrapper_b, _state_dict, _up = _narrower_resume()

    wrapper_b.set_multiplier(0.5)
    assert wrapper_b.lora_modules["block0"].multiplier == 0.5
    assert wrapper_b.frozen_lora_modules["block1"].multiplier == 0.5
    for module in wrapper_b.dummy_lora_modules.values():
        assert module.multiplier == 1.0  # untouched -- never iterated


def test_prune_drops_dummy_but_keeps_frozen_and_trainable():
    wrapper_b, _state_dict, _up = _narrower_resume()

    wrapper_b.prune()
    assert wrapper_b.dummy_lora_modules == {}
    assert "block1" in wrapper_b.frozen_lora_modules
    assert "block0" in wrapper_b.lora_modules


# ---------------------------------------------------------------------------
# Equal and wider filters: the narrower case must not regress these
# ---------------------------------------------------------------------------


def test_equal_filter_resume_creates_no_frozen_or_dummy():
    model = _ThreeBlockNet()
    wrapper_a = _wrapper(model, ["block0", "block1"])
    state_dict = wrapper_a.state_dict()

    wrapper_b = _wrapper(model, ["block0", "block1"])
    wrapper_b.load_state_dict(state_dict)

    assert set(wrapper_b.lora_modules.keys()) == {"block0", "block1"}
    assert wrapper_b.frozen_lora_modules == {}
    assert wrapper_b.dummy_lora_modules == {}


def test_wider_filter_resume_creates_fresh_trainable_modules():
    model = _ThreeBlockNet()
    wrapper_a = _wrapper(model, ["block0"])
    state_dict = wrapper_a.state_dict()

    wrapper_b = _wrapper(model, ["block0", "block1", "block2"])
    wrapper_b.load_state_dict(state_dict)

    assert set(wrapper_b.lora_modules.keys()) == {"block0", "block1", "block2"}
    assert wrapper_b.frozen_lora_modules == {}
    assert wrapper_b.dummy_lora_modules == {}

    new_module = wrapper_b.lora_modules["block1"]
    for param in new_module.parameters():
        assert param.requires_grad
    # untouched by load_state_dict -- still at the fresh LoRA zero-init.
    assert torch.equal(new_module.lora_up.weight, torch.zeros_like(new_module.lora_up.weight))


# ---------------------------------------------------------------------------
# The longest-prefix-match case
# ---------------------------------------------------------------------------


def test_find_target_prefix_picks_the_longest_match():
    """"outer" is a literal string prefix of "outer.inner" -- both are real
    target modules. A checkpoint trained only "outer.inner"; resuming under a
    filter that selects neither must freeze it as "outer.inner", not misroute
    it onto "outer" (which has no data for it and would reject the load).
    """
    model = _NestedNet()
    wrapper_a = _wrapper(model, [r"^outer\.inner$"], regex=True)
    assert set(wrapper_a.lora_modules.keys()) == {"outer.inner"}
    state_dict = wrapper_a.state_dict()

    wrapper_b = _wrapper(model, [r"^sibling$"], regex=True)
    wrapper_b.load_state_dict(state_dict)  # would raise under a first-match bug

    assert set(wrapper_b.frozen_lora_modules.keys()) == {"outer.inner"}
    assert "outer" not in wrapper_b.frozen_lora_modules


if __name__ == "__main__":
    import sys

    failures = 0
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
    if failures:
        print(f"{failures}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"all {len(tests)} tests passed")
