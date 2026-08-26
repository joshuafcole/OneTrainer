"""Tests for the runtime signed multiplier on PEFT modules (the slider knob).

Verifies that additive PEFT types (LoRA, LoHa, LoKr) scale only their delta
contribution by `multiplier` -- linearly, sign-aware, with 0.0 disabling the
adapter and 1.0 a no-op -- while non-additive types (DoRA / OFT) refuse a
non-default multiplier. Runs on small float Linears: CPU-only, no GPU, no
model download. Run with ``python -m pytest tests/test_peft_multiplier.py``
from the repo root, or ``python tests/test_peft_multiplier.py`` directly.
"""

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.module.LoRAModule import (  # noqa: E402
    DoRAModule,
    LoHaModule,
    LoKrModule,
    LoRAModule,
    OFTModule,
)

IN, OUT = 32, 16


def _x():
    torch.manual_seed(1)
    return torch.randn(4, IN)


def _linear():
    torch.manual_seed(0)
    return nn.Linear(IN, OUT)


def _make_lora():
    linear = _linear()
    m = LoRAModule("t.l", linear, rank=4, alpha=2.0)
    # zero-init lora_up makes the delta identically zero; give it real weights
    # so the multiplier has something to scale.
    torch.manual_seed(3)
    nn.init.normal_(m.lora_up.weight, std=0.5)
    return linear, m


def _make_loha():
    linear = _linear()
    m = LoHaModule("t.l", linear, rank=4, alpha=2.0)
    torch.manual_seed(3)
    nn.init.normal_(m.hada_w2_a, std=0.5)  # hada_w2_a is the zero-init factor; make delta nonzero
    return linear, m


def _make_lokr():
    linear = _linear()
    m = LoKrModule(
        "t.l", linear, dim=4, alpha=2.0,
        decompose_both=False, decompose_factor=-1, use_tucker=False,
        weight_decompose=False, dora_on_output=True, full_matrix=False,
        train_device=torch.device("cpu"), lokr_vec_trick=True,
    )
    torch.manual_seed(3)
    # The zero-init factor is lokr_w2_b in the decomposed path, or lokr_w2 if the
    # rank forced the full-matrix fallback. Randomize whichever exists so the
    # adapter delta is nonzero.
    if getattr(m, "use_w2", False):
        nn.init.normal_(m.lokr_w2, std=0.5)
    else:
        nn.init.normal_(m.lokr_w2_b, std=0.5)
    return linear, m


def _check_linear_scaling(make):
    """delta(m) == m * delta(1); delta(0)==0; delta(-1)==-delta(1)."""
    x = _x()
    linear, m = make()
    # Capture the unhooked base output BEFORE hooking replaces forward.
    base = linear(x).detach().clone()
    m.hook_to_module()

    m.set_multiplier(1.0)
    delta1 = (linear(x) - base).detach().clone()
    assert delta1.abs().max() > 1e-4, f"{make.__name__}: delta should be nonzero at m=1"

    for mult in (0.0, 0.5, 2.0, -1.0, -3.0):
        m.set_multiplier(mult)
        delta = (linear(x) - base).detach()
        assert torch.allclose(delta, mult * delta1, atol=1e-5), (
            f"{make.__name__}: delta(m={mult}) != m*delta(1)"
        )


def test_lora_multiplier_scales_linearly():
    _check_linear_scaling(_make_lora)


def test_loha_multiplier_scales_linearly():
    _check_linear_scaling(_make_loha)


def test_lokr_multiplier_scales_linearly():
    _check_linear_scaling(_make_lokr)


def test_default_multiplier_is_one():
    _, m = _make_lora()
    assert m.multiplier == 1.0


def test_dora_rejects_nondefault_multiplier():
    linear = _linear()
    m = DoRAModule("t.l", linear, 4, 2.0, train_device=torch.device("cpu"))
    m.hook_to_module()
    m.forward(_x())  # multiplier 1.0 is fine
    m.set_multiplier(-1.0)
    try:
        m.forward(_x())
    except NotImplementedError:
        return
    raise AssertionError("DoRA should reject a non-default multiplier")


def test_oft_rejects_nondefault_multiplier():
    linear = _linear()
    m = OFTModule("t.l", linear, oft_block_size=4, block_share=False, oft_scaled=False,
                  dropout_probability=0.0)
    m.hook_to_module()
    m.forward(_x())  # multiplier 1.0 is fine
    m.set_multiplier(2.0)
    try:
        m.forward(_x())
    except NotImplementedError:
        return
    raise AssertionError("OFT should reject a non-default multiplier")


if __name__ == "__main__":
    test_lora_multiplier_scales_linearly()
    test_loha_multiplier_scales_linearly()
    test_lokr_multiplier_scales_linearly()
    test_default_multiplier_is_one()
    test_dora_rejects_nondefault_multiplier()
    test_oft_rejects_nondefault_multiplier()
    print("all peft_multiplier tests passed")
