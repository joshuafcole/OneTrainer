"""Rank-mismatch detection when resuming from a PEFT checkpoint.

`LoRAModuleWrapper._check_rank_matches` reads the rank off one tensor per PEFT
type. For LoHa that tensor has to be the *down* projection: `create_layer()`
returns `(down, up)` and `LoHaModule.initialize_weights` binds them as
`hada_w1_b, hada_w1_a = self.create_layer()`, so `hada_w1_b` is `[rank, in]`
while `hada_w1_a` is `[out, rank]`.

Reading `shape[0]` of `hada_w1_a` therefore returns the output width, which
raises a spurious mismatch on every LoHa resume where `out != rank` -- the
false positive that led the check to be disabled outright.
"""

import pytest
import torch
from torch import nn

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType

# Deliberately unequal to any rank under test, so a check that reads the wrong
# tensor cannot accidentally agree.
IN_FEATURES = 32
OUT_FEATURES = 128


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)


def _wrapper(peft_type: PeftType, rank: int) -> LoRAModuleWrapper:
    config = TrainConfig.default_values()
    config.peft_type = peft_type
    config.lora_rank = rank
    config.lokr_dim = rank
    config.lora_alpha = 1.0
    config.train_device = "cpu"
    return LoRAModuleWrapper(_TinyNet(), "lora", config)


@pytest.mark.parametrize("peft_type", [PeftType.LORA, PeftType.LOHA])
def test_matching_rank_loads_without_error(peft_type: PeftType):
    """A checkpoint saved at rank R must load into a wrapper configured at rank R."""
    rank = 16
    assert rank not in (IN_FEATURES, OUT_FEATURES)

    state_dict = _wrapper(peft_type, rank).state_dict()
    _wrapper(peft_type, rank).load_state_dict(state_dict)


@pytest.mark.parametrize("peft_type", [PeftType.LORA, PeftType.LOHA])
def test_mismatched_rank_is_rejected(peft_type: PeftType):
    """The check still has to catch a genuine mismatch, or re-enabling it is pointless."""
    state_dict = _wrapper(peft_type, 16).state_dict()

    with pytest.raises(ValueError, match="mismatch"):
        _wrapper(peft_type, 8).load_state_dict(state_dict)


def test_loha_rank_is_read_from_the_down_projection():
    """Pin the tensor semantics the check depends on."""
    state_dict = _wrapper(PeftType.LOHA, 16).state_dict()

    down = next(v for k, v in state_dict.items() if k.endswith(".hada_w1_b"))
    up = next(v for k, v in state_dict.items() if k.endswith(".hada_w1_a"))

    assert tuple(down.shape) == (16, IN_FEATURES)
    assert tuple(up.shape) == (OUT_FEATURES, 16)
