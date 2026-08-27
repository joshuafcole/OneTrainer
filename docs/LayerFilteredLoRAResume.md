# Layer-Filtered LoRA Resume Plan

## Goal

Support resuming from an existing LoRA while training a narrower active layer filter.

Example:

- Base LoRA was trained with `detail`.
- New run uses `attn-mlp`.
- Filtered-out `detail` layers remain active in the forward pass.
- Only the `attn-mlp` subset is trainable and receives optimizer state.

## Current Failure Mode

The shared LoRA wrapper builds real PEFT modules only for the currently selected filter.
When a wider checkpoint is loaded into a narrower wrapper, leftover checkpoint keys were either
dropped or stored as save-only placeholders with no live target module. Those placeholders cannot
be hooked into the model, so setup crashed when hook installation reached them.

## Desired Behavior

- Selected layers become normal trainable PEFT modules.
- Filtered-out checkpoint layers that still map to real target modules become frozen inherited PEFT modules.
- Frozen inherited modules stay hooked so their contribution remains active.
- Frozen inherited modules do not enter parameter groups.
- Frozen inherited modules do not receive training dropout.
- Truly unmatched checkpoint keys remain save-only state.

## Design

Implemented in the shared `LoRAModuleWrapper` so all model families benefit.

Wrapper state is split into three buckets:

- `lora_modules`: trainable modules created from the active filter.
- `frozen_lora_modules`: real modules loaded from checkpoint leftovers that still map to filtered-out target layers.
- `dummy_lora_modules`: save-only placeholders for checkpoint leftovers that do not map to a current real target.

A module is only ever frozen if `target_modules` (every `Linear`/`Conv2d` in the base model, collected once
at construction) has a real entry for it; everything else becomes a dummy. Because one target module's
name can be a literal string prefix of another's, resolving a leftover key to a target module always picks
the *longest* matching prefix.

Behavior by operation:

- `parameters()`: returns only trainable modules.
- `requires_grad_()`: toggles only trainable modules and keeps inherited modules frozen.
- `hook_to_module()` and `remove_hook_from_module()`: operate on trainable and frozen real modules, never dummies.
- `state_dict()`: returns trainable, frozen, and dummy state so saves preserve inherited weights.
- `set_dropout()`: applies only to trainable modules; inherited frozen modules keep dropout at zero.
- `set_multiplier()`: applies to trainable **and** frozen modules alike. A frozen module is hooked and
  contributes to the forward pass exactly like a trainable one, so the strength slider is a property of the
  whole active adapter, not just the layers currently being trained -- `multiplier=0.0` has to give an
  actual frozen-base pass, and a partial-strength preview must not visibly desync trainable vs. inherited
  layers. Dummy modules are never hooked, so a multiplier on them would be inert either way.
- `to()`: moves both trainable and frozen real modules to the requested device and dtype.

## Compatibility Rules

Supported:

- Resume from a wider checkpoint into a narrower filter when the checkpoint keys still target the same base model modules.
- Resume from the same filter.
- Resume into a wider filter, where newly added layers start fresh. This required one companion fix: the
  per-module strict `load_state_dict` call is now skipped entirely for a trainable module that has zero
  matching keys in the checkpoint (a module the wider filter newly selects), rather than strict-loading an
  empty dict into it. A module that *does* have some keys in the checkpoint still loads strictly, so a
  genuinely incomplete or corrupt entry for an existing module still raises.

Not expanded in this change:

- Burning inherited adapters directly into base weights.
- Cross-PEFT conversion between LoRA, LoHa, DoRA, and OFT.
- Migration of genuinely unknown checkpoint key layouts.
- Fused qkv module groups (`FusedModuleGroup`, upstream's fused/split output-format support). A leftover
  checkpoint key that belongs to a fused group has no matching entry in `target_modules` (the group's
  synthetic fused linear is built only when its whole block is selected), so it always falls back to
  `dummy_lora_modules` -- preserved for save, but not hooked or active -- even when every one of the
  group's real leaves still exists in the base model. In the common case this doesn't come up: fused
  output formats are opt-in per model/format, and a split-format wrapper freezes each leaf individually,
  same as any other target module. `check_fusion_match` already rejects a fused/split *format* mismatch
  outright, independent of layer filtering, so that failure mode was never in scope here.

## Validation

- Static validation: Python compile check for the touched modules.
- Behavior validation: `tests/test_lora_resume_filter.py` builds a checkpoint under one filter and resumes
  it under a narrower, an equal, and a wider filter, asserting the three populations and every
  per-operation consumer above.
- Regression check: `state_dict()` round-trips every key that was loaded, including inherited frozen and
  foreign dummy keys.

## Follow-Up

- Add a targeted compatibility error (or a frozen `FusedModuleGroup`) for a fused-group checkpoint key that
  falls entirely outside the current filter, instead of silently degrading it to dummy (save-only).
