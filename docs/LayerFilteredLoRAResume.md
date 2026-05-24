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
When a wider checkpoint is loaded into a narrower wrapper, leftover checkpoint keys are stored as dummy modules.
Those dummy modules exist only to preserve state for saving.
They cannot be hooked into the model, so setup crashes when hook installation reaches them.

## Desired Behavior

- Selected layers become normal trainable PEFT modules.
- Filtered-out checkpoint layers that still map to real target modules become frozen inherited PEFT modules.
- Frozen inherited modules stay hooked so their contribution remains active.
- Frozen inherited modules do not enter parameter groups.
- Frozen inherited modules do not receive training dropout.
- Truly unmatched checkpoint keys remain save-only state or produce a targeted compatibility error.

## Design

Implement the change in the shared `LoRAModuleWrapper` so all model families benefit.

Wrapper state is split into three buckets:

- `lora_modules`: trainable modules created from the active filter.
- `frozen_lora_modules`: real modules loaded from checkpoint leftovers that still map to filtered-out target layers.
- `dummy_lora_modules`: save-only placeholders for checkpoint leftovers that do not map to a current real target.

Behavior by operation:

- `parameters()`: returns only trainable modules.
- `requires_grad_()`: toggles only trainable modules and keeps inherited modules frozen.
- `hook_to_module()` and `remove_hook_from_module()`: operate on trainable and frozen real modules, never dummies.
- `state_dict()`: returns trainable, frozen, and dummy state so saves preserve inherited weights.
- `set_dropout()`: applies only to trainable modules; inherited frozen modules keep dropout at zero.
- `to()`: moves both trainable and frozen real modules to the requested device and dtype.

## Compatibility Rules

Supported:

- Resume from a wider checkpoint into a narrower filter when the checkpoint keys still target the same base model modules.
- Resume from the same filter.
- Resume into a wider filter, where newly added layers start fresh.

Not expanded in this change:

- Burning inherited adapters directly into base weights.
- Cross-PEFT conversion between LoRA, LoHa, DoRA, and OFT.
- Migration of genuinely unknown checkpoint key layouts.

## Validation

- Static validation: Python compile check for the touched modules.
- Behavior validation: load a wider base LoRA with a narrower filter and confirm setup no longer throws during hook installation.
- Regression check: saving still includes inherited frozen adapter weights.

## Follow-Up

- Add a targeted compatibility error for unresolved leftovers that cannot be mapped back to a real target layer.
- Add an explicit regression test around wider-to-narrower resume once a lightweight test harness exists for PEFT wrapper loading.