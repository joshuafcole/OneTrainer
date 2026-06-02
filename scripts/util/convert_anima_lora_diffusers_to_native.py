"""Convert an Anima LoRA from diffusers naming to ComfyUI-loadable native naming.

OneTrainer's AnimaLoRASaver writes LoRA weights in diffusers /
PeftAdapterMixin convention:

  transformer.transformer_blocks.<i>.<attn1|attn2|ff>.<...>.lora_up.weight

ComfyUI's LoraLoader applies LoRAs to a model loaded by UNETLoader
from the original Anima ``.safetensors`` checkpoint. That model uses
the **native** key naming:

  net.blocks.<i>.<self_attn|cross_attn|mlp>.<...>.weight

Anima LoRAs distributed in the wild therefore use the convention:

  diffusion_model.blocks.<i>.<self_attn|cross_attn|mlp>.<...>.lora_up.weight

(no ``net.`` prefix in the LoRA -- ComfyUI's loader handles that.
The transformer-component prefix is ``diffusion_model.``; the
text-conditioner component, if ever LoRA-tuned, uses
``diffusion_model.llm_adapter.``.)

This script inverts the rename table in the rmatif/diffusers PR --
specifically ``_convert_non_diffusers_anima_lora_to_diffusers`` at
``src/diffusers/loaders/lora_conversion_utils.py``.

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\convert_anima_lora_diffusers_to_native.py ^
        --in  D:\\path\\to\\anima_lora.safetensors ^
        --out D:\\path\\to\\anima_lora_comfy.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


# Inverse of ``_convert_non_diffusers_anima_lora_to_diffusers`` (longest
# patterns first, because ``transformer_blocks.`` would otherwise also
# match ``blocks.``). Each entry is (diffusers_token, native_token).
INVERSE_RENAME: list[tuple[str, str]] = [
    # block-level normalization adapters
    ("norm1.linear_1", "adaln_modulation_self_attn.1"),
    ("norm1.linear_2", "adaln_modulation_self_attn.2"),
    ("norm2.linear_1", "adaln_modulation_cross_attn.1"),
    ("norm2.linear_2", "adaln_modulation_cross_attn.2"),
    ("norm3.linear_1", "adaln_modulation_mlp.1"),
    ("norm3.linear_2", "adaln_modulation_mlp.2"),
    # attention projections
    ("attn1.to_q", "self_attn.q_proj"),
    ("attn1.to_k", "self_attn.k_proj"),
    ("attn1.to_v", "self_attn.v_proj"),
    ("attn1.to_out.0", "self_attn.output_proj"),
    ("attn2.to_q", "cross_attn.q_proj"),
    ("attn2.to_k", "cross_attn.k_proj"),
    ("attn2.to_v", "cross_attn.v_proj"),
    ("attn2.to_out.0", "cross_attn.output_proj"),
    # feed-forward
    ("ff.net.0.proj", "mlp.layer1"),
    ("ff.net.2", "mlp.layer2"),
    # final layer (rarely LoRA-targeted, included for completeness)
    ("norm_out.linear_1", "final_layer.adaln_modulation.1"),
    ("norm_out.linear_2", "final_layer.adaln_modulation.2"),
    ("proj_out", "final_layer.linear"),
    # input embeddings (rarely LoRA-targeted)
    ("time_embed.t_embedder", "t_embedder.1"),
    ("time_embed.norm", "t_embedding_norm"),
    ("patch_embed.proj", "x_embedder.proj.1"),
    # block-collection prefix -- has to be LAST so it does not eat any
    # earlier substring like ``transformer_blocks.`` matching ``blocks.``.
    ("transformer_blocks.", "blocks."),
]


def _convert_one(key: str) -> str:
    """Rewrite one diffusers-naming LoRA key to native naming.

    Returns the new key. Pass-through entries (unrecognized prefixes)
    are returned with a `diffusion_model.` prefix and no body changes so
    downstream metadata keys (which we don't usually have) still
    survive.
    """
    # 0. bundled TI embeddings ride along in the same file under
    # ``bundle_emb.<placeholder>.<qwen|qwen_out|t5>``. They are NOT transformer
    # weights; ComfyUI looks for them under that exact prefix, so pass them
    # through untouched. The component handling below would otherwise bury them
    # under ``diffusion_model.`` and the embedding would silently never load.
    if key.startswith("bundle_emb."):
        return key

    # 1. strip the diffusers-component prefix
    if key.startswith("transformer."):
        body = key.removeprefix("transformer.")
        component_prefix = "diffusion_model."
    elif key.startswith("text_conditioner."):
        body = key.removeprefix("text_conditioner.")
        component_prefix = "diffusion_model.llm_adapter."
    else:
        # Already-native or unfamiliar key; pass through with default prefix.
        return f"diffusion_model.{key}"

    # 2. apply the inverse rename table.
    for diffusers_token, native_token in INVERSE_RENAME:
        body = body.replace(diffusers_token, native_token)

    return component_prefix + body


def convert_state_dict(sd: dict[str, "Tensor"]) -> dict[str, "Tensor"]:  # noqa: F821
    converted: dict[str, "Tensor"] = {}  # noqa: F821
    collisions: list[tuple[str, str]] = []
    for k, v in sd.items():
        new_k = _convert_one(k)
        if new_k in converted:
            collisions.append((k, new_k))
        converted[new_k] = v
    if collisions:
        print("WARNING: key collisions after rename (last value wins):", file=sys.stderr)
        for old, new in collisions[:10]:
            print(f"  {old}  ->  {new}", file=sys.stderr)
    return converted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="input",  required=True, help="diffusers-naming LoRA .safetensors")
    parser.add_argument("--out", dest="output", required=True, help="where to write the native-naming LoRA")
    parser.add_argument("--print", action="store_true", help="dump key mapping to stdout")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"input does not exist: {in_path}")

    print(f"loading  {in_path}")
    sd = load_file(str(in_path))
    print(f"  {len(sd)} keys")

    converted = convert_state_dict(sd)
    if args.print:
        for new_k in sorted(converted.keys())[:16]:
            print(f"  {new_k}")
        print(f"  ... ({len(converted)} total)")

    # Spot-check: every block-level LoRA key should now start with
    # `diffusion_model.blocks.N.<self_attn|cross_attn|mlp|adaln_modulation_*>`.
    suspicious = [k for k in converted if k.startswith("diffusion_model.transformer_blocks.")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} keys still have 'transformer_blocks.' -- rename table missed:")
        for k in suspicious[:5]:
            print(f"  {k}")

    print(f"writing  {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(out_path))
    print(f"wrote {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
