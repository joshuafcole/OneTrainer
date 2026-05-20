"""Smoke test for the Anima diffusers pin.

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\smoke_test_anima.py

Confirms every diffusers/transformers class we plan to depend on for the
Anima integration is importable from the pinned rmatif/diffusers@anima
branch, and reports its on-disk path so we know we are picking up the
editable install in src/diffusers/ and not some other diffusers in
site-packages.
"""

from __future__ import annotations

import importlib
import sys
import traceback


def _probe(module_path: str, names: list[str]) -> bool:
    ok = True
    try:
        module = importlib.import_module(module_path)
    except Exception:  # noqa: BLE001
        print(f"  IMPORT FAIL: {module_path}")
        traceback.print_exc()
        return False

    print(f"  module: {module_path}  ->  {getattr(module, '__file__', '<no __file__>')}")
    for name in names:
        sym = getattr(module, name, None)
        if sym is None:
            print(f"    MISSING: {name}")
            ok = False
        else:
            print(f"    ok:      {name}  ({type(sym).__name__})")
    return ok


def main() -> int:
    print(f"python : {sys.version.split()[0]}")
    print(f"prefix : {sys.prefix}")
    print()

    all_ok = True

    print("[1] Anima-specific (added by PR #13732):")
    all_ok &= _probe(
        "diffusers.modular_pipelines.anima",
        ["AnimaModularPipeline", "AnimaAutoBlocks"],
    )
    all_ok &= _probe(
        "diffusers.models.condition_embedders.condition_embedder_anima",
        ["AnimaTextConditioner"],
    )
    all_ok &= _probe(
        "diffusers.loaders.lora_pipeline",
        ["AnimaLoraLoaderMixin"],
    )
    print()

    print("[2] Existing diffusers classes Anima reuses:")
    all_ok &= _probe(
        "diffusers",
        [
            "CosmosTransformer3DModel",
            "AutoencoderKLQwenImage",
            "FlowMatchEulerDiscreteScheduler",
        ],
    )
    print()

    print("[3] Text-encoder side (transformers):")
    all_ok &= _probe(
        "transformers",
        ["Qwen2Tokenizer", "Qwen3Model", "T5TokenizerFast"],
    )
    print()

    print("[4] Diffusers version / git sha:")
    try:
        import diffusers  # type: ignore[import]

        print(f"  diffusers.__version__ : {getattr(diffusers, '__version__', '?')}")
        print(f"  diffusers.__file__    : {diffusers.__file__}")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        all_ok = False
    print()

    if all_ok:
        print("ALL OK")
        return 0
    print("FAILURES ABOVE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
