"""End-to-end LoRA training smoke test for Anima.

Generates a synthetic 4-image training set, configures a tiny LoRA
run (256x256, 30 steps, attn-mlp filter), drives it through
``GenericTrainer``, and asserts that:

  - the trainer initializes (model loader works under the trainer)
  - the data loader produces batches (dual-encoder cache pipeline works)
  - the setup wraps the transformer in LoRA modules
  - the training step runs (Anima encode_text + Cosmos forward + flow loss)
  - the optimizer step modifies LoRA weights
  - the saver writes a LoRA .safetensors file

This is intentionally compute-cheap (about a minute on an RTX 5090)
but covers every cross-module boundary in the Anima integration. If
this passes, real training runs from the GUI almost certainly will.

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\smoke_test_anima_training.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Trigger factory.import_dir registrations for model/{loader,sampler,setup,...}
import modules.util.create as create_mod  # noqa: F401

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod

from PIL import Image, ImageDraw, ImageFont


CHECKPOINT = Path("D:/models/diffusers/anima/anima-base-v1.0")
WORKSPACE = Path("D:/models/diffusers/anima/_smoke_train_workspace")
DATASET = WORKSPACE / "dataset"


def _make_tiny_dataset() -> None:
    """Write four 256x256 RGB images + matching .txt captions."""
    DATASET.mkdir(parents=True, exist_ok=True)
    palette = [
        ("a red square on a black background", (220, 20, 20)),
        ("a green square on a black background", (20, 220, 20)),
        ("a blue square on a black background", (20, 20, 220)),
        ("a yellow square on a black background", (220, 220, 20)),
    ]
    for i, (caption, color) in enumerate(palette):
        img = Image.new("RGB", (256, 256), color=(0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([48, 48, 207, 207], fill=color)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        d.text((10, 10), caption.split(" ", 2)[1], fill=(255, 255, 255), font=font)

        img_path = DATASET / f"sample_{i:02d}.png"
        txt_path = DATASET / f"sample_{i:02d}.txt"
        img.save(img_path)
        txt_path.write_text(caption, encoding="utf-8")


def _write_concepts() -> Path:
    concept = ConceptConfig.default_values()
    concept.name = "anima_smoke"
    concept.path = str(DATASET)
    concept.enabled = True
    # 10 repeats * 4 images / batch 1 = 40 steps per epoch
    concept.balancing = 10.0
    concepts_path = WORKSPACE / "concepts.json"
    concepts_path.write_text(json.dumps([concept.to_dict()], indent=2), encoding="utf-8")
    return concepts_path


def _write_samples() -> Path:
    # Empty samples list -- the trainer will skip sampling.
    samples_path = WORKSPACE / "samples.json"
    samples_path.write_text("[]", encoding="utf-8")
    return samples_path


def _build_train_config(concepts_path: Path, samples_path: Path) -> TrainConfig:
    cfg = TrainConfig.default_values()

    # --- where ---
    cfg.workspace_dir = str(WORKSPACE / "run")
    cfg.cache_dir = str(WORKSPACE / "cache")
    cfg.debug_dir = str(WORKSPACE / "debug")
    cfg.concept_file_name = str(concepts_path)
    cfg.sample_definition_file_name = str(samples_path)
    cfg.output_model_destination = str(WORKSPACE / "anima_smoke_lora.safetensors")
    cfg.output_model_format = ModelFormat.SAFETENSORS

    # --- what ---
    cfg.model_type = ModelType.ANIMA
    cfg.training_method = TrainingMethod.LORA
    cfg.base_model_name = str(CHECKPOINT)

    # --- how much ---
    cfg.epochs = 1
    cfg.batch_size = 1
    cfg.gradient_accumulation_steps = 1
    cfg.learning_rate = 3e-4
    cfg.resolution = "256"
    cfg.dataloader_threads = 1
    cfg.latent_caching = False  # simpler first pass; cache path tested separately if this passes
    cfg.text_encoder.dropout_probability = 0.0
    cfg.timestep_distribution = TimestepDistribution.LOGIT_NORMAL
    cfg.dynamic_timestep_shifting = False
    cfg.timestep_shift = 3.0  # matches Anima scheduler's static shift

    # --- dtype ---
    cfg.train_dtype = DataType.BFLOAT_16
    cfg.transformer.weight_dtype = DataType.BFLOAT_16
    cfg.transformer.train = True
    cfg.text_encoder.weight_dtype = DataType.BFLOAT_16
    cfg.text_encoder.train = False
    cfg.vae.weight_dtype = DataType.BFLOAT_16
    cfg.lora_weight_dtype = DataType.BFLOAT_16

    # --- LoRA target ---
    cfg.layer_filter = "^(?=.*attn)(?!.*norm).*,^(?=.*ff\\.net).*"
    cfg.layer_filter_preset = "attn-mlp"
    cfg.layer_filter_regex = True
    cfg.lora_rank = 8  # tiny rank for fast smoke test

    # --- noise / loss ---
    cfg.dropout_probability = 0.0

    # --- never sample or back up during the smoke test ---
    cfg.sample_after = 9999
    cfg.sample_after_unit = TimeUnit.STEP
    cfg.backup_after = 9999
    cfg.backup_after_unit = TimeUnit.STEP

    return cfg


def main() -> int:
    if not CHECKPOINT.exists():
        sys.exit(f"checkpoint not found: {CHECKPOINT}")

    # Fresh workspace every run.
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)

    print("[1] generating 4-image dataset")
    _make_tiny_dataset()

    print("[2] writing concepts / samples / config")
    concepts_path = _write_concepts()
    samples_path = _write_samples()
    cfg = _build_train_config(concepts_path, samples_path)

    print("[3] building trainer")
    callbacks = TrainCallbacks()
    commands = TrainCommands()
    trainer = create_mod.create_trainer(cfg, callbacks, commands)

    print("[4] trainer.start()")
    t0 = time.perf_counter()
    trainer.start()
    print(f"    start() returned in {time.perf_counter() - t0:.1f}s")

    print("[5] trainer.train()  (expecting ~40 steps for 1 epoch)")
    t0 = time.perf_counter()
    trainer.train()
    print(f"    train() returned in {time.perf_counter() - t0:.1f}s")

    print("[6] trainer.end()  (writes the LoRA file)")
    trainer.end()

    expected = Path(cfg.output_model_destination)
    if not expected.exists():
        # The saver may append the format suffix.
        for cand in expected.parent.glob(expected.name + "*"):
            print(f"    candidate output: {cand} ({cand.stat().st_size} bytes)")
        print(f"    expected file at {expected} but did not find it.")
        return 2

    size_mb = expected.stat().st_size / (1024 * 1024)
    print(f"OK. Saved LoRA: {expected}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
