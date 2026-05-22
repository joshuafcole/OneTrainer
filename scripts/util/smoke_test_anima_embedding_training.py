"""End-to-end pure-embedding (textual inversion) training smoke test for Anima.

This is the embedding-only sibling of smoke_test_anima_training.py (which
covers LoRA). It generates a synthetic 4-image set whose captions all
contain the trained placeholder token, configures a tiny EMBEDDING run
(256x256, ~40 steps, no LoRA adapter), drives it through GenericTrainer,
and asserts that:

  - the trainer initializes with the EMBEDDING factory trio
    (AnimaEmbeddingModelLoader / AnimaEmbeddingSetup / AnimaEmbeddingModelSaver)
  - the data loader runs Qwen3 live (text cache disabled for embedding runs)
  - the setup injects the placeholder token into the Qwen3 word table and
    leaves the transformer / conditioner / Qwen3 / VAE frozen
  - **the trained token vector's norm actually moves across training** --
    i.e. gradients reach the embedding (the open question from the joint
    embedding+LoRA work) -- with no LoRA adapter to mask a dead embedding
  - the saver writes a standalone embedding .safetensors file

Run with the OneTrainer venv active:
    venv\\Scripts\\python.exe scripts\\util\\smoke_test_anima_embedding_training.py
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
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod

import torch

from PIL import Image, ImageDraw, ImageFont

CHECKPOINT = Path("D:/models/diffusers/anima/anima-base-v1.0")
WORKSPACE = Path("D:/models/diffusers/anima/_smoke_embedding_workspace")
DATASET = WORKSPACE / "dataset"

# The placeholder must appear in every caption so the token receives
# gradient on every step. Real words survive the Qwen3 reformulation
# better than random strings, but for a gradient-flow smoke test a
# distinctive placeholder is what we want.
PLACEHOLDER = "<smoketoken>"


def _make_tiny_dataset() -> None:
    """Write four 256x256 RGB images + captions that all use PLACEHOLDER."""
    DATASET.mkdir(parents=True, exist_ok=True)
    palette = [
        (f"a photo of {PLACEHOLDER}, a red square on a black background", (220, 20, 20)),
        (f"a photo of {PLACEHOLDER}, a green square on a black background", (20, 220, 20)),
        (f"a photo of {PLACEHOLDER}, a blue square on a black background", (20, 20, 220)),
        (f"a photo of {PLACEHOLDER}, a yellow square on a black background", (220, 220, 20)),
    ]
    for i, (caption, color) in enumerate(palette):
        img = Image.new("RGB", (256, 256), color=(0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([48, 48, 207, 207], fill=color)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        d.text((10, 10), str(i), fill=(255, 255, 255), font=font)

        img.save(DATASET / f"sample_{i:02d}.png")
        (DATASET / f"sample_{i:02d}.txt").write_text(caption, encoding="utf-8")


def _write_concepts() -> Path:
    concept = ConceptConfig.default_values()
    concept.name = "anima_embedding_smoke"
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
    cfg.output_model_destination = str(WORKSPACE / "anima_smoke_embedding.safetensors")
    cfg.output_model_format = ModelFormat.SAFETENSORS

    # --- what ---
    cfg.model_type = ModelType.ANIMA
    cfg.training_method = TrainingMethod.EMBEDDING
    cfg.base_model_name = str(CHECKPOINT)

    # --- the embedding being trained ---
    cfg.embedding.placeholder = PLACEHOLDER
    cfg.embedding.train = True
    cfg.embedding.token_count = 4  # complex gestalt; default 1 is too little
    cfg.embedding.initial_embedding_text = "square"  # seed from a real word

    # --- how much ---
    cfg.epochs = 1
    cfg.batch_size = 1
    cfg.gradient_accumulation_steps = 1
    # High embedding LR so the token moves measurably in ~40 steps.
    cfg.embedding_learning_rate = 1e-2
    cfg.resolution = "256"
    cfg.dataloader_threads = 1
    # latent_caching stays on; the text cache is disabled automatically for
    # embedding runs so Qwen3 runs live under grad.
    cfg.latent_caching = True
    cfg.text_encoder.train_embedding = True
    cfg.text_encoder.dropout_probability = 0.0
    cfg.timestep_distribution = TimestepDistribution.LOGIT_NORMAL
    cfg.dynamic_timestep_shifting = False
    cfg.timestep_shift = 3.0  # matches Anima scheduler's static shift
    cfg.preserve_embedding_norm = False  # let the norm move freely for the assertion

    # --- dtype ---
    cfg.train_dtype = DataType.BFLOAT_16
    cfg.transformer.weight_dtype = DataType.BFLOAT_16
    cfg.transformer.train = False  # pure embedding -- transformer is frozen
    cfg.text_encoder.weight_dtype = DataType.BFLOAT_16
    cfg.text_encoder.train = False
    cfg.vae.weight_dtype = DataType.BFLOAT_16
    cfg.embedding_weight_dtype = DataType.FLOAT_32

    # --- never sample or back up during the smoke test ---
    cfg.sample_after = 9999
    cfg.sample_after_unit = TimeUnit.STEP
    cfg.backup_after = 9999
    cfg.backup_after_unit = TimeUnit.STEP

    return cfg


def _embedding_vector_norm(trainer) -> float:
    emb = trainer.model.embedding.text_encoder_embedding
    return emb.vector.detach().float().norm().item()


def main() -> int:
    if not CHECKPOINT.exists():
        sys.exit(f"checkpoint not found: {CHECKPOINT}")

    # Fresh workspace every run.
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)

    print("[1] generating 4-image dataset (captions all contain the placeholder)")
    _make_tiny_dataset()

    print("[2] writing concepts / samples / config")
    concepts_path = _write_concepts()
    samples_path = _write_samples()
    cfg = _build_train_config(concepts_path, samples_path)

    print("[3] building trainer")
    callbacks = TrainCallbacks()
    commands = TrainCommands()
    trainer = create_mod.create_trainer(cfg, callbacks, commands)

    print("[4] trainer.start()  (loads model, injects placeholder token)")
    t0 = time.perf_counter()
    trainer.start()
    print(f"    start() returned in {time.perf_counter() - t0:.1f}s")

    norm_before = _embedding_vector_norm(trainer)
    print(f"    embedding norm before training: {norm_before:.6f}")

    print("[5] trainer.train()  (expecting ~40 steps for 1 epoch)")
    t0 = time.perf_counter()
    trainer.train()
    print(f"    train() returned in {time.perf_counter() - t0:.1f}s")

    norm_after = _embedding_vector_norm(trainer)
    delta = abs(norm_after - norm_before)
    print(f"    embedding norm after training:  {norm_after:.6f}  (|delta| = {delta:.6f})")

    print("[6] trainer.end()  (writes the embedding file)")
    trainer.end()

    # --- gradient-flow assertion: the token must have actually moved ---
    if not (delta > 1e-4 and torch.isfinite(torch.tensor(norm_after))):
        print(f"    FAIL: embedding vector did not move (|delta|={delta:.2e}). "
              "Gradients are not reaching the token.")
        return 3

    expected = Path(cfg.output_model_destination)
    if not expected.exists():
        for cand in expected.parent.glob(expected.name + "*"):
            print(f"    candidate output: {cand} ({cand.stat().st_size} bytes)")
        print(f"    expected file at {expected} but did not find it.")
        return 2

    size_kb = expected.stat().st_size / 1024
    print(f"OK. Token moved (|delta|={delta:.4f}) and saved embedding: {expected}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
