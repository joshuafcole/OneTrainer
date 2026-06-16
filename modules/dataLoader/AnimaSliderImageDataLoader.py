"""Coordinate-labeled image-slider data loader for Anima (docs §10).

Unlike the prompt-pair slider (which has no dataset), the coordinate-labeled
image slider trains on a *real* image dataset: vanilla OneTrainer concepts whose
captions carry a declared-axis coordinate token, e.g. ``(distance:-2)``. So this
reuses the full AnimaBaseDataLoader MGDS pipeline (VAE latents + Qwen3/T5
conditioning, caching, aspect bucketing) and only adds one thing: a pair of nodes
that, right after the per-image caption is selected and *before* tag-dropout /
tokenization, extract the declared-axis coordinates out of the caption.

  * ``slider_coordinate``: a (1,) float tensor holding the *target* axis's raw
    coordinate for the image (0.0 if absent). The per-axis gain ``k`` is applied
    later, at training step time (``m = k * coordinate``), so the gain can be
    retuned without rebuilding the latent/text cache.
  * the caption (``prompt``) with every declared-axis token removed, so the
    conditioning stays orthogonal to the axis (the one load-bearing slider
    constraint, docs §10.0). Ordinary a1111 emphasis tokens are left untouched.

This loader does NOT register itself for (ANIMA, SLIDER); AnimaSliderDataLoader
owns that slot and dispatches to this class when the regime is IMAGE.
"""

from modules.dataLoader.AnimaBaseDataLoader import AnimaBaseDataLoader
from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.BaseAnimaSetup import BaseAnimaSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.slider_caption_util import parse_slider_coordinates

import torch

from mgds.pipelineModules.MapData import MapData


def _resolve_axes(config: TrainConfig) -> tuple[list[str], str | None]:
    """(declared axis names to strip, target axis name) from the enabled axes.

    The loader only needs which names to strip and which one supplies the
    coordinate value; gain / validation live in the setup. Returns target=None
    when no enabled axis is flagged is_target (the setup raises a clear error).
    """
    enabled = [a for a in config.slider_axes if a.enabled]
    declared = [a.name.strip() for a in enabled if a.name and a.name.strip()]
    targets = [a.name.strip() for a in enabled if a.is_target and a.name and a.name.strip()]
    target = targets[0].lower() if targets else None
    return declared, target


class AnimaSliderImageDataLoader(AnimaBaseDataLoader):
    def _load_input_modules(
        self,
        config: TrainConfig,
        train_dtype: DataType,
        vae_frame_dim: bool = False,
    ) -> list:
        modules = super()._load_input_modules(config, train_dtype, vae_frame_dim)

        declared, target = _resolve_axes(config)

        def to_coordinate(prompt: str) -> torch.Tensor:
            _, coords = parse_slider_coordinates(prompt, declared)
            value = coords.get(target, 0.0) if target is not None else 0.0
            return torch.tensor([value], dtype=torch.float32)

        def strip_coordinates(prompt: str) -> str:
            cleaned, _ = parse_slider_coordinates(prompt, declared)
            return cleaned

        # Order matters: extract the coordinate from the *original* caption first,
        # then strip the axis tokens out of the caption. Both run before the
        # augmentation (tag-dropout) and preparation (tokenize) stages.
        extract_coordinate = MapData(in_name="prompt", out_name="slider_coordinate", map_fn=to_coordinate)
        strip_axis_tokens = MapData(in_name="prompt", out_name="prompt", map_fn=strip_coordinates)
        modules.extend([extract_coordinate, strip_axis_tokens])
        return modules

    def _cache_modules(self, config: TrainConfig, model: AnimaModel, model_setup: BaseAnimaSetup):
        # Mirrors AnimaBaseDataLoader._cache_modules with slider_coordinate added
        # to the per-sample (split) set so it survives latent/text caching.
        image_split_names = ["latent_image", "original_resolution", "crop_offset", "slider_coordinate"]
        image_aggregate_names = ["crop_resolution", "image_path"]
        text_split_names = []
        if not config.train_text_encoder_or_embedding():
            text_split_names = [
                "tokens_qwen",
                "tokens_mask_qwen",
                "tokens_t5",
                "tokens_mask_t5",
                "text_encoder_hidden_state",
            ]
        sort_names = (
            image_aggregate_names
            + image_split_names
            + [
                "prompt",
                "tokens_qwen",
                "tokens_mask_qwen",
                "tokens_t5",
                "tokens_mask_t5",
                "text_encoder_hidden_state",
                "concept",
            ]
        )

        return self._cache_modules_from_names(
            model,
            model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=not config.train_text_encoder_or_embedding(),
        )

    def _output_modules(self, config: TrainConfig, model: AnimaModel, model_setup: BaseAnimaSetup):
        output_names = [
            "image_path",
            "latent_image",
            "slider_coordinate",
            "prompt",
            "tokens_qwen",
            "tokens_mask_qwen",
            "tokens_t5",
            "tokens_mask_t5",
            "original_resolution",
            "crop_resolution",
            "crop_offset",
        ]
        if not config.train_text_encoder_or_embedding():
            output_names.append("text_encoder_hidden_state")

        return self._output_modules_from_out_names(
            model,
            model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )
