"""Coordinate-labeled image-slider data loader for Anima.

The point of this regime is that there is no new dataset format. The concepts,
the captions, the aspect bucketing, the latent and text caches are all Anima's
ordinary pipeline -- so this subclasses AnimaBaseDataLoader and adds exactly two
nodes to it:

  * ``slider_coordinate``: a (1,) float tensor holding the target axis's raw
    coordinate for this image, or 0.0 if the caption did not carry one. The gain
    is NOT applied here. It is applied at step time, so retuning it does not
    invalidate a latent cache that can take an hour to rebuild.
  * ``prompt``, with every declared axis token removed -- the load-bearing part.
    A caption that still said "(distance:-2)" would let the base read the
    attribute straight off the prompt, and the adapter would have nothing left to
    explain.

Both run in the input stage: after the per-image caption has been selected, and
before augmentation and tokenization. The ordering matters in both directions --
the coordinate has to be read from the caption as authored, and tag dropout must
not get a chance to delete a coordinate token before it is stripped deliberately.

This loader does NOT register itself for (ANIMA, SLIDER). SliderDataLoader owns
that slot and dispatches here when the regime is IMAGE.
"""

from modules.dataLoader.AnimaBaseDataLoader import AnimaBaseDataLoader
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.slider_caption_util import (
    declared_axis_names,
    parse_slider_coordinates,
    resolve_target_axis,
)

from mgds.pipelineModules.MapData import MapData

import torch

SLIDER_COORDINATE = 'slider_coordinate'


class AnimaSliderImageDataLoader(AnimaBaseDataLoader):
    def _load_input_modules(
            self,
            config: TrainConfig,
            train_dtype: DataType,
            vae_frame_dim: bool = False,
    ) -> list:
        modules = super()._load_input_modules(config, train_dtype, vae_frame_dim)

        # Resolved once, here, rather than per caption: an unusable axis set is a
        # config mistake, and failing while the dataset is being wired is far
        # earlier -- and far clearer -- than failing per image inside MGDS.
        declared = declared_axis_names(config.slider_axes)
        target = resolve_target_axis(config.slider_axes).name.strip().lower()

        def to_coordinate(prompt: str) -> torch.Tensor:
            _, coords = parse_slider_coordinates(prompt, declared)
            return torch.tensor([coords.get(target, 0.0)], dtype=torch.float32)

        def strip_coordinates(prompt: str) -> str:
            cleaned, _ = parse_slider_coordinates(prompt, declared)
            return cleaned

        # Read first, then strip: the second node would otherwise hand the first
        # a caption with the coordinate already gone.
        modules.append(MapData(in_name='prompt', out_name=SLIDER_COORDINATE, map_fn=to_coordinate))
        modules.append(MapData(in_name='prompt', out_name='prompt', map_fn=strip_coordinates))
        return modules

    def _additional_split_names(self, config: TrainConfig) -> list[str]:
        # One value per image, so it belongs in the per-sample split and has to
        # survive latent caching; the base class threads it into the sort and the
        # output names from here.
        return [SLIDER_COORDINATE]
