"""Save a full Anima diffusers pipeline (or transformer-only safetensors).

For LoRA training this code path is unused -- AnimaLoRASaver handles
the only training-time persistence we care about. The fine-tune saver
exists so the factory wiring is symmetric with Z-Image / Flux and so
a future full-fine-tune workflow has a saver to call.
"""

import copy
import os.path
from pathlib import Path

from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch

from safetensors.torch import save_file


class AnimaModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: AnimaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # AnimaModularPipeline.save_pretrained writes the
        # modular_model_index.json + per-component subfolders that
        # AnimaModelLoader (and the upstream
        # ModularPipeline.from_pretrained) can re-load.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        if dtype is not None:
            # Tokenizers' __deepcopy__ tries to reload from disk; pin it
            # to a no-op so deepcopy keeps the in-memory instance.
            for tok in (pipeline.tokenizer, pipeline.t5_tokenizer):
                tok.__deepcopy__ = lambda memo, _t=tok: _t

            save_pipeline = copy.deepcopy(pipeline)
            save_pipeline.to(device="cpu", dtype=dtype)

            for tok in (pipeline.tokenizer, pipeline.t5_tokenizer):
                delattr(tok, '__deepcopy__')
        else:
            save_pipeline = pipeline

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: AnimaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Transformer-only safetensors: useful for ComfyUI consumption
        # of a fine-tuned Cosmos DiT without redistributing the whole
        # 6 GB pipeline directory.
        state_dict = model.transformer.state_dict()
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: AnimaModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: AnimaModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
