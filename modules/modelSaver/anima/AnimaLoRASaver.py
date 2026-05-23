from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor


class AnimaLoRASaver(
    LoRASaverMixin,
):
    """Save Anima LoRA weights.

    The LoRA target is exclusively the Cosmos transformer. The
    upstream rmatif/diffusers PR adds an AnimaLoraLoaderMixin
    (loaders/lora_pipeline.py) that knows how to load weights in
    diffusers' standard PeftAdapterMixin convention, so we save the
    state dict directly without any custom key-remapping pass.

    For ComfyUI compatibility, the saved file uses diffusers naming
    (``transformer.transformer_blocks.<i>.attn1.to_q.lora_up.weight``)
    which ComfyUI's stock LoraLoader will NOT match against an Anima
    model loaded via UNETLoader (that one uses native naming
    ``net.blocks.<i>.self_attn.q_proj.weight``). Run
    ``scripts/util/convert_anima_lora_diffusers_to_native.py`` on the
    saved file to produce a ComfyUI-loadable variant.
    """

    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: AnimaModel) -> list[LoraConversionKeySet] | None:
        return None

    def _get_state_dict(
        self,
        model: AnimaModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        # Bundle TI embeddings into the LoRA file so inference tools
        # (ComfyUI etc.) get the trigger token alongside the adapter.
        # Anima injects into both the Qwen3 word-embedding table ("qwen")
        # and the T5 input-embedding table inside AnimaTextConditioner ("t5").
        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_embedding.placeholder

                if embedding.text_encoder_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.qwen"] = embedding.text_encoder_embedding.vector
                if embedding.text_encoder_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.qwen_out"] = embedding.text_encoder_embedding.output_vector
                if embedding.t5_embedding is not None and embedding.t5_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.t5"] = embedding.t5_embedding.vector

        return state_dict

    def save(
        self,
        model: AnimaModel,
        output_model_format: ModelFormat,
        output_model_destination: str,
        dtype: torch.dtype | None,
    ):
        self._save(model, output_model_format, output_model_destination, dtype)
