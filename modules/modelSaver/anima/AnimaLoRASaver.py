from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin

from torch import Tensor


class AnimaLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_state_dict(
            self,
            model: AnimaModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        # Bundle the trained TI vectors alongside the adapter so the trigger token travels with the file.
        # "qwen" -- the Qwen3 word table is the only table Anima trains into. These keys survive the strict
        # convert() in the COMFY/KOHYA save paths only because AnimaModel declares lora_text_encoders().
        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_embedding.placeholder

                if embedding.text_encoder_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.qwen"] = embedding.text_encoder_embedding.vector

        return state_dict
