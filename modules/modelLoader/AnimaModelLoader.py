import os
import traceback

from modules.model.AnimaModel import AnimaModel
from modules.model.BaseModel import BaseModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
)
from transformers import (
    AutoTokenizer,
    Qwen3Model,
    T5TokenizerFast,
)


class AnimaModelLoader(
    HFModelLoaderMixin,
):
    """Load an Anima diffusers pipeline directory into an AnimaModel.

    The on-disk layout is what
    ``scripts/util/convert_anima_base_v1.py`` produces:

      <root>/
        modular_model_index.json
        transformer/            CosmosTransformer3DModel
        text_conditioner/       AnimaTextConditioner
        text_encoder/           Qwen3Model
        tokenizer/              Qwen2TokenizerFast (resolved by AutoTokenizer)
        t5_tokenizer/           T5TokenizerFast (TOKENIZER ONLY -- no model)
        vae/                    AutoencoderKLQwenImage
        scheduler/              FlowMatchEulerDiscreteScheduler

    We never instantiate an AnimaModularPipeline here. Each component
    is loaded individually so we can apply quantization and dtype
    conversions per-component the way every other OneTrainer loader
    does.
    """

    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        diffusers_sub = ["text_conditioner"]
        transformers_sub = ["text_encoder"]
        if not transformer_model_name:
            diffusers_sub.append("transformer")
        if not vae_model_name:
            diffusers_sub.append("vae")

        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=diffusers_sub,
            transformers_modules=transformers_sub,
        )

        # AutoTokenizer reads tokenizer_config.json and dispatches to the
        # right subclass (Qwen2TokenizerFast for the converted Anima dir).
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        # T5 here is *tokenizer only* -- the T5 token ids feed directly
        # into the AnimaTextConditioner's [32128, 1024] embedding table.
        # We never load a T5 encoder model.
        t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model_name,
            subfolder="t5_tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        # Qwen3Model is the base (encoder-only) variant -- no LM head,
        # so unlike Z-Image's loader we don't need the tied-weights
        # workaround for lm_head.weight = embed_tokens.weight.
        text_encoder = self._load_transformers_sub_module(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        # AnimaTextConditioner is a diffusers ModelMixin (and exposes
        # PeftAdapterMixin in case we ever want to LoRA-tune it). For
        # now it's always frozen and shares the text-encoder dtype
        # bucket because it sits at the same point in the graph.
        text_conditioner = self._load_diffusers_sub_module(
            AnimaTextConditioner,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_conditioner",
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            # Override path: a standalone Cosmos transformer .safetensors
            # (or GGUF) file. We rely on diffusers' single-file loader
            # for Cosmos; it understands the converted-from-Anima
            # key layout because it shares the architecture.
            transformer = CosmosTransformer3DModel.from_single_file(
                transformer_model_name,
                # avoid loading the transformer in float32:
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                CosmosTransformer3DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.t5_tokenizer = t5_tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.text_conditioner = text_conditioner
        model.vae = vae
        model.transformer = transformer

    def __load_safetensors(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        # The single-file Anima checkpoint released by the model authors
        # contains the Cosmos transformer state and the AnimaTextConditioner
        # state interleaved (the conditioner weights live under the
        # net.llm_adapter.* prefix). Splitting + remapping those keys is
        # what scripts/convert_anima_to_diffusers.py does; the safest
        # path for now is to ask the user to run that conversion first
        # rather than re-implement it inline.
        raise NotImplementedError(
            "Loading single-file Anima .safetensors checkpoints is not supported. "
            "Run scripts/util/convert_anima_base_v1.py (or the upstream "
            "convert_anima_to_diffusers.py) to produce a diffusers pipeline directory, "
            "then point the base model at that directory."
        )

    def load(
            self,
            model: AnimaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class AnimaLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        # LoRA target is the Cosmos transformer; weights save in the
        # diffusers convention via PeftAdapterMixin, so no key
        # remapping is needed. The upstream AnimaLoraLoaderMixin can
        # consume what we produce.
        return None

    def load(
            self,
            model: AnimaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


AnimaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.ANIMA: "resources/sd_model_spec/anima-lora.json",
    },
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    lora_loader_class=AnimaLoRALoader,
    embedding_loader_class=None,
)

AnimaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.ANIMA: "resources/sd_model_spec/anima.json",
    },
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
)
