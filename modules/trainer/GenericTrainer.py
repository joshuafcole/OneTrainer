import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import traceback
from collections.abc import Callable
from pathlib import Path

import modules.util.multi_gpu_util as multi
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.bf16_stochastic_rounding import set_seed as bf16_stochastic_rounding_set_seed
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.LokrInitMode import LokrInitMode
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import PeftType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.grad_estimation import WeightGradientEstimator
from modules.util.loss.counterexample_loss import SCHEDULE as counterexample_schedule
from modules.util.loss.counterexample_loss import TELEMETRY as counterexample_telemetry
from modules.util.profiling_util import TorchMemoryRecorder, TorchProfiler
from modules.util.sample_metadata import SampleProvenance, hash_text
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

from mgds.perf_probe import perf

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

import huggingface_hub
from requests.exceptions import ConnectionError
from tqdm import tqdm


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        if multi.is_master():
            tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
            os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
            self.tensorboard = SummaryWriter(
                os.path.join(tensorboard_log_dir, f"{config.save_filename_prefix}{get_string_timestamp()}")
            )
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []
        self.sampled_train_progresses: set[str] = set()
        self.saved_train_progresses: set[str] = set()
        self.train_exited_cleanly = False
        # basename of the most recent *successful* save this run, for sample
        # provenance; None until the first save lands.
        self.last_save_filename: str | None = None

    def start(self):
        # Both are process-global singletons, and a process can train more than
        # once: the GUI runs the trainer on a thread and keeps the process alive
        # across Start presses. Without this, run 2 inherits run 1's FROZEN beta
        # -- calibrated against a different model, resolution, or band -- and
        # says nothing about it, because a beta carried over is indistinguishable
        # from one configured. That is exactly the back-to-back shape an A/B
        # bake-off has.
        counterexample_schedule.reset()
        counterexample_telemetry.reset()

        if multi.is_master():
            self.__save_config_to_workspace()

            if self.config.clear_cache_before_training and self.config.latent_caching:
                self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("logging into Hugging Face")
            with contextlib.suppress(ConnectionError):
                huggingface_hub.login(
                    token=self.config.secrets.huggingface_token,
                    new_session=False,
                )

        self.callbacks.on_update_status("loading the model")

        if self.config.quantization.cache_dir is None:
            self.config.quantization.cache_dir = self.config.cache_dir + "/quantization"
        os.makedirs(self.config.quantization.cache_dir, exist_ok=True)

        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
            quantization=self.config.quantization,
        )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(self.model, self.model_setup, self.model.train_progress)
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()

        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True
            )

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{self.config.save_filename_prefix}{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f"Clearing cache directory {self.config.cache_dir}! "
            f"You can disable this if you want to continue using the same cache."
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith("epoch-") or filename in ["image", "text"]):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [
                    dirpath
                    for dirpath in os.listdir(backup_dirpath)
                    if os.path.isdir(os.path.join(backup_dirpath, dirpath))
                ],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    print(f"Could not delete old rolling backup {dirpath}")

        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    @staticmethod
    def __progress_key(train_progress: TrainProgress) -> str:
        return train_progress.filename_string()

    def __mark_clean_train_exit(self):
        self.train_exited_cleanly = True

    def __has_enabled_samples(self, sample_config_list: list[SampleConfig]) -> bool:
        return any(sample_config.enabled for sample_config in sample_config_list)

    def __should_emit_final_workspace_artifacts(self) -> bool:
        return self.one_step_trained and self.train_exited_cleanly

    def __emit_final_workspace_artifacts(self, train_progress: TrainProgress):
        progress_key = self.__progress_key(train_progress)

        if self.config.save_on_train_end and multi.is_master() and progress_key not in self.saved_train_progresses:
            self.__save(train_progress)

        if self.config.sample_on_train_end and multi.is_master() and progress_key not in self.sampled_train_progresses:
            self.__sample_during_training(train_progress, self.train_device, distribute=False)

    def __emit_scheduled_for_empty_epoch(self, train_progress: TrainProgress, train_device: torch.device):
        """Honor the periodic sample/save schedule for an epoch that produced no step.

        The per-batch loop owns the periodic sample/save checks (they fire at the
        first step of a boundary epoch), so an epoch that yields zero training steps
        -- e.g. a sparse aspect bucket starved by drop-last -- would skip its
        scheduled sample/save entirely, dropping that column from the monitor. The
        schedule predicates already require ``epoch_step == 0``, so this is a no-op
        for any epoch that ran at least one step; it only fills the gap for a starved
        boundary epoch. Deduped against an in-loop emit at this same progress via the
        §177 tracking sets, so a normal epoch never double-emits."""
        progress_key = self.__progress_key(train_progress)

        if self.__needs_sample(train_progress) and progress_key not in self.sampled_train_progresses:
            self.__enqueue_sample_during_training(
                lambda: self.__sample_during_training(train_progress, train_device)
            )
            self.__execute_sample_during_training()

        if (
            multi.is_master()
            and self.__needs_save(train_progress)
            and progress_key not in self.saved_train_progresses
        ):
            self.model.to(self.temp_device)
            self.__save(train_progress, True, print)
            self.model_setup.setup_train_device(self.model, self.config)

    def __sample_loop(
        self,
        train_progress: TrainProgress,
        train_device: torch.device,
        sample_config_list: list[SampleConfig],
        ema_applied: bool,
        distribute: bool = True,
        folder_postfix: str = "",
        is_custom_sample: bool = False,
    ):
        for i, sample_config in multi.distributed(
            [(i, sample_config) for i, sample_config in enumerate(sample_config_list) if sample_config.enabled],
            distribute=distribute and not self.config.samples_to_tensorboard and not ema_applied,
        ):
            try:
                safe_prompt = path_util.safe_filename(sample_config.prompt)

                if is_custom_sample:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        "custom",
                    )
                else:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        f"{str(i)} - {safe_prompt}{folder_postfix}",
                    )

                sample_path = os.path.join(
                    sample_dir,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}-training-sample-{train_progress.filename_string()}",
                )

                def on_sample_default(sampler_output: ModelSamplerOutput):
                    if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                        self.tensorboard.add_image(
                            f"sample{str(i)} - {safe_prompt}",
                            pil_to_tensor(sampler_output.data),  # noqa: B023
                            train_progress.global_step,
                        )
                    self.callbacks.on_sample_default(sampler_output)

                def on_sample_custom(sampler_output: ModelSamplerOutput):
                    self.callbacks.on_sample_custom(sampler_output)

                on_sample = on_sample_custom if is_custom_sample else on_sample_default
                on_update_progress = (
                    self.callbacks.on_update_sample_custom_progress
                    if is_custom_sample
                    else self.callbacks.on_update_sample_default_progress
                )

                self.model.to(self.temp_device)
                self.model.eval()

                sample_config = copy.copy(sample_config)
                sample_config.from_train_config(self.config)

                # Provenance describes the *normalized* config actually sampled,
                # not the raw list entry -- hashed after from_train_config above.
                # A failure here must not cost the run its sample, so it only
                # ever leaves provenance unset, never skips model_sampler.sample.
                try:
                    self.model_sampler.set_provenance(SampleProvenance(
                        global_step=train_progress.global_step,
                        epoch=train_progress.epoch,
                        epoch_step=train_progress.epoch_step,
                        seed=None if sample_config.random_seed else sample_config.seed,
                        prompt_hash=hash_text(sample_config.prompt),
                        sample_config_hash=hash_text(json.dumps(
                            sample_config.to_dict(), sort_keys=True, ensure_ascii=True,
                            separators=(",", ":"), default=str,
                        )),
                        last_save_filename=self.last_save_filename,
                    ))
                except Exception:
                    traceback.print_exc()
                    print("Could not build sample provenance, sampling without it")
                    self.model_sampler.set_provenance(None)

                self.model_sampler.sample(
                    sample_config=sample_config,
                    destination=sample_path,
                    image_format=self.config.sample_image_format,
                    video_format=self.config.sample_video_format,
                    audio_format=self.config.sample_audio_format,
                    on_sample=on_sample,
                    on_update_progress=on_update_progress,
                )
            except Exception:
                traceback.print_exc()
                print("Error during sampling, proceeding without sampling")

            torch_gc()

    def __sample_during_training(
        self,
        train_progress: TrainProgress,
        train_device: torch.device,
        sample_params_list: list[SampleConfig] = None,
        distribute: bool = True,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("Sampling ...")

        is_custom_sample = False
        if sample_params_list:
            is_custom_sample = True
        elif self.config.samples is not None:
            sample_params_list = self.config.samples
        else:
            try:
                with open(self.config.sample_definition_file_name, "r") as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values(self.config.model_type).from_dict(samples[i])
                    sample_params_list = samples
            # We absolutely do not want to fail training just because the sample definition file becomes missing or broken right before sampling.
            except Exception:
                traceback.print_exc()
                print("Error during loading the sample definition file, proceeding without sampling")
                sample_params_list = []

        has_enabled_samples = self.__has_enabled_samples(sample_params_list)

        if self.model.ema:
            # the EMA model only exists in the master process, so EMA sampling is done on one GPU only
            # non-EMA sampling is done on all GPUs
            assert multi.is_master() and self.config.ema != EMAMode.OFF
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            distribute=distribute,
            is_custom_sample=is_custom_sample,
            ema_applied=self.config.ema != EMAMode.OFF,
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if ema is enabled:
        if self.config.ema != EMAMode.OFF and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                distribute=distribute,
                folder_postfix=" - no-ema",
                ema_applied=False,
            )

        if has_enabled_samples and not is_custom_sample and multi.is_master():
            self.sampled_train_progresses.add(self.__progress_key(train_progress))

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("Calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation,
            )

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model, validation_batch, self.config, train_progress, deterministic=True
                    )
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config
                    )

                # since validation batch size = 1
                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                # check and fix collision to display both graphs in tensorboard
                if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(
                    f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                    average_loss,
                    train_progress.global_step,
                )

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar(
                    "loss/validation_step/total_average", total_average_loss, train_progress.global_step
                )

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def __backup(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("Creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __save(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("Saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}",
        )
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype(),
            )
            if multi.is_master():
                self.saved_train_progresses.add(self.__progress_key(train_progress))
                self.last_save_filename = os.path.basename(save_path)
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first", self.config.sample_skip_first, self.config.sample_after_unit, train_progress
        ) and self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(self, scaler):
        fused_optimizer_step = (
            self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass
        )
        fused_reduce = self.config.multi_gpu and self.config.fused_gradient_reduce
        if fused_optimizer_step:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating Fused Back Pass with Accumulation Steps > 1 does not reduce VRAM usage.")
            if self.config.multi_gpu and not fused_reduce:
                raise ValueError("if Fused Back Pass and Multi-GPU is enabled, Fused Reduce must also be enabled")
        elif not fused_reduce:
            return

        for param_group in self.model.optimizer.param_groups:
            for i, parameter in enumerate(param_group["params"]):
                # TODO: Find a better check instead of "parameter.requires_grad".
                #       This will break if the some parameters don't require grad during the first training step.
                if parameter.requires_grad:
                    if scaler:

                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            scaler.unscale_parameter_(tensor, self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                            tensor.grad = None
                    else:

                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            self.model.optimizer.step_parameter(tensor, param_group, i)
                            tensor.grad = None

                    def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                        if self.__is_update_step(self.model.train_progress):
                            if fused_reduce:
                                multi.reduce_grads_mean(
                                    [tensor],
                                    self.config.gradient_reduce_precision,
                                    after_reduce=__optimizer_step if fused_optimizer_step else None,
                                    async_op=self.config.async_gradient_reduce,
                                    max_buffer=self.config.async_gradient_reduce_buffer * 1024 * 1024,
                                )
                            elif fused_optimizer_step:
                                __optimizer_step(tensor)

                    handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                    self.grad_hook_handles.append(handle)

    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    def __lokr_init_cache_path(self) -> str | None:
        """Cache file for the estimated GA Van Loan factors.

        The factors depend on the base model, the dataset, the estimation
        length and the Kronecker factorization — but NOT on LoKr rank (dim),
        alpha or gain, so one estimation pass serves a whole LoKr sweep. The
        key also carries peft_type so a LoRA-GA cache (1-tuple right-singular
        matrices) never collides with a GA cache (2-tuple Van Loan pairs);
        the LoRA-GA factors are truncated to rank on replay, so lora_rank is
        included only for LoRA to keep the cached matrices wide enough.
        Lives under cache_dir next to the latent cache (same reuse semantics).
        """
        config = self.config
        if not config.cache_dir:
            return None
        key_data = {
            "base_model_name": config.base_model_name,
            "concept_file_name": config.concept_file_name,
            "lokr_init_steps": config.lokr_init_steps,
            "lokr_decompose_factor": config.lokr_decompose_factor,
            "peft_type": str(config.peft_type),
        }
        if config.peft_type == PeftType.LORA:
            key_data["lora_rank"] = config.lora_rank
        key = json.dumps(key_data, sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return os.path.join(config.cache_dir, "kron_ga", f"{digest}.pt")

    def __run_lokr_gradient_init(self):
        """GA: gradient-aligned LoKr initialization (LoRA-GA ported to LoKr).

        Estimates per-layer dL/dW of the frozen base weights over the first
        lokr_init_steps batches, then re-initializes the nonzero LoKr factors
        with the principal Kronecker factors of the estimated gradient. The
        zero factor is untouched, so the model output is unchanged until the
        first real optimizer step.
        """
        config = self.config
        if config.lokr_init_mode != LokrInitMode.GRADIENT or config.peft_type not in (PeftType.LOKR, PeftType.LORA):
            return
        if config.training_method not in (TrainingMethod.LORA, TrainingMethod.SLIDER):
            return
        if self.model.train_progress.global_step > 0:
            print("GA init: skipping, training is being resumed.")
            return
        if config.model_names().lora:
            print("GA init: skipping, an existing LoKr checkpoint is being loaded.")
            return
        if multi.world_size() > 1:
            print("GA init: skipping, multi-GPU training is not supported yet.")
            return

        wrappers = [
            module for module in vars(self.model).values()
            if isinstance(module, LoRAModuleWrapper)
        ]
        if not wrappers:
            return

        cache_path = self.__lokr_init_cache_path()
        if cache_path is not None and os.path.isfile(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"GA init: ignoring unreadable cache {cache_path}: {e}")
                cached = None
            if cached is not None:
                self.callbacks.on_update_status("GA factor initialization")
                applied = skipped = 0
                for wrapper in wrappers:
                    factors = {
                        name: pair for name, pair in
                        ((name, cached.get(f"{wrapper.prefix}.{name}")) for name in wrapper.lora_modules)
                        if pair is not None
                    }
                    wrapper_applied, wrapper_skipped = wrapper.init_lokr_from_factors(
                        factors, config.lokr_init_gain)
                    applied += wrapper_applied
                    skipped += wrapper_skipped
                print(f"GA init: applied to {applied} layers ({skipped} skipped) from cache {cache_path}")
                return

        self.callbacks.on_update_status("GA gradient estimation")
        self.callbacks.on_update_aux_progress("kron-ga", 0, config.lokr_init_steps)

        # The fp32 accumulators are weight-shaped, one per adapted Linear:
        # accumulating on the train device avoids a per-layer device-to-host
        # sync every batch, at the cost of that VRAM; offload trades it back.
        store_device = torch.device("cpu") if config.lokr_init_offload else torch.device(config.train_device)

        estimators = []
        for wrapper in wrappers:
            estimator = WeightGradientEstimator(store_device=store_device)
            estimator.attach({name: module.orig_module for name, module in wrapper.lora_modules.items()})
            estimators.append(estimator)

        # Keeps fp16 gradients from underflowing without a GradScaler. The init
        # only uses gradient directions, so the constant has no other effect.
        loss_scale = 1024.0 if enable_grad_scaling(config.train_dtype, self.parameters) else 1.0

        if config.latent_caching:
            self.data_loader.get_data_set().start_next_epoch()
            self.model_setup.setup_train_device(self.model, config)
        else:
            self.model_setup.setup_train_device(self.model, config)
            self.data_loader.get_data_set().start_next_epoch()

        # An advancing copy so timestep/noise sampling varies across batches,
        # without moving the real training progress.
        progress = copy.deepcopy(self.model.train_progress)
        step_count = 0
        # force_eager: dynamo hard-errors on tensor hook registration inside
        # compiled regions, so compiled blocks must run eagerly during the
        # estimation pass. Compiled training resumes normally afterwards.
        with torch.compiler.set_stance("force_eager"):
            for batch in tqdm(self.data_loader.get_data_loader(), desc="kron-ga", total=config.lokr_init_steps):
                model_output_data = self.model_setup.predict(self.model, batch, config, progress)

                # Exclude prior-prediction (regularization) samples: their true
                # step-0 gradient is ~0, because the adapter still outputs zero,
                # so the trained model *is* the prior model. Detaching the
                # prediction as their target zeroes their loss without running
                # the prior model.
                #
                # Counterexamples are excluded for the same reason and by the same
                # trick: at step 0 their true gradient is the half-scale repulsion
                # (delta == 0 exactly, because the adapter is still zero), which
                # would pollute the gradient-alignment estimate with a term that
                # points away from the data. Opting out of the repulsion here is
                # explicit -- the loss raises rather than silently training the
                # counterexample as a positive.
                inert_indices = [
                    i
                    for i in range(config.batch_size)
                    if ConceptType(batch["concept_type"][i])
                    in (ConceptType.PRIOR_PREDICTION, ConceptType.COUNTEREXAMPLE)
                ]
                model_output_data["skip_counterexample_repulsion"] = True
                if len(inert_indices) > 0:
                    predicted_detached = model_output_data["predicted"].detach().to(
                        dtype=model_output_data["target"].dtype
                    )
                    model_output_data["target"][inert_indices] = predicted_detached[inert_indices]

                loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, config)
                (loss * loss_scale).backward()
                for estimator in estimators:
                    estimator.count_step()
                progress.next_step(config.batch_size)
                step_count += 1
                self.callbacks.on_update_aux_progress("kron-ga", step_count, config.lokr_init_steps)
                if step_count >= config.lokr_init_steps:
                    break

        self.callbacks.on_update_status("GA factor initialization")
        applied = skipped = 0
        all_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for wrapper, estimator in zip(wrappers, estimators, strict=True):
            estimator.detach_hooks()
            grads = {
                name: grad for name in wrapper.lora_modules
                if (grad := estimator.mean_gradient(name)) is not None
            }
            wrapper_applied, wrapper_skipped, factors = \
                wrapper.init_lokr_from_gradients(grads, config.lokr_init_gain)
            applied += wrapper_applied
            skipped += wrapper_skipped
            for name, pair in factors.items():
                all_factors[f"{wrapper.prefix}.{name}"] = pair
            estimator.clear()

        if cache_path is not None and all_factors:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(all_factors, cache_path)
                print(f"GA init: cached Van Loan factors to {cache_path}")
            except OSError as e:
                print(f"GA init: failed to write cache {cache_path}: {e}")

        self.model.optimizer.zero_grad(set_to_none=True)
        torch_gc()
        self.callbacks.on_update_aux_progress("kron-ga", 0, 0)
        print(f"GA init: applied to {applied} layers ({skipped} skipped) from {step_count} batches.")

    def train(self):
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress
        self.train_exited_cleanly = False

        if self.config.only_cache:
            if multi.is_master():
                self.callbacks.on_update_status("Caching")
                for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
                    self.data_loader.get_data_set().start_next_epoch()
            self.__mark_clean_train_exit()
            return

        self.__run_lokr_gradient_init()

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = torch.tensor(0.0, device=train_device)
        ema_loss = None
        ema_loss_steps = 0
        epochs = range(train_progress.epoch, self.config.epochs, 1)

        for _epoch in tqdm(epochs, desc="epoch") if multi.is_master() else epochs:
            multi.sync_commands(self.commands)
            if self.commands.get_stop_command():
                self.__mark_clean_train_exit()
                return
            self.callbacks.on_update_status("Starting epoch/caching")

            # call start_next_epoch with only one process at first, because it might write to the cache. All subsequent processes can read in parallel:
            for _ in multi.master_first():
                if self.config.latent_caching:
                    self.data_loader.get_data_set().start_next_epoch()
                    self.model_setup.setup_train_device(self.model, self.config)
                else:
                    self.model_setup.setup_train_device(self.model, self.config)
                    self.data_loader.get_data_set().start_next_epoch()

            if self.config.debug_mode:
                multi.warn_parameter_divergence(self.parameters, train_device)

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step,
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()
            # Same arithmetic create_lr_scheduler uses for `total_steps`, so the
            # counterexample ramp and the LR schedule agree on where "the end" is
            # -- the whole point of the fraction form is to land full strength
            # while the LR is annealing.
            counterexample_total_steps = int(
                current_epoch_length * self.config.epochs / self.config.gradient_accumulation_steps
            )

            if multi.is_master():
                batches = step_tqdm = tqdm(
                    self.data_loader.get_data_loader(),
                    desc="step",
                    total=current_epoch_length,
                    initial=train_progress.epoch_step,
                )
            else:
                batches = self.data_loader.get_data_loader()
            for batch in batches:
                multi.sync_commands(self.commands)
                if self.commands.get_stop_command():
                    multi.warn_parameter_divergence(self.parameters, train_device)

                if (
                    not self.commands.get_stop_command()
                    and self.__needs_sample(train_progress)
                    or self.commands.get_and_reset_sample_default_command()
                ):
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )
                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:

                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    backup = self.commands.get_and_reset_backup_command()
                    save = self.commands.get_and_reset_save_command()
                    if multi.is_master() and (backup or save):
                        self.model.to(self.temp_device)
                        if backup:
                            self.__backup(train_progress, True, step_tqdm.write)
                        if save:
                            self.__save(train_progress, True, step_tqdm.write)
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("Training ...")

                perf.step_begin(train_progress.global_step)
                # latent token count (~Cosmos self-attn seq length, = h_lat*w_lat) so the
                # profiler can fire on the first high-res / VRAM-saturating step rather than a
                # fixed index (see OT_PROFILE_MIN_TOKENS). Best-effort; None if shape is absent.
                _lat = batch.get("latent_image") if isinstance(batch, dict) else None
                _profile_tokens = int(_lat.shape[-2] * _lat.shape[-1]) if _lat is not None and _lat.ndim >= 2 else None
                with (
                    TorchMemoryRecorder(enabled=False, filename=f"memory-step{train_progress.global_step}.pickle"),
                    TorchProfiler(
                        enabled=perf.should_profile(train_progress.global_step, tokens=_profile_tokens),
                        filename=f"profile-step{train_progress.global_step}.json",
                    ),
                ):
                    step_seed = train_progress.global_step
                    bf16_stochastic_rounding_set_seed(step_seed, train_device)

                    prior_pred_indices = [
                        i
                        for i in range(self.config.batch_size)
                        if ConceptType(batch["concept_type"][i]) == ConceptType.PRIOR_PREDICTION
                    ]
                    # A counterexample row needs the same frozen forward, but NOT the
                    # target substitution below: its target stays the wrong image it
                    # actually is, and the loss measures how much better than the
                    # reference the adapter has learned to reproduce it.
                    has_counterexample = any(
                        ConceptType(batch["concept_type"][i]) == ConceptType.COUNTEREXAMPLE
                        for i in range(self.config.batch_size)
                    )
                    if len(prior_pred_indices) > 0 or has_counterexample or (
                        self.config.masked_training
                        and self.config.masked_prior_preservation_weight > 0
                        and self.config.training_method == TrainingMethod.LORA
                    ):
                        with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                            # do NOT create a subbatch using the indices, even though it would be more efficient:
                            # different timesteps are used for a smaller subbatch by predict(), but the conditioning must match exactly:
                            prior_model_output_data = self.model_setup.predict(
                                self.model, batch, self.config, train_progress
                            )
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                        prior_model_prediction = prior_model_output_data["predicted"].to(
                            dtype=model_output_data["target"].dtype
                        )
                        model_output_data["target"][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
                        model_output_data["prior_target"] = prior_model_prediction
                        # The counterexample ramp and its beta calibration need a
                        # clock. `calculate_loss` takes no TrainProgress, so it
                        # rides the same dict `prior_target` does.
                        model_output_data["counterexample_step"] = train_progress.global_step
                        model_output_data["counterexample_total_steps"] = counterexample_total_steps
                    else:
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                    loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)

                    loss = loss / self.config.gradient_accumulation_steps
                    perf.tic("backward")
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    perf.toc("backward")

                    has_gradient = True
                    detached_loss = loss.detach()
                    multi.reduce_tensor_mean(detached_loss)
                    accumulated_loss += detached_loss

                    if self.__is_update_step(train_progress):
                        # Drained on every rank, not just the logging one, so a
                        # non-master's accumulator never carries a previous
                        # window's rows into the next.
                        window = counterexample_telemetry.take()
                        counterexample = window.stats
                        perf.tic("optimizer")
                        if self.config.fused_gradient_reduce:
                            multi.finish_async(self.config.gradient_reduce_precision)
                        else:
                            multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)

                        if (
                            scaler
                            and self.config.optimizer.optimizer.supports_fused_back_pass()
                            and self.config.optimizer.fused_back_pass
                        ):
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()

                        lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                        self.model.optimizer.zero_grad(set_to_none=True)
                        has_gradient = False
                        perf.toc("optimizer")

                        if multi.is_master():
                            self.model_setup.report_to_tensorboard(
                                self.model, self.config, lr_scheduler, self.tensorboard
                            )

                            accumulated_loss_cpu = accumulated_loss.item()
                            if math.isnan(accumulated_loss_cpu):
                                raise RuntimeError(
                                    "Training loss became NaN. This may be due to invalid parameters, precision issues, or a bug in the loss computation."
                                )

                            self.tensorboard.add_scalar(
                                "loss/train_step", accumulated_loss_cpu, train_progress.global_step
                            )

                            # Counterexample readout (phase 502). Drained once per
                            # optimizer step, so it aggregates the whole GA window.
                            # `gate_mean` is the one to read first: it is the mean
                            # per-row multiplier on the repulsion gradient, so ~0
                            # means the term is inert -- which a warm start makes
                            # entirely possible, because `prior_model()` detaches
                            # *every* adapter and the reference is therefore the
                            # foundation, not the run this one resumed from.
                            if counterexample.rows > 0:
                                for tag, value in (
                                    ("counterexample/rows", float(counterexample.rows)),
                                    ("counterexample/delta_mean", counterexample.delta_mean),
                                    ("counterexample/gate_mean", counterexample.gate_mean),
                                    ("counterexample/saturated_fraction", counterexample.saturated_fraction),
                                    ("counterexample/loss_mean", counterexample.loss_mean),
                                    # The ramp's current strength, and the beta in
                                    # force -- which is not necessarily the
                                    # configured one, since `counterexample_beta = 0`
                                    # means "solve it from this run's delta".
                                    ("counterexample/weight", window.weight),
                                    ("counterexample/beta", window.beta),
                                    # Where on the schedule the term operated,
                                    # and how much of it the noise band let
                                    # through. band_pass is the DOSE: a band
                                    # passing 0.4 delivers 40% of the repulsion,
                                    # which an A/B has to match across arms.
                                    ("counterexample/noise_level", counterexample.noise_level_mean),
                                    ("counterexample/band_pass", counterexample.band_mean),
                                ):
                                    self.tensorboard.add_scalar(tag, value, train_progress.global_step)
                            ema_loss = ema_loss or accumulated_loss_cpu
                            ema_loss_steps += 1
                            ema_loss_decay = min(0.99, 1 - (1 / ema_loss_steps))
                            ema_loss = (ema_loss * ema_loss_decay) + (accumulated_loss_cpu * (1 - ema_loss_decay))
                            step_tqdm.set_postfix(
                                {
                                    "loss": accumulated_loss_cpu,
                                    "smooth loss": ema_loss,
                                }
                            )
                            self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)

                        accumulated_loss = 0.0
                        self.model_setup.after_optimizer_step(self.model, self.config, train_progress)

                        if self.model.ema:
                            assert multi.is_master()
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay", self.model.ema.get_current_decay(update_step), train_progress.global_step
                            )
                            self.model.ema.step(self.parameters, update_step)

                        self.one_step_trained = True

                perf.step_end()

                if self.config.validation and multi.is_master():
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    self.__mark_clean_train_exit()
                    return

            # If this epoch produced no training step (a starved bucket), the in-loop
            # periodic sample/save never ran; honor the schedule on the epoch boundary
            # so the cadence holds. No-op for any epoch that ran a step.
            self.__emit_scheduled_for_empty_epoch(train_progress, train_device)

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                self.__mark_clean_train_exit()
                return

        self.__mark_clean_train_exit()

    def end(self):
        if self.one_step_trained:
            if self.__should_emit_final_workspace_artifacts():
                self.__emit_final_workspace_artifacts(self.model.train_progress)

            self.model.to(self.temp_device)

            if self.config.backup_before_save and multi.is_master():
                self.__backup(self.model.train_progress)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            if multi.is_master():
                self.callbacks.on_update_status("Saving the final model")

                if self.model.ema:
                    self.model.ema.copy_ema_to(self.parameters, store_temp=False)
                if (
                    os.path.isdir(self.config.output_model_destination)
                    and self.config.output_model_format.is_single_file()
                ):
                    save_path = os.path.join(
                        self.config.output_model_destination,
                        f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}",
                    )
                else:
                    save_path = self.config.output_model_destination
                print("Saving " + save_path)

                self.model_saver.save(
                    model=self.model,
                    model_type=self.config.model_type,
                    output_model_format=self.config.output_model_format,
                    output_model_destination=save_path,
                    dtype=self.config.output_dtype.torch_dtype(),
                )

        if self.model is not None:
            self.model.to(self.temp_device)

        if multi.is_master():
            self.tensorboard.close()

            if self.config.tensorboard and not self.config.tensorboard_always_on:
                super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
