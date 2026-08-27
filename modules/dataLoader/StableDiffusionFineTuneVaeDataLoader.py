import os
import re

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderMgdsMixin import dataset_concepts
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory, path_util
from modules.util.bucket_tiers import (
    BUCKET_BUDGET_NAME,
    BUCKET_GROUP_NAME,
    BUCKET_KEEP_NAME,
    BUCKET_REPEAT_NAME,
    BucketingParams,
    aspect_bucket_rebalance_modules,
    aspect_bucketing_module,
    bucket_tags_enabled,
    bucketing_params,
)
from modules.util.cache_key import cache_salts
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.InlineAspectBatchSorting import InlineAspectBatchSorting
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation
from mgds.pipelineModules.VariationSorting import VariationSorting

import torch


@factory.register(BaseDataLoader, ModelType.STABLE_DIFFUSION_15, TrainingMethod.FINE_TUNE_VAE)
@factory.register(BaseDataLoader, ModelType.STABLE_DIFFUSION_15_INPAINTING, TrainingMethod.FINE_TUNE_VAE)
@factory.register(BaseDataLoader, ModelType.STABLE_DIFFUSION_20, TrainingMethod.FINE_TUNE_VAE)
@factory.register(BaseDataLoader, ModelType.STABLE_DIFFUSION_20_BASE, TrainingMethod.FINE_TUNE_VAE)
@factory.register(BaseDataLoader, ModelType.STABLE_DIFFUSION_20_INPAINTING, TrainingMethod.FINE_TUNE_VAE)
@factory.register(BaseDataLoader, ModelType.STABLE_DIFFUSION_20_DEPTH, TrainingMethod.FINE_TUNE_VAE)
@factory.register(BaseDataLoader, ModelType.STABLE_DIFFUSION_21, TrainingMethod.FINE_TUNE_VAE)
@factory.register(BaseDataLoader, ModelType.STABLE_DIFFUSION_21_BASE, TrainingMethod.FINE_TUNE_VAE)
class StableDiffusionFineTuneVaeDataLoader(BaseDataLoader):
    def _setup_cache_device(
            self,
            model: StableDiffusionModel,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
    ):
        model.materialize_only("vae")

        model.eval()

    def __enumerate_input_modules(self, config: TrainConfig) -> list:
        supported_extensions = path_util.supported_image_extensions()

        collect_paths = CollectPaths(
            concept_in_name='concept', path_in_name='path', include_subdirectories_in_name='concept.include_subdirectories', enabled_in_name='enabled',
            path_out_name='image_path', concept_out_name='concept',
            extensions=supported_extensions, include_postfix=None, exclude_postfix=['-masklabel']
        )

        # mask_path is derived in __derive_path_modules instead, so it sits after
        # AspectBucketRebalance and covers the duplicate rows borrow-copy mints.
        return [collect_paths]

    def __derive_path_modules(self, config: TrainConfig) -> list:
        mask_path = ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png')

        modules = []

        if config.masked_training:
            modules.append(mask_path)

        return modules

    def __load_input_modules(self, config: TrainConfig) -> list:
        load_image = LoadImage(path_in_name='image_path', image_out_name='image', range_min=-1.0, range_max=1.0, supported_extensions=path_util.supported_image_extensions())
        load_mask = LoadImage(path_in_name='mask_path', image_out_name='latent_mask', range_min=0, range_max=1, channels=1, supported_extensions=path_util.supported_image_extensions())

        modules = [load_image]

        if config.masked_training:
            modules.append(load_mask)

        return modules

    def __mask_augmentation_modules(self, config: TrainConfig) -> list:
        inputs = ['image']

        lowest_resolution = min([int(x.strip()) for x in re.split(r'\D', config.resolution) if x.strip() != ''])

        random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='latent_mask', additional_names=inputs, min_size=lowest_resolution,
                                                       min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20,
                                                       enabled_in_name='concept.image.enable_random_circular_mask_shrink')

        modules = []

        if config.masked_training:
            modules.append(random_mask_rotate_crop)

        return modules

    def __aspect_bucketing_in(self, config: TrainConfig, bucketing: BucketingParams):
        calc_aspect = CalcAspect(image_in_name='image', resolution_out_name='original_resolution')

        aspect_bucketing = aspect_bucketing_module(
            bucketing,
            resolution_in_name='original_resolution',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            target_frames_in_name='settings.target_frames',
            frame_dim_enabled=False,
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions',
        )

        single_aspect_calculation = SingleAspectCalculation(
            resolution_in_name='original_resolution',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions'
        )

        modules = [calc_aspect]

        if config.aspect_ratio_bucketing:
            modules.append(aspect_bucketing)
        else:
            modules.append(single_aspect_calculation)

        return modules

    def __bucket_rebalance_modules(self, config: TrainConfig, bucketing: BucketingParams):
        if not config.aspect_ratio_bucketing:
            return []

        return aspect_bucket_rebalance_modules(
            bucketing,
            path_in_name='image_path',
            concept_in_name='concept',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            image_extensions=path_util.supported_image_extensions(),
        )

    def __crop_modules(self, config: TrainConfig):
        inputs = ['image']

        if config.masked_training:
            inputs.append('latent_mask')

        scale_crop = ScaleCropImage(names=inputs, scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.image.enable_crop_jitter', crop_offset_out_name='crop_offset')

        modules = [scale_crop]

        return modules

    def __augmentation_modules(self, config: TrainConfig):
        inputs = ['image']

        if config.masked_training:
            inputs.append('latent_mask')

        random_flip = RandomFlip(names=inputs, enabled_in_name='concept.image.enable_random_flip', fixed_enabled_in_name='concept.image.enable_fixed_flip')
        random_rotate = RandomRotate(names=inputs, enabled_in_name='concept.image.enable_random_rotate', fixed_enabled_in_name='concept.image.enable_fixed_rotate', max_angle_in_name='concept.image.random_rotate_max_angle')
        random_brightness = RandomBrightness(names=['image'], enabled_in_name='concept.image.enable_random_brightness', fixed_enabled_in_name='concept.image.enable_fixed_brightness', max_strength_in_name='concept.image.random_brightness_max_strength')
        random_contrast = RandomContrast(names=['image'], enabled_in_name='concept.image.enable_random_contrast', fixed_enabled_in_name='concept.image.enable_fixed_contrast', max_strength_in_name='concept.image.random_contrast_max_strength')
        random_saturation = RandomSaturation(names=['image'], enabled_in_name='concept.image.enable_random_saturation', fixed_enabled_in_name='concept.image.enable_fixed_saturation', max_strength_in_name='concept.image.random_saturation_max_strength')
        random_hue = RandomHue(names=['image'], enabled_in_name='concept.image.enable_random_hue', fixed_enabled_in_name='concept.image.enable_fixed_hue', max_strength_in_name='concept.image.random_hue_max_strength')

        modules = [
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
            random_saturation,
            random_hue,
        ]

        return modules

    def __preparation_modules(self, config: TrainConfig, model: StableDiffusionModel):
        image = EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae)

        modules = [image]

        return modules

    def __cache_modules(
            self,
            config: TrainConfig,
            model: StableDiffusionModel,
            bucketing: BucketingParams,
            is_validation: bool,
    ):
        split_names = ['image', 'latent_image_distribution']

        if config.masked_training:
            split_names.append('latent_mask')

        aggregate_names = ['crop_resolution', 'image_path']

        # Carry the planner's tags through the cache alongside crop_resolution. Only
        # here: with latent caching off this loader has no VariationSorting either, and
        # the tags reach the sorter straight from AspectBucketing.
        if bucket_tags_enabled(config):
            aggregate_names = aggregate_names + [BUCKET_KEEP_NAME, BUCKET_REPEAT_NAME]

        sort_names = ['concept']

        def before_cache_fun():
            self._setup_cache_device(model, self.train_device, self.temp_device, config)

        # Nest under a content-addressed salt, exactly as the text2image loaders do:
        # this cache is keyed on the same group key, has the same length-only
        # staleness check, and gains the same two aggregate names once a tier exists.
        # See modules/util/cache_key.py.
        salts = cache_salts(
            config,
            bucketing=bucketing,
            concepts=dataset_concepts(config, is_validation),
            image_names=split_names + aggregate_names,
            text_names=[],
        )
        cache_dir = os.path.join(config.cache_dir, "vae", salts.image)

        disk_cache = DiskCache(cache_dir=cache_dir, split_names=split_names, aggregate_names=aggregate_names, variations_in_name='concept.image_variations', balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy',
                               variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'], group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_fun)
        variation_sorting = VariationSorting(names=sort_names, balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'],
                               group_enabled_in_name='concept.enabled', group_out_name=BUCKET_GROUP_NAME, budget_out_name=BUCKET_BUDGET_NAME)

        modules = []

        if config.latent_caching:
            modules.append(disk_cache)
            modules.append(variation_sorting)

        return modules

    def __output_modules(self, config: TrainConfig):
        output_names = ['image', 'latent_image', 'image_path']

        if config.masked_training:
            output_names.append('latent_mask')

        sort_names = output_names + ['concept']
        output_names = output_names + [('concept.loss_weight', 'loss_weight')]
        output_names = output_names + [('concept.type', 'concept_type')]

        # add for calculating loss per concept
        if config.validation:
            output_names.append(('concept.name', 'concept_name'))
            output_names.append(('concept.path', 'concept_path'))
            output_names.append(('concept.seed', 'concept_seed'))

        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')

        # The planner's drop / repeat decisions; None until a tier exists to make one,
        # which is the sorter's previous behaviour.
        bucket_tags = bucket_tags_enabled(config)
        keep_in_name = BUCKET_KEEP_NAME if bucket_tags else None
        repeat_in_name = BUCKET_REPEAT_NAME if bucket_tags else None
        # The balancing group id and SAMPLES budget, which only VariationSorting emits
        # and this loader only builds when latent caching is on. Without them a SAMPLES
        # budget is discarded rather than honoured, because VariationSorting no longer
        # takes the (bucket-blind) subset itself.
        group_in_name = BUCKET_GROUP_NAME if config.latent_caching else None
        budget_in_name = BUCKET_BUDGET_NAME if config.latent_caching else None

        if config.latent_caching:
            batch_sorting = AspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=config.batch_size,
                                               keep_in_name=keep_in_name, repeat_in_name=repeat_in_name, group_in_name=group_in_name, budget_in_name=budget_in_name)
        else:
            batch_sorting = InlineAspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=config.batch_size,
                                                     keep_in_name=keep_in_name, repeat_in_name=repeat_in_name, group_in_name=group_in_name, budget_in_name=budget_in_name)

        output = OutputPipelineModule(names=output_names)

        modules = [image_sample]

        modules.append(batch_sorting)

        modules.append(output)

        return modules

    def __debug_modules(self, config: TrainConfig, model: StableDiffusionModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.materialize("vae")

        decode_image = DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=model.vae)
        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=8)

        save_image = SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)
        save_mask = SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)

        modules = [decode_image, save_image]

        if config.masked_training or config.model_type.has_mask_input():
            modules += [upscale_mask, save_mask]

        return modules

    def _create_dataset(
            self,
            config: TrainConfig,
            model: StableDiffusionModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        # One derivation, shared by the planner and the bucketer, so the two cannot be
        # handed different geometry and pool aspect rungs differently.
        bucketing = bucketing_params(config, quantization=8, batch_size=config.batch_size)

        enumerate_input = self.__enumerate_input_modules(config)
        bucket_rebalance = self.__bucket_rebalance_modules(config, bucketing)
        derive_paths = self.__derive_path_modules(config)
        load_input = self.__load_input_modules(config)
        mask_augmentation = self.__mask_augmentation_modules(config)
        aspect_bucketing_in = self.__aspect_bucketing_in(config, bucketing)
        crop_modules = self.__crop_modules(config)
        augmentation_modules = self.__augmentation_modules(config)
        preparation_modules = self.__preparation_modules(config, model)
        cache_modules = self.__cache_modules(config, model, bucketing, is_validation)
        output_modules = self.__output_modules(config)

        debug_modules = self.__debug_modules(config, model)

        return self._create_mgds(
            config,
            [
                enumerate_input,
                bucket_rebalance,
                derive_paths,
                load_input,
                mask_augmentation,
                aspect_bucketing_in,
                crop_modules,
                augmentation_modules,
                preparation_modules,
                cache_modules,
                output_modules,

                debug_modules if config.debug_mode else None,
            ],
            train_progress,
            is_validation,
        )
