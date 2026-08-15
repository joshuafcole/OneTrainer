import io
import os
import traceback
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from pathlib import Path

from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.sample_metadata import SampleProvenance, build_exif, build_png_info

import torch

import av
from PIL import Image


class ModelSamplerOutput:
    def __init__(
            self,
            file_type: FileType,
            data: Image.Image | torch.Tensor | bytes,

    ):
        self.file_type = file_type
        if isinstance(data, bytes):
            assert file_type == FileType.IMAGE
            self.data = Image.open(io.BytesIO(data))
        else:
            self.data = data

    #Reduce to a JPEG bytestream for cloud training:
    def __reduce__(self):
        match self.file_type:
            case FileType.IMAGE:
                b = io.BytesIO()
                self.data.save(b, format='JPEG')
                return ModelSamplerOutput, (self.file_type, b.getvalue())
            case FileType.VIDEO:
                #do not transfer videos; they are not shown anyway
                #the video sample file is transferred via workspace sync
                return ModelSamplerOutput, (self.file_type, None)
            case FileType.AUDIO:
                # TODO
                return ModelSamplerOutput, (self.file_type, None)
            case _:
                return ModelSamplerOutput, (self.file_type, None)


class BaseModelSampler(metaclass=ABCMeta):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device
        self._provenance: SampleProvenance | None = None

    def set_provenance(self, prov: SampleProvenance | None):
        """Provenance for the next ``save_sampler_output`` call. Set by the
        trainer, which is the only layer that knows the training state a
        sample corresponds to; the sampler itself has no notion of step/epoch."""
        self._provenance = prov

    @abstractmethod
    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            video_format: VideoFormat,
            audio_format: AudioFormat,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        pass

    @staticmethod
    def quantize_resolution(resolution: int, quantization: int) -> int:
        return round(resolution / quantization) * quantization

    def save_sampler_output(
            self,
            sampler_output: ModelSamplerOutput,
            destination: str,
            image_format: ImageFormat | None,
            video_format: VideoFormat | None,
            audio_format: AudioFormat | None,
            fps: int = 24,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        if sampler_output.file_type == FileType.IMAGE:
            if image_format is None:
                raise ValueError("Image format required for sampling an image")
            image = sampler_output.data

            # Provenance rides whichever container the run is configured for --
            # PNG text chunks, JPEG EXIF. JPG is sample_image_format's default,
            # so a PNG-only stamp would be inert on most real runs.
            provenance_kwargs = {}
            if self._provenance is not None:
                try:
                    if image_format == ImageFormat.PNG:
                        provenance_kwargs = {"pnginfo": build_png_info(self._provenance)}
                    elif image_format == ImageFormat.JPG:
                        provenance_kwargs = {"exif": build_exif(self._provenance)}
                except Exception:
                    # A broken provenance chunk must never cost a training run
                    # its sample -- log and fall through to a plain save.
                    traceback.print_exc()
                    print("Could not build sample provenance metadata, saving without it")
                    provenance_kwargs = {}

            image.save(
                destination + image_format.extension(),
                format=image_format.pil_format(),
                **provenance_kwargs,
            )
        elif sampler_output.file_type == FileType.VIDEO:
            if video_format is None:
                raise ValueError("Video format required for sampling a video")

            if isinstance(sampler_output.data, torch.Tensor):
                video_tensor = sampler_output.data.detach().cpu()

                if len(video_tensor.shape) == 4:
                    shape = video_tensor.shape
                    # (T, H, W, C) if last dim is channels, otherwise assume (C, T, H, W)
                    frames = video_tensor.numpy() if shape[-1] == 3 else video_tensor.permute(1, 2, 3, 0).numpy()

                    frames = (
                        (frames * 255).astype('uint8')
                        if frames.max() <= 1.0
                        else frames.astype('uint8')
                    )

                    with av.open(destination + video_format.extension(), 'w') as container:
                        stream = container.add_stream('libx264', rate=fps)
                        stream.options = {'crf': '17'}
                        stream.width = frames.shape[2]
                        stream.height = frames.shape[1]
                        stream.pix_fmt = 'yuv420p'  # Required pixel format for H.264

                        for frame_data in frames:
                            frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                            for packet in stream.encode(frame):
                                container.mux(packet)

                        for packet in stream.encode():
                            container.mux(packet)
                else:
                    raise ValueError(f"Expected 4D video tensor (T, H, W, C) or (C, T, H, W), got shape {video_tensor.shape}")
        elif sampler_output.file_type == FileType.AUDIO:
            pass # TODO
