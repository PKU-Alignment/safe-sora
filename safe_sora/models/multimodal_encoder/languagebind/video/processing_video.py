import torch


try:
    import cv2
except ImportError:
    _HAS_CV2 = False
else:
    _HAS_CV2 = True
# from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import math

import decord
import numpy as np
from decord import VideoReader, cpu

# from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomHorizontalFlipVideo,
)
from transformers import ProcessorMixin


@torch.jit.ignore
def _interpolate_opencv(x: torch.Tensor, size: tuple[int, int], interpolation: str) -> torch.Tensor:
    """
    Down/up samples the input torch tensor x to the given size with given interpolation
    mode.
    Args:
        input (Tensor): the input tensor to be down/up sampled.
        size (Tuple[int, int]): expected output spatial size.
        interpolation: model to perform interpolation, options include `nearest`,
            `linear`, `bilinear`, `bicubic`.
    """
    if not _HAS_CV2:
        raise ImportError(
            'opencv is required to use opencv transforms. Please '
            "install with 'pip install opencv-python'.",
        )

    _opencv_pytorch_interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
    }
    assert interpolation in _opencv_pytorch_interpolation_map
    new_h, new_w = size
    img_array_list = [
        img_tensor.squeeze(0).numpy() for img_tensor in x.permute(1, 2, 3, 0).split(1, dim=0)
    ]
    resized_img_array_list = [
        cv2.resize(
            img_array,
            (new_w, new_h),  # The input order for OpenCV is w, h.
            interpolation=_opencv_pytorch_interpolation_map[interpolation],
        )
        for img_array in img_array_list
    ]
    img_array = np.concatenate(
        [np.expand_dims(img_array, axis=0) for img_array in resized_img_array_list],
        axis=0,
    )
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_array))
    img_tensor = img_tensor.permute(3, 0, 1, 2)
    return img_tensor


def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = 'bilinear',
    backend: str = 'pytorch',
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
    Returns:
        An x-like Tensor with scaled spatial dims.
    """
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    assert backend in ('pytorch', 'opencv')
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    if backend == 'pytorch':
        return torch.nn.functional.interpolate(
            x,
            size=(new_h, new_w),
            mode=interpolation,
            align_corners=False,
        )
    elif backend == 'opencv':
        return _interpolate_opencv(x, size=(new_h, new_w), interpolation=interpolation)
    else:
        raise NotImplementedError(f'{backend} backend not supported.')


class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``short_side_scale``.
    """

    def __init__(self, size: int, interpolation: str = 'bilinear', backend: str = 'pytorch'):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return short_side_scale(
            x,
            self._size,
            self._interpolation,
            self._backend,
        )


decord.bridge.set_bridge('torch')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x


def get_video_transform(config):
    config = config.vision_config

    if config.video_decode_backend == 'decord' or config.video_decode_backend == 'opencv':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ],
        )
    else:
        raise NameError('video_decode_backend should specify in (decord, opencv)')
    return transform


def load_and_transform_video(
    video_path,
    transform,
    video_decode_backend='opencv',
    clip_start_sec=0.0,
    clip_end_sec=None,
    num_frames=8,
):

    if video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

    # elif video_decode_backend == 'pytorchvideo':
    #     #  decord pyav
    #     video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
    #     duration = video.duration
    #     start_sec = clip_start_sec  # secs
    #     end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
    #     video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    #     video_outputs = transform(video_data)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration - 1, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs


class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = 'LanguageBindVideoTokenizer'

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError('You have to specify either text or images. Both cannot be none.')

        if text is not None:
            encoding = self.tokenizer(
                text,
                max_length=context_length,
                padding='max_length',
                truncation=True,
                return_tensors=return_tensors,
                **kwargs,
            )

        if images is not None:
            images = make_list_of_images(images)
            image_features = [
                self.image_processor(
                    image,
                    self.transform,
                    video_decode_backend=self.config.vision_config.video_decode_backend,
                    num_frames=self.config.vision_config.num_frames,
                )
                for image in images
            ]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding['pixel_values'] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {'pixel_values': image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
