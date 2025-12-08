from dataclasses import dataclass
from typing import Optional

import torch

from diffusers.utils import BaseOutput


@dataclass
class LTXPipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
        latents (`torch.Tensor`, *optional*):
            The latent representations of the video before VAE decoding. Only returned when `output_type="both"`.
    """

    frames: torch.Tensor
    latents: Optional[torch.Tensor] = None
