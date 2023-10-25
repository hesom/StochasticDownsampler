from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

class StochasticDownsampler(nn.Module):
    """Stochastically downsamples a feature map to a target resolution. Conceptually approximates a pixel integral by monte carlo sampling"""
    def __init__(self,
                 resolution: Tuple[int, int],
                 spp: int = 16,
                 reduction: Literal["mean", "sum", "min", "max", "prod"] = "mean",
                 jitter_type: Literal["uniform", "normal"] = "uniform"
                 ):
        super().__init__()
        if (not isinstance(resolution, tuple)
            and not isinstance(resolution, list)):
            resolution = (resolution, resolution)
        if len(resolution) != 2:
            raise ValueError(f"Resolution must be a tuple of length 2, got {resolution}")

        self.resolution = resolution
        self.spp = spp
        self.reduction = reduction
        self.jitter_type = jitter_type
        if self.jitter_type == "uniform":
            self.jitter_fn = torch.rand
        elif self.jitter_type == "normal":
            self.jitter_fn = lambda *args, **kwargs : torch.randn(*args, **kwargs) + 0.5
        else:
            raise NotImplementedError(f"Jitter type {jitter_type} not supported")

    def forward(self, x: torch.Tensor):
        """
        Downsamples x to the target resolution
        :param x: high-res input feature map, shape (batch_size, C, H, W)
        :return: downsampled image, shape (batch_size, C, resolution[0], resolution[1])
        """
        b, c, h, w = x.shape
        resolution, spp = self.resolution, self.spp

        step_x = (1 + 1) / resolution[1]
        step_y = (1 + 1) / resolution[0]
        pixel_pos_x = torch.arange(-1, 1, step_x, device=x.device)
        pixel_pos_y = torch.arange(-1, 1, step_y, device=x.device)
        pixel_pos = torch.stack(torch.meshgrid(pixel_pos_x, pixel_pos_y, indexing='xy'), dim=2)

        # add subpixel jitter
        jitter = self.jitter_fn((spp, resolution[0], resolution[1], 2), device=x.device)
        jitter[..., 0] *= step_x
        jitter[..., 1] *= step_y
        pixel_pos = pixel_pos.unsqueeze(0) + jitter # (spp, resolution[0], resolution[1], 2)

        pixel_pos = einops.repeat(pixel_pos, 'spp h w c -> (b spp) h w c', b=b)
        x_tiled = einops.repeat(x, 'b c h w -> (b spp) c h w', spp=spp)

        samples = F.grid_sample(x_tiled, pixel_pos, mode='bilinear', padding_mode='border', align_corners=False)
        return einops.reduce(samples, '(b spp) c h w -> b c h w', self.reduction, b=b)
