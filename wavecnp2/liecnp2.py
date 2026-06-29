
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
# from torch.distributions import MultivariateNormal
import numpy as np

from gpytorch.kernels import RBFKernel, ScaleKernel

from .modules import PowerFunction, Apply, Swish, LieConv
from .modules.lieconv import SeparableLieConv
from .liegroups import T




class GridLieCNP(nn.Module):
    """Grid LieGroup Convolutional Conditional Neural Process
    """
    def __init__(self, channel=1, group=T(2)):
        super().__init__()
        self.channel = channel
        self.group = group

        # self.conv_theta = LieConv(channel, 128, group=group,
        #                           num_nbhd=61, sampling_fraction=1., fill=1 / 10,
        #                           use_bn=True, mean=True, cache=True)
        self.conv_theta = LieConv(channel, 128, group=group,
                                  num_nbhd=121, sampling_fraction=1., fill=1 / 10,
                                  use_bn=True, mean=True, cache=True)
        # self.conv_theta = SeparableLieConv(channel, 128, num_nbhd=81, fill=81/4096, sample=1., group=SE2(), r=4.2)

        self.cnn = nn.Sequential(
            Apply(nn.Linear(128 * 2, 128), dim=1),
            ResBlock(128, 128, mean=True, group=group),
            ResBlock(128, 128, mean=True, group=group),
            ResBlock(128, 128, mean=True, group=group),
            Apply(nn.Linear(128, 2 * channel))
        )
        self.pos = nn.Softplus()

    def forward(self, x, cnt_mask_maxtrix, tgt_mask_maxtrix, tgt_mask_vec, cnt_num, tgt_num):
        # LieConv materialises large neighbourhood tensors. Chunking by image
        # keeps the default training batch from exhausting GPU memory.
        tgt_masks = tgt_mask_vec.reshape(x.shape[0], -1)
        predictions = [
            self._forward_batch(
                x[i:i + 1],
                cnt_mask_maxtrix[i:i + 1],
                tgt_masks[i],
            )
            for i in range(x.shape[0])
        ]
        means, stds = zip(*predictions)
        return torch.cat(means), torch.cat(stds)

    def _forward_batch(self, x, cnt_mask_maxtrix, tgt_mask_vec):
        ctx_coords, ctx_density, ctx_signal, ctx_mask = self.get_masked_image(
            x, cnt_mask_maxtrix
        )
        lifted_ctx_coords, lifted_ctx_density, lifted_ctx_mask = self.group.lift((ctx_coords, ctx_density, ctx_mask), 1)
        lifted_ctx_signal, _ = self.group.expand_like(ctx_signal, ctx_mask, lifted_ctx_coords)

        lifted_ctx_coords, density_prime, lifted_ctx_mask = self.conv_theta((lifted_ctx_coords, lifted_ctx_density, lifted_ctx_mask))
        _, signal_prime, _ = self.conv_theta((lifted_ctx_coords, lifted_ctx_signal, lifted_ctx_mask))

        ctx_h = torch.cat([density_prime, signal_prime], -1)
        _, f, _ = self.cnn((lifted_ctx_coords, ctx_h, lifted_ctx_mask))
        mean, std = f.split(self.channel, -1)

        mean = mean.flatten()
        std = self.pos(std)[:].flatten()

        return mean[tgt_mask_vec], std[tgt_mask_vec]

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])

    def get_masked_image(self, img, ctx_mask):
        """Get Context image and Target image

        Args:
            img (FloatTensor): image tensor (B, C, W, H)

        Returns:
            ctx_coords (FloatTensor): [B, W*H, 2]
            ctx_density (FloatTensor): [B, W*H, C]
            ctx_signal (FloatTensor): [B, W*H, C]

        """
        B, C, H, W = img.shape
        ctx_mask = ctx_mask.to(dtype=img.dtype).expand(B, C, H, W)

        #  [B, C, W, H] -> [B, W, H, C] -> [B, W*H, C]
        ctx_signal = (ctx_mask * img).permute(0, 2, 3, 1).reshape(B, -1, C)

        ctx_coords = torch.linspace(-W / 2., W / 2., W, device=img.device)
        # [B, W*H, 2]
        ctx_coords = torch.stack(
            torch.meshgrid(ctx_coords, ctx_coords, indexing='ij'), -1
        ).reshape(1, -1, 2).repeat(B, 1, 1)
        ctx_density = ctx_mask.reshape(B, -1, C)
        # ctx_mask = torch.ones(*ctx_signal.shape[:2], device=img.device).bool()
        ctx_mask = img.new_ones(B, W * H, dtype=torch.bool)
        return ctx_coords, ctx_density, ctx_signal, ctx_mask


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=T(2), mean=False, r=2.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group

        self.conv = nn.Sequential(
            SeparableLieConv(in_channels, out_channels, num_nbhd=81, fill=1 / 15, sample=1., group=group, r=r, use_bn=True, mean=True),
            Apply(nn.ReLU(inplace=True), dim=1),
            SeparableLieConv(out_channels, out_channels, num_nbhd=81, fill=1 / 15, sample=1., group=group, r=r, use_bn=True, mean=True)
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        coords, values, mask = self.conv(x)
        values = self.final_relu(values + shortcut[1])
        return coords, values, mask
