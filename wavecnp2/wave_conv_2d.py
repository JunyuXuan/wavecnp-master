import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from wavecnp2.architectures import CNN, ResConvBlock
from wavecnp2.transform2dnew import DWTForward, DWTInverse
from wavecnp2.utils import StandardDecoder2, device, make_abs_conv


__all__ = ["ConvCNP2", "WaveCNP2"]


def _depthwise_positive_conv(y_dim, kernel_size):
    return make_abs_conv(nn.Conv2d)(
        y_dim,
        y_dim,
        groups=y_dim,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=False,
    )


def _cnn_factory(n_conv_layers, n_blocks, kernel_size):
    return partial(
        CNN,
        ConvBlock=ResConvBlock,
        n_conv_layers=n_conv_layers,
        Conv=nn.Conv2d,
        n_blocks=n_blocks,
        Normalization=nn.BatchNorm2d,
        is_chan_last=True,
        kernel_size=kernel_size,
    )


class ConvCNP2(nn.Module):
    """Two-dimensional ConvCNP model."""

    def __init__(
        self,
        y_dim,
        r_dim,
        conv_kernel_size=11,
        cnn_n_conv_layers=1,
        cnn_n_blocks=3,
        cnn_kernel_size=5,
    ):
        super().__init__()

        self.y_dim = y_dim
        self.r_dim = r_dim
        self.conv = _depthwise_positive_conv(y_dim, conv_kernel_size).to(device)
        self.resizer = nn.Linear(self.y_dim * 2, self.r_dim).to(device)
        self.induced_to_induced = _cnn_factory(
            cnn_n_conv_layers, cnn_n_blocks, cnn_kernel_size
        )(self.r_dim).to(device)
        self.decoder = StandardDecoder2(
            input_dim=self.r_dim,
            latent_dim=self.r_dim,
            output_dim=self.y_dim,
        ).to(device)

    def forward(self, x, cnt_mask_m, tgt_mask_m, tgt_mask, cnt_num, tgt_num):
        x_context = x * cnt_mask_m.float()
        signal = self.conv(x_context)
        density = self.conv(cnt_mask_m.float())

        h = signal / torch.clamp(density, min=1e-5)
        h = torch.cat([h, density], dim=1)
        h = h.permute(0, 2, 3, 1)
        h = self.resizer(h)

        f = self.induced_to_induced(h)
        f = f.reshape(-1, self.r_dim)
        f = f[tgt_mask]
        mean, std = self.decoder(f)
        return mean.squeeze(), std.squeeze()

    @property
    def num_params(self):
        return np.sum([torch.tensor(param.shape).prod() for param in self.parameters()])


class WaveCNP2(nn.Module):
    """Two-dimensional WaveCNP model."""

    def __init__(
        self,
        y_dim,
        r_dim,
        img_size,
        conv_kernel_size=11,
        cnn_n_conv_layers=1,
        cnn_n_blocks=3,
        cnn_kernel_size=5,
        TASK_DWT=True,
        ADAPT=False,
        level=3,
        smooth=1.0,
        wavelet="db2",
        wavelet_filter_length=4,
        wavelet_mode="zero",
        wave_gate_init=-3.0,
    ):
        super().__init__()

        self.y_dim = y_dim
        self.r_dim = r_dim
        self.img_size = img_size
        self.TASK_DWT = TASK_DWT
        self.level = level
        self.smooth = smooth

        self.conv = _depthwise_positive_conv(y_dim, conv_kernel_size).to(device)
        self.resizer = nn.Linear(self.y_dim * 2, self.r_dim).to(device)

        cnn_builder = _cnn_factory(cnn_n_conv_layers, cnn_n_blocks, cnn_kernel_size)
        self.induced_to_induced = cnn_builder(self.r_dim).to(device)
        self.decoder = StandardDecoder2(
            input_dim=self.r_dim,
            latent_dim=self.r_dim,
            output_dim=self.y_dim,
        ).to(device)

        if self.TASK_DWT:
            if ADAPT:
                self.wt_alpha_col = nn.Parameter(
                    torch.tensor(math.pi / 3.0), requires_grad=True
                )
                self.wt_alpha_row = nn.Parameter(
                    torch.tensor(math.pi / 3.0), requires_grad=True
                )
            else:
                self.wt_alpha_col = torch.tensor(math.pi / 2.0, device=device)
                self.wt_alpha_row = torch.tensor(math.pi / 2.0, device=device)

            self.dwt = DWTForward(
                J=self.level,
                wave=wavelet,
                mode=wavelet_mode,
                adapt=True,
                L=wavelet_filter_length,
                wt_alpha_col=self.wt_alpha_col,
                wt_alpha_row=self.wt_alpha_row,
            ).to(device)
            self.iwt = DWTInverse(
                wave=wavelet,
                mode=wavelet_mode,
                adapt=True,
                L=wavelet_filter_length,
                wt_alpha_col=self.wt_alpha_col,
                wt_alpha_row=self.wt_alpha_row,
            ).to(device)
            self.filterband_transform_list = nn.ModuleList(
                cnn_builder(self.r_dim).to(device) for _ in range(self.level)
            )
            self.detail_transform_list = nn.ModuleList(
                cnn_builder(self.r_dim).to(device) for _ in range(self.level)
            )
            self.selector = nn.Linear(self.r_dim, self.level).to(device)
            self.wave_gate = nn.Parameter(
                torch.tensor(float(wave_gate_init), device=device), requires_grad=True
            )

    def _clone_bands(self, bands):
        return [band.clone() for band in bands]

    def forward(self, x, cnt_mask_m, tgt_mask_m, tgt_mask, cnt_num, tgt_num):
        x_context = x * cnt_mask_m.float()
        signal = self.conv(x_context)
        density = self.conv(cnt_mask_m.float())

        h = signal / torch.clamp(density, min=1e-5)
        h = torch.cat([h, density], dim=1)
        h = h.permute(0, 2, 3, 1)
        h = self.resizer(h)

        f_base = self.induced_to_induced(h)
        if self.TASK_DWT:
            batch_size = x.shape[0]
            pooled_h = h.mean(dim=(1, 2))
            level_weight = nn.functional.softmax(
                self.smooth * self.selector(pooled_h),
                dim=-1,
            )

            yl, yh = self.dwt(h.permute(0, 3, 1, 2))
            f_wave = None
            for i, (smooth_transform, detail_transform) in enumerate(
                zip(self.filterband_transform_list, self.detail_transform_list)
            ):
                smooth_yh = self._clone_bands(yh)
                smooth_yh[i] = torch.zeros_like(smooth_yh[i])
                smooth_reconstructed = self.iwt((yl, smooth_yh)).permute(0, 2, 3, 1)

                detail_yh = [torch.zeros_like(band) for band in yh]
                detail_yh[i] = yh[i]
                detail_reconstructed = self.iwt(
                    (torch.zeros_like(yl), detail_yh)
                ).permute(0, 2, 3, 1)

                transformed = (
                    smooth_transform(smooth_reconstructed)
                    + detail_transform(detail_reconstructed)
                )
                weight = level_weight[:, i].reshape(batch_size, 1, 1, 1)
                f_wave = transformed * weight if f_wave is None else f_wave + transformed * weight
            f = f_base + torch.sigmoid(self.wave_gate) * f_wave
        else:
            f = f_base

        f = f.reshape(-1, f.shape[-1])
        f = f[tgt_mask]
        mean, std = self.decoder(f)
        return mean.squeeze(), std.squeeze() + 1e-5

    @property
    def num_params(self):
        return np.sum([torch.tensor(param.shape).prod() for param in self.parameters()])
