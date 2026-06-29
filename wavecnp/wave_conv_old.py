import numpy as np
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
from wavecnp.architectures import SimpleMLP, SimpleConv
from math import log, pow
import pywt

from wavecnp.utils import (
    init_sequential_weights,
    compute_dists,
    to_multiple,
    device
)


__all__ = ['ConvDeepSet', 'WaveDeepSet', 'WaveCNP']


class ConvDeepSet(nn.Module):
    """One-dimensional set convolution layer. Uses an RBF kernel for
    `psi(x, x')`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        learn_length_scale (bool): Learn the length scales of the channels.
        init_length_scale (float): Initial value for the length scale.
        use_density (bool, optional): Append density channel to inputs.
            Defaults to `True`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 learn_length_scale,
                 init_length_scale,
                 use_density=True):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.use_density = use_density
        self.in_channels = in_channels + 1 if self.use_density else in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels),
                                  requires_grad=learn_length_scale)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        `in_channels + 1`-dimensional representation to dimensionality
        `out_channels`.

        Returns:
            :class:`torch.nn.Module`: Linear layer applied point-wise to
                channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations `t`.

        Args:
            x (tensor): Inputs of observations of shape `(n, 1)`.
            y (tensor): Outputs of observations of shape `(n, in_channels)`.
            t (tensor): Inputs to evaluate function at of shape `(m, 1)`.

        Returns:
            tensor: Outputs of evaluated function at `z` of shape
                `(m, out_channels)`.
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        if len(y.shape) == 2:
            y = y.unsqueeze(2)
        if len(t.shape) == 2:
            t = t.unsqueeze(2)

        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        if self.use_density:
            # Compute the extra density channel.
            # Shape: (batch, n_in, 1).
            density = torch.ones(batch_size, n_in, 1).to(device)

            # Concatenate the channel.
            # Shape: (batch, n_in, in_channels).
            y_out = torch.cat([density, y], dim=2)
        else:
            y_out = y

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        if self.use_density:
            # Use density channel to normalize convolution
            density, conv = y_out[..., :1], y_out[..., 1:]
            normalized_conv = conv / (density + 1e-8)
            y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between `x` and `t`.

        Returns:
            tensor: Evaluation of `psi(x, t)` with `psi` an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)


class WaveDeepSet(nn.Module):
    """One-dimensional set convolution layer. Uses an RBF kernel for
    `psi(x, x')`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        learn_length_scale (bool): Learn the length scales of the channels.
        init_length_scale (float): Initial value for the length scale.
        use_density (bool, optional): Append density channel to inputs.
            Defaults to `True`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 learn_length_scale,
                 init_length_scale,
                 num_points,
                 use_density=True):
        super(WaveDeepSet, self).__init__()
        self.out_channels = out_channels
        self.use_density = use_density
        self.in_channels = in_channels + 1 if self.use_density else in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels),
                                  requires_grad=learn_length_scale)
        self.sigma_fn = torch.exp

        ## parameters of dwt
        self.num_points = num_points

        self.ind_level = 3  # int(log(num_points) / log(2.))
        self.ind_wavename = 'db2'
        self.ind_L = 4  # filter coefficient vector len
        self.ind_mode = 'zero'

        self.ind_dwt = DWT1DForward(wave=self.ind_wavename, J=self.ind_level).to(device)
        self.ind_iwt = DWT1DInverse(wave=self.ind_wavename).to(device)

        self.ind_filterband_size_list = []
        self.ind_filterband_transform_list = []

        tmp = self.num_points
        for _ in range(self.ind_level):
            tmp = pywt.dwt_coeff_len(tmp, self.ind_L, mode=self.ind_mode)
            self.ind_filterband_size_list.append(tmp)
            # self.filterband_transform_list.append(SimpleMLP(tmp))
            self.ind_filterband_transform_list.append(SimpleConv(self.in_channels, self.in_channels).to(device))

        self.yl_transform = SimpleConv()
        self.yl_transform.to(device)

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        `in_channels + 1`-dimensional representation to dimensionality
        `out_channels`.

        Returns:
            :class:`torch.nn.Module`: Linear layer applied point-wise to
                channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        model.to(device)
        init_sequential_weights(model)
        return model

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations `t`.

        Args:
            x (tensor): Inputs of observations of shape `(n, 1)`.
            y (tensor): Outputs of observations of shape `(n, in_channels)`.
            t (tensor): Inputs to evaluate function at of shape `(m, 1)`.

        Returns:
            tensor: Outputs of evaluated function at `z` of shape
                `(m, out_channels)`.
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        if len(y.shape) == 2:
            y = y.unsqueeze(2)
        if len(t.shape) == 2:
            t = t.unsqueeze(2)

        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        if self.use_density:
            # Compute the extra density channel.
            # Shape: (batch, n_in, 1).
            density = torch.ones(batch_size, n_in, 1).to(device)

            # Concatenate the channel.
            # Shape: (batch, n_in, in_channels).
            y_out = torch.cat([density, y], dim=2)
        else:
            y_out = y

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Apply wavelet transform
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y_out.permute(0, 1, 3, 2)
        y_out = y_out.reshape(-1, y_out.shape[-2], y_out.shape[-1])
        yl, yh = self.ind_dwt(y_out)

        # Apply non-linear transform (conv or mlp) seperately
        for l in range(self.ind_level):
            yh[l] = self.ind_filterband_transform_list[l](yh[l])

        y_out = self.ind_iwt((self.yl_transform(yl), yh))
        y_out = y_out.reshape(batch_size, -1, y_out.shape[-2], y_out.shape[-1])
        y_out = y_out.permute(0, 1, 3, 2)

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        if self.use_density:
            # Use density channel to normalize convolution
            density, conv = y_out[..., :1], y_out[..., 1:]
            normalized_conv = conv / (density + 1e-8)
            y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between `x` and `t`.

        Returns:
            tensor: Evaluation of `psi(x, t)` with `psi` an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        dist_new = dists.view(a, b, c, -1)
        return torch.exp(-0.5 * dist_new / scales ** 2)


class WaveCNP(nn.Module):
    """One-dimensional WaveCNP model.

    Args:
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
        architecture (:class:`nn.Module`): Convolutional architecture to place
            on functional representation (rho).
    """

    def __init__(self,
                 learn_length_scale,
                 points_per_unit,
                 num_points,
                 architecture,
                 IND_DWT=False):
        super(WaveCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.conv_net = architecture
        self.conv_net.to(device)
        self.num_points = num_points
        self.IND_DWT = IND_DWT
        self.multiplier = 2 ** self.conv_net.num_halving_layers

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit

        if self.IND_DWT:
            self.l0 = WaveDeepSet(
                in_channels=1,
                out_channels=self.conv_net.in_channels,
                learn_length_scale=learn_length_scale,
                init_length_scale=init_length_scale,
                num_points=self.num_points,
                use_density=True
            )
        else:
            self.l0 = ConvDeepSet(
                in_channels=1,
                out_channels=self.conv_net.in_channels,
                learn_length_scale=learn_length_scale,
                init_length_scale=init_length_scale,
                use_density=True
            )
        self.mean_layer = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )
        self.sigma_layer = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )

        # parameters of dwt
        self.level = 3          # int(log(num_points) / log(2.))
        self.wavename = 'db2'
        self.L = 4              # filter coefficient vecotr len
        self.mode = 'zero'

        self.dwt = DWT1DForward(wave=self.wavename, J=self.level).to(device)
        self.iwt = DWT1DInverse(wave=self.wavename).to(device)

        self.filterband_size_list = []
        self.filterband_transform_list = []

        tmp = self.num_points
        for _ in range(self.level):
            tmp = pywt.dwt_coeff_len(tmp, self.L, mode=self.mode)
            self.filterband_size_list.append(tmp)
            # self.filterband_transform_list.append(SimpleMLP(tmp).to(device))
            self.filterband_transform_list.append(SimpleConv().to(device))

        self.yl_transform = SimpleConv()
        self.yl_transform.to(device)

        self.TASK_DWT = False

    def forward(self, x, y, x_out):
        """Run the model forward.

        Args:
            x (tensor): Observation locations of shape
                `(batch, data, features)`.
            y (tensor): Observation values of shape
                `(batch, data, outputs)`.
            x_out (tensor): Locations of outputs of shape
                `(batch, data, features)`.
        Returns:
            tuple[tensor]: Means and standard deviations of shape
                `(batch_out, channels_out)`.
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        if len(y.shape) == 2:
            y = y.unsqueeze(2)
        if len(x_out.shape) == 2:
            x_out = x_out.unsqueeze(2)

        # Determine the grid on which to evaluate functional representation.
        x_min = torch.minimum(torch.minimum(x.amin(), x_out.amin()), x.new_tensor(-2.0)) - 0.1
        x_max = torch.maximum(torch.maximum(x.amax(), x_out.amax()), x.new_tensor(2.0)) + 0.1
        # num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
        #                              self.multiplier))
        x_grid = x_min + (x_max - x_min) * torch.arange(self.num_points, device=x.device, dtype=x.dtype) / (self.num_points - 1)
        x_grid = x_grid[None, :, None].expand(x.shape[0], -1, -1)

        # Apply first layer. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.l0(x, y, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], self.num_points)

        # if self.TASK_DWT:
        #     # Apply wavelet transform and conv net
        #     # Shape: (batch, out_channels, num_points).
        #     yl, yh = self.dwt(h)
        #
        #     # Apply non-linear transform (conv or mlp) separately
        #     for l in range(self.level):
        #         yh[l] = self.filterband_transform_list[l](yh[l])
        #
        #     h = self.iwt((self.yl_transform(yl), yh))
        # else:
        #     h = self.conv_net(h)

        # Apply wavelet transform and conv net
        # Shape: (batch, out_channels, num_points).
        yl, yh = self.dwt(h)

        # Apply non-linear transform (conv or mlp) separately
        for l in range(self.level):
            yh[l] = self.filterband_transform_list[l](yh[l])

        h = self.iwt((self.yl_transform(yl), yh))

        #
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations.
        mean = self.mean_layer(x_grid, h, x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))

        return mean, sigma

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
