import numpy as np
import torch
import torch.nn as nn
# from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
from wavecnp.transform1dnew import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
from wavecnp.architectures import SimpleConv
import math

from wavecnp.utils import (
    init_sequential_weights,
    compute_dists,
    to_multiple,
    device
)

from .liegroups import T
from gpytorch.kernels import RBFKernel, ScaleKernel
from .modules import PowerFunction, Apply, Swish, LieConv


__all__ = ['ConvDeepSet', 'WaveDeepSet', 'ConvCNP', 'LieCNP', 'WaveCNP']


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
                 use_density=True,
                 ADAPT=False):
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

        if ADAPT:
            self.ind_wt_alpha = nn.Parameter(torch.tensor(math.pi / 3.0), requires_grad=True)
        else:
            self.ind_wt_alpha = -1

        self.ind_dwt = DWT1DForward(wave=self.ind_wavename, J=self.ind_level, adapt=ADAPT, L=self.ind_L,
                                wt_alpha=self.ind_wt_alpha).to(device)
        self.ind_iwt = DWT1DInverse(wave=self.ind_wavename, adapt=ADAPT, L=self.ind_L,
                                wt_alpha=self.ind_wt_alpha).to(device)

        self.conv_net2 = SimpleConv(self.in_channels, self.in_channels).to(device)
        self.conv_net1 = SimpleConv(self.in_channels, self.in_channels).to(device)
        self.conv_net0 = SimpleConv(self.in_channels, self.in_channels).to(device)

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

        # Apply non-linear transform (conv or mlp) seperately
        yl, yh = self.ind_dwt(y_out)

        #
        yh[0] = torch.zeros_like(yh[0])
        y_out0 = self.ind_iwt((yl, yh))

        yh[1] = torch.zeros_like(yh[1])
        y_out1 = self.ind_iwt((yl, yh))

        yh[2] = torch.zeros_like(yh[2])
        y_out2 = self.ind_iwt((yl, yh))

        #
        y_out2 = self.conv_net2(y_out2)
        y_out1 = self.conv_net1(y_out1)
        y_out0 = self.conv_net0(y_out0)

        y_out = (y_out + y_out2 + y_out1 + y_out0) / 4

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


class ConvCNP(nn.Module):
    """One-dimensional ConvCNP model.

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
                 latent_dim
                 ):
        super(ConvCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.h_dim = latent_dim
        self.conv_net = SimpleConv(self.h_dim, self.h_dim)
        self.num_points = num_points
        self.multiplier = 2 ** self.conv_net.num_halving_layers

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit

        self.l0 = ConvDeepSet(
            in_channels=1,
            out_channels=self.conv_net.in_channels,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=True
        )
        self.mean_layer = ConvDeepSet(
            in_channels=int(self.conv_net.out_channels /2),
            # in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )
        self.sigma_layer = ConvDeepSet(
            in_channels=int(self.conv_net.out_channels/2),
            # in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )

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

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.l0(x, y, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], self.num_points)

        h = self.conv_net(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations.
        # mean = self.mean_layer(x_grid, h, x_out)
        # sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))
        mean = self.mean_layer(x_grid, h[:, :, :int(self.h_dim / 2)], x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h[:, :, int(self.h_dim / 2):], x_out))

        return mean, sigma

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])




class LieCNP(nn.Module):
    """Lie Group Neural Process
    """

    def __init__(self, x_dim, y_dim, group=T(1), nbhd=5, fill=1 / 15):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.group = group
        self.fill = fill
        self.num_nbhd = nbhd

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(
            LieConv(x_dim + 2, 16, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(16, 32, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(32, 64, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(64, 64, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(64, 32, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(32, 16, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(16, 2, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())

    def forward(self,  x, y, x_out):
        # ctx_coords, ctx_values = ctx

        rep_coords = self.support_points(x, x_out)

        h = self.psi(rep_coords, x).matmul(self.phi(y))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)

        rep_values = torch.cat([rep_coords, h0, h1], -1)  # (B, T, K+1+2) = (B, 784, 4)
        rep_mask = torch.ones(rep_values.shape[:2], dtype=torch.bool, device=rep_values.device)
        lifted_coords, lifted_values, lifted_mask = self.group.lift((rep_coords, rep_values, rep_mask), nsamples=1)

        _, f, _ = self.cnn((lifted_coords, lifted_values, lifted_mask))
        f_mu, f_sigma = f.split(1, -1)

        mu = self.psi_rho(x_out, rep_coords).matmul(f_mu)
        sigma = self.psi_rho(x_out, rep_coords).matmul(self.pos(f_sigma))

        return mu, sigma

    def support_points(self, ctx_coords, tgt_coords):
        if self.x_dim == 1:
            tmp = torch.cat([ctx_coords.reshape(-1), tgt_coords.reshape(-1)])
            lower, upper = tmp.min(), tmp.max()
            num_t = max(int((16 * (upper - lower)).item()), 1)
            t_coords = torch.linspace(start=lower, end=upper, steps=num_t, device=ctx_coords.device).reshape(1, -1, self.x_dim)

        elif self.x_dim == 2:
            i = torch.linspace(-28 / 2, 28 / 2, 28, device=ctx_coords.device)
            t_coords = torch.stack(torch.meshgrid([i, i]), dim=-1).reshape(1, -1, 2)
        else:
            raise NotImplementedError
        t_coords = t_coords.repeat(ctx_coords.size(0), 1, 1)
        return t_coords

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])


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
                 latent_dim=128,
                 IND_DWT=False,
                 TASK_DWT=False,
                 ADAPT=False,
                 level=3,
                 smooth=1.0,
                 wave_gate_init=-3.0):
        super(WaveCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.h_dim = latent_dim
        self.conv_net = SimpleConv(self.h_dim, self.h_dim)
        self.num_points = num_points
        self.level = level
        self.smooth = smooth

        self.IND_DWT = IND_DWT
        self.TASK_DWT = TASK_DWT
        # self.multiplier = 2 ** self.conv_net.num_halving_layers

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
                use_density=True,
                ADAPT=ADAPT
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
            # in_channels=self.conv_net.out_channels,
            in_channels=int(self.conv_net.out_channels / 2),
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )
        self.sigma_layer = ConvDeepSet(
            # in_channels=self.conv_net.out_channels,
            in_channels=int(self.conv_net.out_channels/2),
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )

        if self.TASK_DWT:
            # parameters of dwt
            self.wavename = 'db2'
            self.L = 4              # filter coefficient vecotr len
            self.mode = 'zero'

            if ADAPT:
                self.wt_alpha_dwt = nn.Parameter(torch.tensor(math.pi / 3.0), requires_grad=True)
                # self.wt_alpha_iwt = nn.Parameter(torch.tensor(math.pi / 3.0), requires_grad=True)
            else:
                self.wt_alpha_dwt = -1
                self.wt_alpha_iwt = -1

            self.dwt = DWT1DForward(wave=self.wavename, J=self.level, adapt=ADAPT, L=self.L,
                                    wt_alpha=self.wt_alpha_dwt).to(device)
            self.iwt = DWT1DInverse(wave=self.wavename, adapt=ADAPT, L=self.L,
                                    wt_alpha=self.wt_alpha_dwt).to(device)
            self.filterband_transform_list = nn.ModuleList(
                SimpleConv(self.h_dim, self.h_dim).to(device)
                for _ in range(self.level)
            )
            self.detail_transform_list = nn.ModuleList(
                SimpleConv(self.h_dim, self.h_dim).to(device)
                for _ in range(self.level)
            )
            self.selector = nn.Linear(self.h_dim, self.level).to(device)
            self.wave_gate = nn.Parameter(
                torch.tensor(float(wave_gate_init), device=device),
                requires_grad=True,
            )

    def _clone_bands(self, bands):
        return [band.clone() for band in bands]

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

        f_base = self.conv_net(h)
        if self.TASK_DWT:
            batch_size = h.shape[0]
            pooled_h = h.mean(dim=-1)
            level_weight = nn.functional.softmax(
                self.smooth * self.selector(pooled_h),
                dim=-1,
            )

            yl, yh = self.dwt(h)
            f_wave = None
            for i, (smooth_transform, detail_transform) in enumerate(
                zip(self.filterband_transform_list, self.detail_transform_list)
            ):
                smooth_yh = self._clone_bands(yh)
                smooth_yh[i] = torch.zeros_like(smooth_yh[i])
                smooth_reconstructed = self.iwt((yl, smooth_yh))

                detail_yh = [torch.zeros_like(band) for band in yh]
                detail_yh[i] = yh[i]
                detail_reconstructed = self.iwt((torch.zeros_like(yl), detail_yh))

                transformed = (
                    smooth_transform(smooth_reconstructed)
                    + detail_transform(detail_reconstructed)
                )
                weight = level_weight[:, i].reshape(batch_size, 1, 1)
                f_wave = transformed * weight if f_wave is None else f_wave + transformed * weight

            h = f_base + torch.sigmoid(self.wave_gate) * f_wave
        else:
            h = f_base

        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations.
        # mean = self.mean_layer(x_grid, h, x_out)
        # sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))
        mean = self.mean_layer(x_grid, h[:, :, :int(self.h_dim/2)], x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h[:, :, int(self.h_dim/2):], x_out))

        return mean, sigma

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
