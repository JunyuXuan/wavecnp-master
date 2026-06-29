import torch.nn as nn

from wavecnp.utils import (
    init_sequential_weights,
    init_layer_weights,
    weights_init,
    pad_concat,
    channels_to_2nd_dim,
    channels_to_last_dim,
    make_depth_sep_conv
)

from .modules import LieConv
from .liegroups import T

__all__ = ['SimpleMLP', 'SimpleConv', 'UNet', 'ResConvBlock', 'CNN']


class SimpleMLP(nn.Module):

    def __init__(self, in_channels=8):
        super(SimpleMLP, self).__init__()
        self.hidden_size = in_channels
        self.transform_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, self.hidden_size*4),
            nn.ReLU(),
            nn.Linear(self.hidden_size*4, self.hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )
        init_sequential_weights(self.transform_net)

    def forward(self, x):
        """Forward pass through the structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.transform_net(x)


class SimpleConv(nn.Module):
    """Small convolutional architecture from 1d experiments in the paper.
    This is a 4-layer convolutional network with fixed stride and channels,
    using ReLU activations.

    Args:
        in_channels (int, optional): Number of channels on the input to the
            network. Defaults to 8.
        out_channels (int, optional): Number of channels on the output by the
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=self.out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        init_sequential_weights(self.conv_net)
        self.num_halving_layers = 0

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.conv_net(x)



class SimpleLieConv(nn.Module):
    """Small convolutional architecture from 1d experiments in the paper.
    This is a 4-layer convolutional network with fixed stride and channels,
    using ReLU activations.

    Args:
        in_channels (int, optional): Number of channels on the input to the
            network. Defaults to 8.
        out_channels (int, optional): Number of channels on the output by the
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleLieConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.ReLU()
        num = 5
        group=T(1)
        self.conv_net = nn.Sequential(
            LieConv(in_channels=self.in_channels, out_channels=16,
                      num_nbhd=5, fill=num/64, sampling_fraction=1., group=group, use_bn=True, mean=True, coeff=0.3),
            nn.ReLU(),
            LieConv(in_channels=16, out_channels=32,
                      num_nbhd=5, fill=num/64, sampling_fraction=1., group=group, use_bn=True, mean=True, coeff=0.3),
            nn.ReLU(),
            LieConv(in_channels=32, out_channels=16,
                      num_nbhd=5, fill=num/64, sampling_fraction=1., group=group, use_bn=True, mean=True, coeff=0.3),
            nn.ReLU(),
            LieConv(in_channels=16, out_channels=self.out_channels,
                      num_nbhd=5, fill=num/64, sampling_fraction=1., group=group, use_bn=True, mean=True, coeff=0.3),
            nn.ReLU(),
        )
        init_sequential_weights(self.conv_net)
        self.num_halving_layers = 0

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.conv_net(x)



class SimpleConv1(nn.Module):
    """Small convolutional architecture from 1d experiments in the paper.
    This is a 2-layer convolutional network with fixed stride and channels,
    using ReLU activations.

    Args:
        in_channels (int, optional): Number of channels on the input to the
            network. Defaults to 8.
        out_channels (int, optional): Number of channels on the output by the
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleConv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=16, out_channels=32,
            #           kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=16,
            #           kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=self.out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        init_sequential_weights(self.conv_net)
        self.num_halving_layers = 0

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.conv_net(x)


class SimpleConv3(nn.Module):
    """Small convolutional architecture from 1d experiments in the paper.
    This is a 6-layer convolutional network with fixed stride and channels,
    using ReLU activations.

    Args:
        in_channels (int, optional): Number of channels on the input to the
            network. Defaults to 8.
        out_channels (int, optional): Number of channels on the output by the
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleConv3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=self.out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        init_sequential_weights(self.conv_net)
        self.num_halving_layers = 0

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.conv_net(x)


class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.

    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = 8
        self.num_halving_layers = 6

        self.l1 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l2 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=2 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=2 * self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)


class ResConvBlock(nn.Module):
    """Convolutional block inspired by the pre-activation Resnet [1]
    and depthwise separable convolutions [2].

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel. Should be odd to keep the same size.

    activation: callable, optional
        Activation object. E.g. `nn.RelU()`.

    Normalization : nn.Module, optional
        Normalization layer (unitialized). E.g. `nn.BatchNorm1d`.

    n_conv_layers : int, optional
        Number of convolutional layers, can be 1 or 2.

    is_bias : bool, optional
        Whether to use a bias.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        is_bias=True,
        n_conv_layers=1,
    ):
        super().__init__()
        self.activation = activation
        self.n_conv_layers = n_conv_layers
        assert self.n_conv_layers in [1, 2]

        if kernel_size % 2 == 0:
            raise ValueError("`kernel_size={}`, but should be odd.".format(kernel_size))

        padding = kernel_size // 2

        if self.n_conv_layers == 2:
            self.norm1 = Normalization(in_chan)
            self.conv1 = make_depth_sep_conv(Conv)(
                in_chan, in_chan, kernel_size, padding=padding, bias=is_bias
            )
        self.norm2 = Normalization(in_chan)
        self.conv2_depthwise = Conv(
            in_chan, in_chan, kernel_size, padding=padding, groups=in_chan, bias=is_bias
        )
        self.conv2_pointwise = Conv(in_chan, out_chan, 1, bias=is_bias)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):

        if self.n_conv_layers == 2:
            out = self.conv1(self.activation(self.norm1(X)))
        else:
            out = X

        out = self.conv2_depthwise(self.activation(self.norm2(out)))
        # adds residual before point wise => output can change number of channels
        out = out + X
        out = self.conv2_pointwise(out.contiguous())  # for some reason need contiguous
        return out


class CNN(nn.Module):
    """Simple multilayer CNN.

    Parameters
    ----------
    n_channels : int or list
        Number of channels, same for input and output. If list then needs to be
        of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a
        `[ConvBlock(16,32), ConvBlock(32, 64)]`.

    ConvBlock : nn.Module
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.

    n_blocks : int, optional
        Number of convolutional blocks.

    is_chan_last : bool, optional
        Whether the channels are on the last dimension of the input.

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(self, n_channels, ConvBlock, n_blocks=3, is_chan_last=False, **kwargs):

        super().__init__()
        self.n_blocks = n_blocks
        self.is_chan_last = is_chan_last
        self.in_out_channels = self._get_in_out_channels(n_channels, n_blocks)
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_chan, out_chan, **kwargs)
                for in_chan, out_chan in self.in_out_channels
            ]
        )
        self.is_return_rep = False  # never return representation for vanilla conv

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels."""
        if isinstance(n_channels, int):
            channel_list = [n_channels] * (n_blocks + 1)
        else:
            channel_list = list(n_channels)

        assert len(channel_list) == (n_blocks + 1), "{} != {}".format(
            len(channel_list), n_blocks + 1
        )

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, X):
        if self.is_chan_last:
            X = channels_to_2nd_dim(X)

        X, representation = self.apply_convs(X)

        if self.is_chan_last:
            X = channels_to_last_dim(X)

        if self.is_return_rep:
            return X, None

        return X

    def apply_convs(self, X):
        for conv_block in self.conv_blocks:
            X = conv_block(X)
        return X, None
