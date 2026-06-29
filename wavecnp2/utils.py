import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import sys

__all__ = ['device',
           'resolve_device',
           'set_device',
           'to_multiple',
           'BatchLinear',
           'init_layer_weights',
           'init_sequential_weights',
           'compute_dists',
           'pad_concat',
           'gaussian_logpdf',
           'channels_to_2nd_dim',
           'channels_to_last_dim',
           'make_abs_conv',
           'weights_init',
           'make_depth_sep_conv']

def resolve_device(requested='auto'):
    """Resolve a device string. GPU is preferred by default when available."""
    requested = str(requested or 'auto').strip().lower()
    if requested.isdigit():
        requested = f'cuda:{requested}'

    if requested in {'auto', 'gpu', 'cuda'}:
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        if requested in {'gpu', 'cuda'}:
            print('CUDA was requested but is not available. Falling back to CPU.')
        return torch.device('cpu')

    selected = torch.device(requested)
    if selected.type == 'cuda':
        if not torch.cuda.is_available():
            print(f'{selected} was requested but CUDA is not available. Falling back to CPU.')
            return torch.device('cpu')
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise ValueError(f'{selected} is not available. CUDA device count: {torch.cuda.device_count()}.')
    return selected


def set_device(requested='auto', update_modules=True, verbose=True):
    """Set the shared runtime device used by WaveCNP helper modules."""
    global device
    device = resolve_device(requested)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if update_modules:
        for module in list(sys.modules.values()):
            if isinstance(getattr(module, 'device', None), torch.device):
                module.device = device

    if verbose:
        if device.type == 'cuda':
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")
    return device


device = set_device('auto', update_modules=False, verbose=False)


"""Device perform computations on."""


class Named(type):
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


def square_distance(src, target):
    """Calculate Euclid Distancve between two points: src and dst.

    Args:
        src: source points, (B, N, D)
        target: target points, (B, M, D)

    Returns:
        dist: per-point square distance, (B, N, M)

    """
    if src.shape[1] == 1 or target.shape[1] == 1:
        return (src - target).pow(2).sum(-1)
    B, N, _ = src.shape
    _, M, _ = target.shape
    dist = -2 * src.matmul(target.permute(0, 2, 1))
    dist += src.pow(2).sum(-1).reshape(B, N, 1)
    dist += target.pow(2).sum(-1).reshape(B, 1, M)
    return dist


def index_points(points, idx):
    """Get points［idx]

    Args:
        points: input point-cloud data, (B, N, D)
        idx: sampled data index, (B, S)

    Returns:
        new_points: indexed_points data, (B, S, D)

    """
    device = points.device
    B = points.size(0)
    index_shape = list(idx.shape)  # [B, S]
    index_shape[1:] = [1] * (len(index_shape) - 1)  # [B, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=device, dtype=torch.long).reshape(index_shape).expand(repeat_shape)
    new_points = points[batch_indices, idx, ...]
    return new_points



def knn_points(nbhd: int, all_coords, query_coords, mask, distance=square_distance):
    """Point-cloud k-nearest neighborhood

    Args:
        nbhd: max sample number in local region
        coords: all data in Batch (B, N, D)
        query_coords: query data in Batch (B, M, D)
        mask: valid mask (B, N)

    Returns:
        group_idx, (B, M, nbhd)

    """
    dist = distance(query_coords.unsqueeze(-2), all_coords.unsqueeze(-3))  # [B, M, N]
    dist[~mask[:, None, :].expand(*dist.shape)] = 1e8
    _, group_idx = torch.topk(dist, nbhd, dim=-1, largest=False, sorted=False)
    return group_idx


class Metric(object):
    def __init__(self):
        self.total = 0
        self.trials = 0

    def log(self, score, trial):
        self.total += score * trial
        self.trials += trial

    @property
    def average(self):
        return self.total / self.trials


def make_depth_sep_conv(Conv):
    """Make a convolution module depth separable."""

    class DepthSepConv(nn.Module):
        """Make a convolution depth separable.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        kernel_size : int

        **kwargs :
            Additional arguments to `Conv`
        """

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            confidence=False,
            bias=True,
            **kwargs
        ):
            super().__init__()
            self.depthwise = Conv(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                bias=bias,
                **kwargs
            )
            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)
            self.reset_parameters()

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

        def reset_parameters(self):
            weights_init(self)

    return DepthSepConv



def weights_init(module, **kwargs):
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    module.is_resetted = True
    for m in module.modules():
        try:
            if hasattr(module, "reset_parameters") and module.is_resetted:
                # don't reset if resetted already (might want special)
                continue
        except AttributeError:
            pass

        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            nn.init.kaiming_normal_(m.weight, mode="fan_out", **kwargs)
        elif isinstance(m, nn.Linear):
            linear_init(m, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



def linear_init(module, activation="relu"):
    """Initialize a linear layer.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight

    if module.bias is not None:
        module.bias.data.zero_()

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity="relu")
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def make_abs_conv(Conv):
    """Make a convolution have only positive parameters."""

    class AbsConv(Conv):
        def forward(self, input):
            return F.conv2d(
                input,
                # self.weight,
                self.weight.abs(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv


def channels_to_last_dim(X):
    """
    Takes a signal with channels on the second dimension (for convolutions) and
    returns it with channels on the last dimension (for most operations).
    """
    return X.permute(*([0] + list(range(2, X.dim())) + [1]))


def channels_to_2nd_dim(X):
    """
    Takes a signal with channels on the last dimension (for most operations) and
    returns it with channels on the second dimension (for convolutions).
    """
    return X.permute(*([0, X.dim() - 1] + list(range(1, X.dim() - 1))))


def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


class BatchLinear(nn.Linear):
    """Helper class for linear layers on order-3 tensors.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Use a bias. Defaults to `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BatchLinear, self).__init__(in_features=in_features,
                                          out_features=out_features,
                                          bias=bias)
        nn.init.xavier_normal_(self.weight, gain=1)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        """Forward pass through layer. First unroll batch dimension, then pass
        through dense layer, and finally reshape back to a order-3 tensor.

        Args:
              x (tensor): Inputs of shape `(batch, n, in_features)`.

        Returns:
              tensor: Outputs of shape `(batch, n, out_features)`.
        """
        num_functions, num_inputs = x.shape[0], x.shape[1]
        x = x.reshape(num_functions * num_inputs, self.in_features)
        out = super(BatchLinear, self).forward(x)
        return out.view(num_functions, num_inputs, self.out_features)



def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.

    Args:
        layer (:class:`nn.Module`): Single dense or convolutional layer from
            :mod:`torch.nn`.

    Returns:
        :class:`nn.Module`: Single dense or convolutional layer with
            initialized weights.
    """
    nn.init.xavier_normal_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 1e-3)


def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.

    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model


def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape `(batch, n, 1)`.
        y (tensor): Inputs of shape `(batch, m, 1)`.

    Returns:
        tensor: Pair-wise distances of shape `(batch, n, m)`.
    """
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2


def pad_concat(t1, t2):
    """Concat the activations of two layer channel-wise by padding the layer
    with fewer points with zeros.

    Args:
        t1 (tensor): Activations from first layers of shape `(batch, n1, c1)`.
        t2 (tensor): Activations from second layers of shape `(batch, n2, c2)`.

    Returns:
        tensor: Concatenated activations of both layers of shape
            `(batch, max(n1, n2), c1 + c2)`.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:  # Even difference
            t2 = F.pad(t2, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t2 = F.pad(t2, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:  # Even difference
            t1 = F.pad(t1, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t1 = F.pad(t1, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')

    return torch.cat([t1, t2], dim=1)


def gaussian_logpdf(inputs, mean, sigma, reduction=None):
    """Gaussian log-density.

    Args:
        inputs (tensor): Inputs.
        mean (tensor): Mean.
        sigma (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    dist = Normal(loc=mean, scale=sigma)
    logp = dist.log_prob(inputs)

    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')



class StandardDecoder(nn.Module):
    """Decoder used for standard CNP model.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the output.
    """

    def __init__(self, input_dim, latent_dim, output_dim):
        super(StandardDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        post_pooling_fn = nn.Sequential(
            BatchLinear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            BatchLinear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            BatchLinear(self.latent_dim, 2 * self.output_dim),
        )
        self.post_pooling_fn = init_sequential_weights(post_pooling_fn)
        self.sigma_fn = nn.functional.softplus
        self.sigmoid_fn = torch.nn.functional.sigmoid

    def forward(self, x):
        """Forward pass through the decoder.

        Args:
            x (tensor): Target locations of shape
                `(batch, num_targets, input_dim)`.

        Returns:
            tensor: Output values at each query point of shape
                `(batch, num_targets, output_dim)`
        """
        z = self.post_pooling_fn(x)

        # Separate mean and standard deviations and return.
        mean = z[..., :self.output_dim]
        # mean = self.sigmoid_fn(z[..., :self.output_dim])
        sigma = self.sigma_fn(z[..., self.output_dim:])

        return mean, sigma

class StandardDecoder2(nn.Module):
    """Decoder used for standard CNP model.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the output.
    """

    def __init__(self, input_dim, latent_dim, output_dim):
        super(StandardDecoder2, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        post_pooling_fn = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 2 * self.output_dim),
        )
        self.post_pooling_fn = init_sequential_weights(post_pooling_fn)
        self.sigma_fn = nn.functional.softplus
        self.sigmoid_fn = torch.nn.functional.sigmoid

    def forward(self, x):
        """Forward pass through the decoder.

        Args:
            x (tensor): Target locations of shape
                `(batch, num_targets, input_dim)`.

        Returns:
            tensor: Output values at each query point of shape
                `(batch, num_targets, output_dim)`
        """
        z = self.post_pooling_fn(x)

        # Separate mean and standard deviations and return.
        mean = z[..., :self.output_dim]
        # mean = self.sigmoid_fn(z[..., :self.output_dim])
        sigma = self.sigma_fn(z[..., self.output_dim:])

        return mean, sigma



class SimpleStandardDecoder(nn.Module):
    """Decoder used for standard CNP model.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the output.
    """

    def __init__(self, input_dim, latent_dim, output_dim):
        super(SimpleStandardDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        post_pooling_fn = nn.Sequential(
            BatchLinear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            BatchLinear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            BatchLinear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            BatchLinear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            BatchLinear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            BatchLinear(self.latent_dim, 2 * self.output_dim),
        )
        self.post_pooling_fn = init_sequential_weights(post_pooling_fn)
        self.sigma_fn = nn.functional.softplus
        self.sigmoid_fn = torch.nn.functional.sigmoid

    def forward(self, x):
        """Forward pass through the decoder.

        Args:
            x (tensor): Target locations of shape
                `(batch, num_targets, input_dim)`.

        Returns:
            tensor: Output values at each query point of shape
                `(batch, num_targets, output_dim)`
        """
        z = self.post_pooling_fn(x)

        # Separate mean and standard deviations and return.
        mean = z[..., :self.output_dim]
        # mean = self.sigmoid_fn(z[..., :self.output_dim])
        sigma = self.sigma_fn(z[..., self.output_dim:])

        return mean, sigma



class StdStandardDecoder(nn.Module):
    """Decoder used for standard CNP model.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the output.
    """

    def __init__(self, input_dim, latent_dim, output_dim):
        super(StdStandardDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        post_pooling_fn = nn.Sequential(
            BatchLinear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            BatchLinear(self.latent_dim, self.output_dim),
        )
        self.post_pooling_fn = init_sequential_weights(post_pooling_fn)
        self.sigma_fn = nn.functional.softplus
        self.sigmoid_fn = torch.nn.functional.sigmoid

    def forward(self, x):
        """Forward pass through the decoder.

        Args:
            x (tensor): Target locations of shape
                `(batch, num_targets, input_dim)`.

        Returns:
            tensor: Output values at each query point of shape
                `(batch, num_targets, output_dim)`
        """
        z = self.post_pooling_fn(x)

        sigma = self.sigma_fn(z[..., self.output_dim:])

        return sigma
