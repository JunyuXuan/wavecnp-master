import torch.nn as nn
import torch.nn.functional as F
import pywt
# import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch
import numpy as np
import math

from wavecnp.utils import device



def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))



def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n+end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:,-n:], x[:,:-n+end]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:,:,-n:], x[:,:,:-n+end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n+end]), dim=3)


def mypad(x, pad, mode='constant', value=0):
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.

    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,xe]
        # horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1+m2, dtype='int32'), -0.5, l1-0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2+m2, dtype='int32'), -0.5, l2-0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]
    elif mode == 'periodic':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:,:,xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:,:,:,xe]
        # Both
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


class DWT1DForward(nn.Module):
    """ Performs a 1d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels1 of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero', adapt=False, L=4, wt_alpha=0):
        super().__init__()
        self.adapt = adapt
        self.L = L
        if adapt:
            self.alpha = wt_alpha
            # h0, h1, _, _ = parameterizeQMF(alpha, L)
            #
            # self.h0 = h0
            # self.h1 = h1
        else:
            if isinstance(wave, str):
                wave = pywt.Wavelet(wave)
            if isinstance(wave, pywt.Wavelet):
                h0, h1 = wave.dec_lo, wave.dec_hi
            else:
                assert len(wave) == 2
                h0, h1 = wave[0], wave[1]

            # Prepare the filters - this makes them into column filters
            filts = self.prep_filt_afb1d(h0, h1)
            self.register_buffer('h0', filts[0])
            self.register_buffer('h1', filts[1])

        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, L_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients.
        """
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        mode = mode_to_int(self.mode)

        if self.adapt:
            h0, h1 = self.parameterizeQMF(self.alpha, self.L)

            self.h0 = h0
            self.h1 = h1

        # Do a multilevel transform
        for j in range(self.J):
            x0, x1 = self.AFB1D_apply(x0, self.h0, self.h1, mode)
            highs.append(x1)

        return x0, highs

    def parameterizeQMF(self, alpha, L):
        h0 = torch.zeros(1, 1, L, device=device)
        h1 = torch.zeros(1, 1, L, device=device)

        h0[:, :, 0] = (1 - torch.cos(alpha) + ((-1) ** (0 + 1)) * torch.sin(alpha)) / (2 * math.sqrt(2))
        h0[:, :, 1] = (1 + torch.cos(alpha) + ((-1) ** (1 + 1 - 1)) * torch.sin(alpha)) / (2 * math.sqrt(2))
        h0[:, :, 2] = (1 + torch.cos(alpha) + ((-1) ** (2 + 1 - 1)) * torch.sin(alpha)) / (2 * math.sqrt(2))
        h0[:, :, 3] = (1 - torch.cos(alpha) + ((-1) ** (3 + 1)) * torch.sin(alpha)) / (2 * math.sqrt(2))

        h1[:, :, 0] = - h0[:, :, 3]
        h1[:, :, 1] = h0[:, :, 2]
        h1[:, :, 2] = - h0[:, :, 1]
        h1[:, :, 3] = h0[:, :, 0]

        # h0 = [-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416]
        # h1 = [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037]
        # g0 = [0.48296291314453416, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037]
        # g1 = [-0.12940952255126037, -0.2241438680420134, 0.8365163037378079, -0.48296291314453416]

        return h0, h1

    def AFB1D_apply(self, x, h0, h1, mode):
        mode = int_to_mode(mode)

        # Make inputs 4d
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        lohi = self.afb1d(x, h0, h1, mode=mode, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1

    def afb1d(self, x, h0, h1, mode='zero', dim=-1):
        """ 1D analysis filter bank (along one dimension only) of an image

        Inputs:
            x (tensor): 4D input with the last two dimensions the spatial input
            h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
                h, 1) or (1, 1, 1, w)
            h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
                h, 1) or (1, 1, 1, w)
            mode (str): padding method
            dim (int) - dimension of filtering. d=2 is for a vertical filter (called
                column filtering but filters across the rows). d=3 is for a
                horizontal filter, (called row filtering but filters across the
                columns).

        Returns:
            lohi: lowpass and highpass subbands concatenated along the channel
                dimension
        """
        C = x.shape[1]
        # Convert the dim to positive
        d = dim % 4
        s = (2, 1) if d == 2 else (1, 2)
        N = x.shape[d]
        # If h0, h1 are not tensors, make them. If they are, then assume that they
        # are in the right order
        if not isinstance(h0, torch.Tensor):
            h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                              dtype=torch.float, device=x.device)
        if not isinstance(h1, torch.Tensor):
            h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                              dtype=torch.float, device=x.device)
        L = h0.numel()
        L2 = L // 2
        shape = [1, 1, 1, 1]
        shape[d] = L
        # If h aren't in the right shape, make them so
        if h0.shape != tuple(shape):
            h0 = h0.reshape(*shape)
        if h1.shape != tuple(shape):
            h1 = h1.reshape(*shape)
        h = torch.cat([h0, h1] * C, dim=0)

        if mode == 'per' or mode == 'periodization':
            if x.shape[dim] % 2 == 1:
                if d == 2:
                    x = torch.cat((x, x[:, :, -1:]), dim=2)
                else:
                    x = torch.cat((x, x[:, :, :, -1:]), dim=3)
                N += 1
            x = roll(x, -L2, dim=d)
            pad = (L - 1, 0) if d == 2 else (0, L - 1)
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
            N2 = N // 2
            if d == 2:
                lohi[:, :, :L2] = lohi[:, :, :L2] + lohi[:, :, N2:N2 + L2]
                lohi = lohi[:, :, :N2]
            else:
                lohi[:, :, :, :L2] = lohi[:, :, :, :L2] + lohi[:, :, :, N2:N2 + L2]
                lohi = lohi[:, :, :, :N2]
        else:
            # Calculate the pad size
            outsize = pywt.dwt_coeff_len(N, L, mode=mode)
            p = 2 * (outsize - 1) - N + L
            if mode == 'zero':
                # Sadly, pytorch only allows for same padding before and after, if
                # we need to do more padding after for odd length signals, have to
                # prepad
                if p % 2 == 1:
                    pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                    x = F.pad(x, pad)
                pad = (p // 2, 0) if d == 2 else (0, p // 2)
                # Calculate the high and lowpass
                # print("x.device = ", x.device())
                # print("h.device = ", h.device())
                lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
            elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
                pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
                x = mypad(x, pad=pad, mode=mode)
                lohi = F.conv2d(x, h, stride=s, groups=C)
            else:
                raise ValueError("Unkown pad type: {}".format(mode))

        return lohi

    def prep_filt_afb1d(self, h0, h1, device=None):
        """
        Prepares the filters to be of the right form for the afb2d function.  In
        particular, makes the tensors the right shape. It takes mirror images of
        them as as afb2d uses conv2d which acts like normal correlation.

        Inputs:
            h0 (array-like): low pass column filter bank
            h1 (array-like): high pass column filter bank
            device: which device to put the tensors on to

        Returns:
            (h0, h1)
        """
        h0 = np.array(h0[::-1]).ravel()
        h1 = np.array(h1[::-1]).ravel()
        t = torch.get_default_dtype()
        h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
        h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
        return h0, h1


class DWT1DInverse(nn.Module):
    """ Performs a 1d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """
    def __init__(self, wave='db1', mode='zero', adapt=False, L=4, wt_alpha=0):
        super().__init__()
        self.adapt = adapt
        self.L = L
        if adapt:
            self.alpha = wt_alpha
            # _, _, g0, g1 = parameterizeQMF(alpha, L)
            #
            # self.g0 = g0
            # self.g1 = g1
        else:
            if isinstance(wave, str):
                wave = pywt.Wavelet(wave)
            if isinstance(wave, pywt.Wavelet):
                g0, g1 = wave.rec_lo, wave.rec_hi
            else:
                assert len(wave) == 2
                g0, g1 = wave[0], wave[1]

            # Prepare the filters
            filts = self.prep_filt_sfb1d(g0, g1)
            self.register_buffer('g0', filts[0])
            self.register_buffer('g1', filts[1])

        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, should
              match the format returned by DWT1DForward.

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, L_{in})`

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """

        if self.adapt:
            g0, g1 = self.parameterizeQMF(self.alpha, self.L)

            self.g0 = g0
            self.g1 = g1

        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        mode = mode_to_int(self.mode)
        # Do a multilevel inverse transform
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)

            # 'Unpad' added signal
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = self.SFB1D_apply(x0, x1, self.g0, self.g1, mode)
        return x0

    def parameterizeQMF(self, alpha, L):
        g0 = torch.zeros(1, 1, L, device=device)
        g1 = torch.zeros(1, 1, L, device=device)

        g1[:, :, 0] = (1 - torch.cos(alpha) + ((-1) ** (0 + 1)) * torch.sin(alpha)) / (2 * math.sqrt(2))
        g1[:, :, 1] = - (1 + torch.cos(alpha) + ((-1) ** (1 + 1 - 1)) * torch.sin(alpha)) / (2 * math.sqrt(2))
        g1[:, :, 2] = (1 + torch.cos(alpha) + ((-1) ** (2 + 1 - 1)) * torch.sin(alpha)) / (2 * math.sqrt(2))
        g1[:, :, 3] = - (1 - torch.cos(alpha) + ((-1) ** (3 + 1)) * torch.sin(alpha)) / (2 * math.sqrt(2))

        g0[:, :, 0] = - g1[:, :, 3]
        g0[:, :, 1] = g1[:, :, 2]
        g0[:, :, 2] = - g1[:, :, 1]
        g0[:, :, 3] = g1[:, :, 0]

        # h0 = [-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416]
        # h1 = [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037]
        # g0 = [0.48296291314453416, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037]
        # g1 = [-0.12940952255126037, -0.2241438680420134, 0.8365163037378079, -0.48296291314453416]

        return g0, g1

    def SFB1D_apply(self, low, high, g0, g1, mode):
        mode = int_to_mode(mode)
        # Make into a 2d tensor with 1 row
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        return self.sfb1d(low, high, g0, g1, mode=mode, dim=3)[:, :, 0]

    def sfb1d(self, lo, hi, g0, g1, mode='zero', dim=-1):
        """ 1D synthesis filter bank of an image tensor
        """
        C = lo.shape[1]
        d = dim % 4
        # If g0, g1 are not tensors, make them. If they are, then assume that they
        # are in the right order
        if not isinstance(g0, torch.Tensor):
            g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                              dtype=torch.float, device=lo.device)
        if not isinstance(g1, torch.Tensor):
            g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                              dtype=torch.float, device=lo.device)
        L = g0.numel()
        shape = [1, 1, 1, 1]
        shape[d] = L
        N = 2 * lo.shape[d]
        # If g aren't in the right shape, make them so
        if g0.shape != tuple(shape):
            g0 = g0.reshape(*shape)
        if g1.shape != tuple(shape):
            g1 = g1.reshape(*shape)

        s = (2, 1) if d == 2 else (1, 2)
        g0 = torch.cat([g0] * C, dim=0)
        g1 = torch.cat([g1] * C, dim=0)
        if mode == 'per' or mode == 'periodization':
            y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, groups=C)
            if d == 2:
                y[:, :, :L - 2] = y[:, :, :L - 2] + y[:, :, N:N + L - 2]
                y = y[:, :, :N]
            else:
                y[:, :, :, :L - 2] = y[:, :, :, :L - 2] + y[:, :, :, N:N + L - 2]
                y = y[:, :, :, :N]
            y = roll(y, 1 - L // 2, dim=dim)
        else:
            if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                    mode == 'periodic':
                pad = (L - 2, 0) if d == 2 else (0, L - 2)
                y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                    F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
            else:
                raise ValueError("Unkown pad type: {}".format(mode))

        return y

    def prep_filt_sfb1d(self, g0, g1, device=None):
        """
        Prepares the filters to be of the right form for the sfb1d function. In
        particular, makes the tensors the right shape. It does not mirror image them
        as as sfb2d uses conv2d_transpose which acts like normal convolution.

        Inputs:
            g0 (array-like): low pass filter bank
            g1 (array-like): high pass filter bank
            device: which device to put the tensors on to

        Returns:
            (g0, g1)
        """
        g0 = np.array(g0).ravel()
        g1 = np.array(g1).ravel()
        t = torch.get_default_dtype()
        g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
        g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))

        return g0, g1
