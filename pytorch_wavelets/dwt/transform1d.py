import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch
import numpy as np
import math

from wavecnp.utils import device

def parameterizeQMF(alpha, L):
    h0 = torch.zeros(1, 1, L, device=device)
    h1 = torch.zeros(1, 1, L, device=device)
    g0 = torch.zeros(1, 1, L, device=device)
    g1 = torch.zeros(1, 1, L, device=device)

    h0[:, :, 0] = (1 - math.cos(alpha) + ((-1) ** (0+1)) * math.sin(alpha)) / (2 * math.sqrt(2))
    h0[:, :, 1] = (1 + math.cos(alpha) + ((-1) ** (1+1 - 1)) * math.sin(alpha)) / (2 * math.sqrt(2))
    h0[:, :, 2] = (1 + math.cos(alpha) + ((-1) ** (2+1 - 1)) * math.sin(alpha)) / (2 * math.sqrt(2))
    h0[:, :, 3] = (1 - math.cos(alpha) + ((-1) ** (3+1)) * math.sin(alpha)) / (2 * math.sqrt(2))

    h1[:, :, 0] = - h0[:, :, 3]
    h1[:, :, 1] = h0[:, :, 2]
    h1[:, :, 2] = - h0[:, :, 1]
    h1[:, :, 3] = h0[:, :, 0]

    g0[:, :, 0] = - h1[:, :, 0]
    g0[:, :, 1] = h1[:, :, 1]
    g0[:, :, 2] = - h1[:, :, 2]
    g0[:, :, 3] = h1[:, :, 3]

    g1[:, :, 0] = h0[:, :, 0]
    g1[:, :, 1] = - h0[:, :, 1]
    g1[:, :, 2] = h0[:, :, 2]
    g1[:, :, 3] = - h0[:, :, 3]

    # h0 = [-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416]
    # h1 = [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037]
    # g0 = [0.48296291314453416, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037]
    # g1 = [-0.12940952255126037, -0.2241438680420134, 0.8365163037378079, -0.48296291314453416]

    return h0, h1, g0, g1

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
    def __init__(self, J=1, wave='db1', mode='zero', adapt=False, L=4):
        super().__init__()
        if adapt:
            alpha = nn.Parameter(torch.tensor(math.pi/3.0), requires_grad=True)
            h0, h1, _, _ = parameterizeQMF(alpha, L)

            self.h0 = h0
            self.h1 = h1
        else:
            if isinstance(wave, str):
                wave = pywt.Wavelet(wave)
            if isinstance(wave, pywt.Wavelet):
                h0, h1 = wave.dec_lo, wave.dec_hi
            else:
                assert len(wave) == 2
                h0, h1 = wave[0], wave[1]

            # Prepare the filters - this makes them into column filters
            filts = lowlevel.prep_filt_afb1d(h0, h1)
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
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            x0, x1 = lowlevel.AFB1D.apply(x0, self.h0, self.h1, mode)
            highs.append(x1)

        return x0, highs


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
    def __init__(self, wave='db1', mode='zero', adapt=False, L=4):
        super().__init__()
        if adapt:
            alpha = nn.Parameter(torch.tensor(math.pi / 3.0), requires_grad=True)
            _, _, g0, g1 = parameterizeQMF(alpha, L)

            self.g0 = g0
            self.g1 = g1
        else:
            if isinstance(wave, str):
                wave = pywt.Wavelet(wave)
            if isinstance(wave, pywt.Wavelet):
                g0, g1 = wave.rec_lo, wave.rec_hi
            else:
                assert len(wave) == 2
                g0, g1 = wave[0], wave[1]

            # Prepare the filters
            filts = lowlevel.prep_filt_sfb1d(g0, g1)
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
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        mode = lowlevel.mode_to_int(self.mode)
        # Do a multilevel inverse transform
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)

            # 'Unpad' added signal
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = lowlevel.SFB1D.apply(x0, x1, self.g0, self.g1, mode)
        return x0
