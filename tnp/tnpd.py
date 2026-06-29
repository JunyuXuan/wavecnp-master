import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
# from attrdict import AttrDict
from addict import Dict
import numpy as np

from tnp.tnp import TNP

from wavecnp.utils import device

# device_str = 'cpu'

class TNPD(TNP):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        bound_std=False
    ):
        super(TNPD, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            bound_std
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

        self.sigma_fn = nn.Softplus()

    def forward(self, batch, reduce_ll=True):
        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            # std = torch.exp(std)
            std = self.sigma_fn(std)

        # pred_tar = Normal(mean, std)

        # outs = AttrDict()
        # if reduce_ll:
        #     outs.tar_ll = pred_tar.log_prob(batch.y_target).sum(-1).mean()
        # else:
        #     outs.tar_ll = pred_tar.log_prob(batch.y_target).sum(-1)
        # outs.loss = - (outs.tar_ll)

        return mean, std

    def predict(self, xc, yc, xt):
        batch = Dict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device=device)

        z_target = self.encode(batch, autoreg=False)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])