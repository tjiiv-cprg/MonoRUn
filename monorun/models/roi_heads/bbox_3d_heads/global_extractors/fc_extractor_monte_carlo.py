import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import auto_fp16
from mmdet.models.builder import HEADS
from .fc_extractor import FCExtractor


class Dropout2d(nn.Dropout2d):
    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)


class Dropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


@HEADS.register_module()
class FCExtractorMonteCarlo(FCExtractor):

    def __init__(self,
                 num_samples=50,
                 dropout_rate=0.5,
                 dropout2d_rate=0.2,
                 **kwargs):
        super(FCExtractorMonteCarlo, self).__init__(
            dropout_rate=dropout_rate,
            dropout2d_rate=dropout2d_rate,
            **kwargs)
        self.num_samples = num_samples
        self.use_dropout = dropout_rate > 0
        self.use_dropout2d = dropout2d_rate > 0
        assert self.use_dropout
        self.dropout = Dropout(dropout_rate)
        assert self.use_dropout2d
        self.dropout2d = Dropout2d(dropout2d_rate)

    @auto_fp16()
    def forward(self, x):
        if not self.training:
            x = x.repeat(self.num_samples, 1, 1, 1)
        dim_latent_pred, dim_latent_var, distance_pred, distance_logstd, x = \
            super().forward(x)

        if not self.training:
            num_actual_classes = 1 if self.latent_class_agnostic \
                else self.num_classes
            dim_latent_pred = dim_latent_pred.view(
                self.num_samples, -1,
                (self.dim_dim + self.latent_channels) * num_actual_classes)
            dim_latent_var, dim_latent_pred = torch.var_mean(dim_latent_pred, dim=0)

            distance_pred = distance_logstd = None

            x = x.view(self.num_samples, -1, x.size(1))
            x = torch.mean(x, dim=0)

        return dim_latent_pred, dim_latent_var, distance_pred, distance_logstd, x

    def slice_pred(self, dim_latent_pred, dim_latent_var, labels):
        if not self.training:
            if self.latent_class_agnostic:
                pos_dim_latent_pred = dim_latent_pred
                pos_dim_latent_var = dim_latent_var
            else:
                inds = torch.arange(len(labels), device=labels.device)
                pos_dim_latent_pred = dim_latent_pred.view(
                    dim_latent_pred.size(0), -1, self.dim_dim + self.latent_channels
                )[inds, labels]
                pos_dim_latent_var = dim_latent_var.view(
                    dim_latent_var.size(0), -1, self.dim_dim + self.latent_channels
                )[inds, labels] if dim_latent_var is not None else None
            dim_pred, latent_pred = pos_dim_latent_pred.split(
                [self.dim_dim, self.latent_channels], dim=1)
            dim_var, latent_var = pos_dim_latent_var.split(
                [self.dim_dim, self.latent_channels], dim=1)
            return dim_pred, dim_var, latent_pred, latent_var
        else:
            return super().slice_pred(
                dim_latent_pred, dim_latent_var, labels)
