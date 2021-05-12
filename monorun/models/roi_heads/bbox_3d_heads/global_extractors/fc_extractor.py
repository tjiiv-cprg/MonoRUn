import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss

from .....core import build_dim_coder


@HEADS.register_module()
class FCExtractor(nn.Module):

    def __init__(self,
                 with_dim=True,
                 with_latent_vec=True,
                 latent_channels=16,
                 num_fcs=2,
                 in_channels=256,
                 fc_out_channels=1024,
                 num_classes=3,
                 roi_feat_size=7,
                 latent_class_agnostic=False,
                 loss_dim=dict(
                     type='SmoothL1Loss', loss_weight=1.0, beta=1.0),
                 dim_coder=dict(
                     type='MultiClassNormCoder',
                     target_means=[
                         (3.89, 1.53, 1.62),  # car
                         (0.82, 1.78, 0.63),  # pedestrian
                         (1.77, 1.72, 0.57)],  # cyclist
                     target_stds=[
                         (0.44, 0.14, 0.11),
                         (0.25, 0.13, 0.12),
                         (0.15, 0.10, 0.14)]),
                 dropout_rate=0.5,
                 dropout2d_rate=0.2,
                 num_dropout_layers=2):
        super(FCExtractor, self).__init__()

        self.with_dim = with_dim
        self.with_latent_vec = with_latent_vec
        self.dim_dim = 3
        self.latent_channels = latent_channels if self.with_latent_vec else 0
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.latent_class_agnostic = latent_class_agnostic
        self.loss_dim = build_loss(loss_dim) if self.with_dim else None
        self.dim_coder = build_dim_coder(dim_coder)
        self.relu = nn.ReLU(inplace=True)
        self.fp16_enabled = False
        self.use_dropout = dropout_rate > 0
        self.use_dropout2d = dropout2d_rate > 0
        self.num_dropout_layers = num_dropout_layers

        assert num_fcs > 0
        self.fcs = nn.ModuleList()
        last_layer_dim = self.in_channels * self.roi_feat_area
        for i in range(num_fcs):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.fcs.append(
                nn.Linear(fc_in_channels, self.fc_out_channels))
        last_layer_dim = self.fc_out_channels

        out_dim_reg = self.dim_dim + self.latent_channels
        if not self.latent_class_agnostic:
            out_dim_reg *= self.num_classes
        self.fc_reg = nn.Linear(last_layer_dim, out_dim_reg)

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        if self.use_dropout2d:
            self.dropout2d = nn.Dropout2d(dropout2d_rate)

    def init_weights(self):
        for m in self.fcs.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.33)
                nn.init.normal_(m.bias, mean=0.02, std=0.04)
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.use_dropout2d:
            x = self.dropout2d(x)
        x = x.flatten(1)
        for i, fc in enumerate(self.fcs):
            x = self.relu(fc(x))
            if self.use_dropout and i < self.num_dropout_layers:
                x = self.dropout(x)

        dim_latent_pred = self.fc_reg(x)  # (n, 4) or (n, 4 * Nclass)

        dim_latent_var = distance_pred = distance_logstd = None
        return dim_latent_pred, dim_latent_var, distance_pred, distance_logstd, x

    def _get_dim_target_single(self, pos_assigned_gt_inds,
                               gt_bboxes_3d, gt_labels):
        dimensions = gt_bboxes_3d[pos_assigned_gt_inds, :3]
        labels = gt_labels[pos_assigned_gt_inds]
        return self.dim_coder.encode(dimensions, labels)

    def get_targets(self,
                    sampling_results,
                    gt_bboxes_3d,
                    gt_labels,
                    concat=True):
        """Get dimension and distance targets."""
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results]

        if self.loss_dim is not None:
            dim_targets = [
                self._get_dim_target_single(*args)
                for args in zip(
                    pos_assigned_gt_inds,
                    gt_bboxes_3d,
                    gt_labels)]
            if concat:
                dim_targets = torch.cat(dim_targets, dim=0)
        else:
            dim_targets = None

        distance_targets = None
        return dim_targets, distance_targets

    def slice_pred(self, dim_latent_pred, dim_latent_var, labels):
        if self.latent_class_agnostic:
            pos_dim_latent_pred = dim_latent_pred
        else:
            inds = torch.arange(len(labels), device=labels.device)
            pos_dim_latent_pred = dim_latent_pred.view(
                dim_latent_pred.size(0), -1, self.dim_dim + self.latent_channels
            )[inds, labels]
        dim_pred, latent_pred = pos_dim_latent_pred.split(
            [self.dim_dim, self.latent_channels], dim=1)
        dim_var = latent_var = None
        return dim_pred, dim_var, latent_pred, latent_var

    @force_fp32(apply_to=('dim_pred', ))
    def loss(self, dim_pred, distance_pred, distance_logstd,
             dim_targets, distance_targets):
        losses = dict()
        if self.loss_dim is not None:
            if dim_pred.size(0) == 0:
                losses['loss_dim'] = dim_pred.sum()
            else:
                losses['loss_dim'] = self.loss_dim(
                    dim_pred, dim_targets)
        return losses
