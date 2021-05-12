import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _NormBase

from mmdet.core import auto_fp16
from mmdet.models.builder import HEADS, build_loss

from .....core import build_iou3d_sampler


@HEADS.register_module()
class MLPScoreHead(nn.Module):
    def __init__(self,
                 reg_fc_out_channels=1024,
                 num_pose_fcs=1,
                 pose_fc_out_channels=1024,
                 fusion_type='add',
                 num_fused_fcs=1,
                 fc_out_channels=256,
                 loss_score=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 mode='linear_average',
                 iou_thres=0.7,
                 linear_coefs=(-0.5, 2),
                 detach_preds=True,
                 use_pose_norm=True,
                 train_cfg=None):
        super(MLPScoreHead, self).__init__()
        assert mode in ['average', 'thres', 'linear_average']
        self.num_pose_fcs = num_pose_fcs
        self.num_fused_fcs = num_fused_fcs
        self.fc_out_channels = fc_out_channels
        self.pose_fc_out_channels = pose_fc_out_channels
        self.mode = mode
        self.iou_thres = iou_thres
        self.linear_coefs = linear_coefs
        self.fp16_enabled = False
        self.loss_score = build_loss(loss_score) if loss_score is not None \
            else None
        self.relu = nn.ReLU(inplace=True)
        self.pre_sigmoid = True
        self.detach_preds = detach_preds
        self.reg_fc_out_channels = reg_fc_out_channels
        assert fusion_type in ['add', 'concat']
        self.fusion_type = fusion_type
        self.use_pose_norm = use_pose_norm
        self.train_cfg = train_cfg

        if self.train_cfg and hasattr(self.train_cfg, 'iou3d_sampler'):
            self.iou3d_sampler = build_iou3d_sampler(
                self.train_cfg.iou3d_sampler)
        else:
            self.iou3d_sampler = None

        layer_dim = reg_fc_out_channels

        assert self.num_pose_fcs > 0
        if self.fusion_type == 'add':
            assert self.pose_fc_out_channels == self.reg_fc_out_channels
        elif self.fusion_type == 'concat':
            layer_dim += self.pose_fc_out_channels
        self.pose_fcs = nn.ModuleList()
        pose_last_layer_dim = 1 + 3 + 10 + 3  # only use lower triangle of cov
        if self.use_pose_norm:
            self.pose_norm = BatchNormSmooth1D(pose_last_layer_dim, momentum=0.01)
        for i in range(self.num_pose_fcs):
            fc_in_channels = (
                pose_last_layer_dim if i == 0 else self.pose_fc_out_channels)
            self.pose_fcs.append(
                nn.Linear(fc_in_channels, self.pose_fc_out_channels))

        assert self.num_fused_fcs > 0
        self.fused_fcs = nn.ModuleList()
        for i in range(self.num_fused_fcs):
            fc_in_channels = (
                layer_dim if i == 0 else self.fc_out_channels)
            self.fused_fcs.append(
                nn.Linear(fc_in_channels, self.fc_out_channels))

        self.fc_out = nn.Linear(self.fc_out_channels, 1)

    def init_weights(self):
        for m in self.pose_fcs.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.fused_fcs.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_out.weight, 0, 0.01)
        nn.init.constant_(self.fc_out.bias, 0)

    @auto_fp16()
    def forward(self, reg_fc_out, yaw, t_vec, pose_cov, dimensions):
        if self.detach_preds:
            yaw = yaw.detach()
            t_vec = t_vec.detach()
            pose_cov = pose_cov.detach()
            dimensions = dimensions.detach()
        cov_x_inds, cov_y_inds = torch.tril_indices(4, 4, device=pose_cov.device)
        pose_cov_tril = pose_cov[:, cov_x_inds, cov_y_inds]
        x = torch.cat([yaw, t_vec, pose_cov_tril, dimensions], dim=1)
        if self.use_pose_norm:
            x = self.pose_norm(x)
        for fc in self.pose_fcs:
            x = self.relu(fc(x))
        if self.fusion_type == 'add':
            x = x + reg_fc_out
        else:
            x = torch.cat([x, reg_fc_out], dim=1)
        for fc in self.fused_fcs:
            x = self.relu(fc(x))
        scores = self.fc_out(x).squeeze(1)
        return scores  # shape (n, )

    def loss(self, scores, ious):
        if self.loss_score is not None:
            if scores.size(0) == 0:
                loss_score = scores.sum()
            else:
                scores = scores.unsqueeze(1)  # (N, 1)
                if self.mode == 'thres':
                    targets = (ious >= self.iou_thres).to(scores.dtype)
                elif self.mode == 'linear_average':
                    targets = self.linear_coefs[0] + ious * self.linear_coefs[1]
                    targets.clamp_(min=0, max=1)
                else:
                    targets = ious
                if self.iou3d_sampler is None:
                    loss_score = self.loss_score(scores, targets)
                else:
                    weight = self.iou3d_sampler.sample(ious).to(ious.dtype)
                    weight /= weight.mean().clamp(min=1e-2)
                    loss_score = self.loss_score(scores, targets, weight=weight)
            losses = dict(loss_score=loss_score)
            return losses
        else:
            return dict()


class BatchNormSmooth1D(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNormSmooth1D, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        if bn_training and input.size(0) > 1:
            var, mean = torch.var_mean(input, dim=0)
            self.running_mean *= 1 - exponential_average_factor
            self.running_mean += exponential_average_factor * mean
            self.running_var *= 1 - exponential_average_factor
            self.running_var += exponential_average_factor * var

        out = input.sub(self.running_mean).div(
            (self.running_var + self.eps).sqrt()).mul(self.weight).add(self.bias)

        return out

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))
