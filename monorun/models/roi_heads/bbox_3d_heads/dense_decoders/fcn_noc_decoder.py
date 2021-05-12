import torch
import torch.nn as nn
from torch.nn import Upsample
from mmcv.cnn import ConvModule, build_upsample_layer, build_plugin_layer
from mmcv.ops import Conv2d
from mmcv.ops.carafe import CARAFEPack
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, multi_apply
from mmdet.models.builder import HEADS, build_loss

from .....core import build_coord_coder, masked_dense_target


@HEADS.register_module()
class FCNNOCDecoder(nn.Module):

    def __init__(self,
                 num_convs=3,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=3,
                 class_agnostic=False,
                 upsample_cfg=dict(
                     type='carafe',
                     scale_factor=2,
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1,
                     compressed_channels=64),
                 num_convs_upsampled=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_noc=None,
                 noc_channels=3,
                 uncert_channels=2,
                 dropout2d_rate=0.2,
                 num_dropout2d_layers=1,
                 flip_correction=True,
                 plugins=None,
                 coord_coder=dict(
                     type='NOCCoder',
                     target_means=(-0.1, -0.5, 0.0),
                     target_stds=(0.35, 0.23, 0.34),
                     eps=1e-5),
                 use_latent_vec=True,
                 latent_activation=None,
                 latent_channels=16):
        super(FCNNOCDecoder, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        assert num_convs > 0
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_convs_upsampled = num_convs_upsampled
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_noc = (build_loss(loss_noc) if loss_noc is not None
                         else None)
        self.flip_correction = flip_correction
        self.with_plugins = plugins is not None
        self.noc_channels = noc_channels
        self.uncert_channels = uncert_channels
        self.channel_per_class = self.noc_channels + self.uncert_channels

        self.coord_coder = build_coord_coder(coord_coder)
        self.use_latent_vec = use_latent_vec
        self.latent_activation = (nn.ReLU() if latent_activation == 'ReLU'
                                  else nn.LeakyReLU() if latent_activation == 'LeakyReLU'
                                  else None)
        if self.use_latent_vec:
            self.latent_decoder = nn.Linear(
                latent_channels, self.conv_out_channels)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        if self.with_plugins:
            self.plugin_names = self.make_block_plugins(
                self.conv_out_channels, plugins)

        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=self.conv_out_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=self.conv_out_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        self.convs_upsampled = nn.ModuleList()
        for i in range(self.num_convs_upsampled):
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_upsampled.append(
                ConvModule(
                    self.conv_out_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        final_out_channels = (
            self.channel_per_class if self.class_agnostic
            else self.channel_per_class * self.num_classes)
        if self.flip_correction:
            final_out_channels *= 2
        self.conv_final = Conv2d(self.conv_out_channels, final_out_channels, 1)

        self.use_dropout2d = dropout2d_rate > 0
        if self.use_dropout2d:
            self.dropout2d = nn.Dropout2d(dropout2d_rate)
        self.num_dropout2d_layers = num_dropout2d_layers

    def init_weights(self):
        for m in [self.upsample, self.conv_final]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            elif isinstance(m, Upsample):
                continue
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        if self.use_latent_vec:
            nn.init.constant_(self.latent_decoder.weight, 0)
            nn.init.constant_(self.latent_decoder.bias, 0)

    def make_block_plugins(self, in_channels, plugins):
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    @auto_fp16()
    def forward(self, x, latent_pred, latent_var, labels, flip=False):
        """
        Args:
            x (Tensor)
            flip (bool | list[bool])

        Returns:
            Tensor
        """
        if self.use_dropout2d and self.num_dropout2d_layers > 0:
            x = self.dropout2d(x)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if self.use_dropout2d and i + 1 < self.num_dropout2d_layers:
                x = self.dropout2d(x)
        if self.use_latent_vec:
            if self.latent_activation is not None:
                latent_pred = self.latent_activation(latent_pred)
            deform = self.latent_decoder(latent_pred)[..., None, None]
            x = x + deform
        n, c, h, w = x.size()
        if n == 0:
            x = x.new_zeros((1, c, h, w)) + x.sum()
        if self.with_plugins:
            for name in self.plugin_names:
                x = getattr(self, name)(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        if n == 0:
            x = x[:0]
        for conv_upsampled in self.convs_upsampled:
            x = conv_upsampled(x)
        all_pred = self.conv_final(x)  # noc and uncert
        if self.flip_correction:
            all_pred = all_pred.view(
                all_pred.size(0), 2, all_pred.size(1) // 2,
                all_pred.size(2), all_pred.size(3))
            if isinstance(flip, bool):
                all_pred = all_pred[:, 0] if not flip else all_pred[:, 1]
            else:
                inds = torch.arange(
                    0, all_pred.size(0),
                    dtype=torch.long, device=all_pred.device)
                all_pred = all_pred[inds, inds.new_tensor(flip)]

        noc_pred, noc_var, proj_logstd = self.slice_pred(all_pred, labels)

        regular_params = None
        return noc_pred, noc_var, proj_logstd, regular_params

    def slice_pred(self, all_pred, labels):
        num_actual_classes = 1 if self.class_agnostic else self.num_classes

        split_list = [self.noc_channels * num_actual_classes,
                      self.uncert_channels * num_actual_classes]
        all_noc_pred, all_proj_logstd = all_pred.split(split_list, dim=1)

        # slice channels by classes
        if self.class_agnostic:
            noc_pred = all_noc_pred
            proj_logstd = all_proj_logstd

        else:
            num_rois, _, height, width = all_noc_pred.size()
            inds = torch.arange(
                0, num_rois, dtype=torch.long, device=all_noc_pred.device)
            noc_pred = all_noc_pred.view(
                num_rois, self.num_classes, 3, height, width
            )[inds, labels]  # (n, 3, 28, 28)

            proj_logstd = all_proj_logstd.view(
                num_rois, self.num_classes, self.uncert_channels, height, width
            )[inds, labels]  # (n, 2, 28, 28)

        noc_var = None
        return noc_pred, noc_var, proj_logstd

    def get_targets(self, sampling_results, gt_coords_3d,
                    gt_coords_3d_mask, gt_bboxes_3d, rcnn_train_cfg, img_metas):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results]
        dimensions = [gt_bboxes_3d_single[:, :3]
                      for gt_bboxes_3d_single in gt_bboxes_3d]
        flips = [img_metas_single['flip']
                 for img_metas_single in img_metas]
        nocs, nocs_mask = multi_apply(
            self.coord_coder.encode,
            gt_coords_3d,
            gt_coords_3d_mask,
            dimensions,
            flips)
        return masked_dense_target(
            pos_proposals, pos_assigned_gt_inds,
            nocs, nocs_mask, rcnn_train_cfg)

    @force_fp32(apply_to=('noc_pred',))
    def loss(self, noc_pred, noc_targets, noc_weights):
        if self.loss_noc is not None:
            if noc_pred.size(0) == 0:
                loss_noc = noc_pred.sum()
            else:
                loss_noc = self.loss_noc(noc_pred, noc_targets, weight=noc_weights)
            return dict(loss_noc=loss_noc)
        else:
            return dict()
