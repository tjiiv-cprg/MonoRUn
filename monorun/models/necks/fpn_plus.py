import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.core import auto_fp16
from mmdet.models import NECKS, FPN


@NECKS.register_module()
class FPNplus(FPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 num_lower_outs=1,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPNplus, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            extra_convs_on_inputs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            upsample_cfg)
        self.num_lower_outs = num_lower_outs
        self.lower_fpn_convs = nn.ModuleList()
        for i in range(self.num_lower_outs):
            lower_fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lower_fpn_convs.append(lower_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function"""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        lower_levels = []
        for i in range(self.num_lower_outs):
            lower_levels.append(F.interpolate(
                laterals[0],
                scale_factor=2**(self.num_lower_outs - i),
                mode='bilinear'))

        # build outputs
        # part 0: lower levels
        outs_lower = [
            lower_fpn_conv(lower_levels[i])
            for i, lower_fpn_conv in enumerate(self.lower_fpn_convs)
        ]
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs_lower + outs)
