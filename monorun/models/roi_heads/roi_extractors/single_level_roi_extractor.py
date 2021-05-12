from mmdet.core import force_fp32
from mmdet.models import ROI_EXTRACTORS, SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractorMod(SingleRoIExtractor):

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 starting_level=0):
        super().__init__(roi_layer, out_channels, featmap_strides,
                         finest_scale=finest_scale)
        self.starting_level = starting_level

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        feats = feats[self.starting_level:]
        roi_feats = super().forward(feats, rois, roi_scale_factor=roi_scale_factor)
        return roi_feats
