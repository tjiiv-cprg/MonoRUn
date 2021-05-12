from mmdet.core import multi_apply
from mmdet.models import HEADS, RPNHead


@HEADS.register_module()
class RPNHeadMod(RPNHead):

    def __init__(self, in_channels, starting_level=0, **kwargs):
        super().__init__(in_channels, **kwargs)
        self.starting_level = starting_level

    def forward(self, feats):
        return multi_apply(self.forward_single, feats[self.starting_level:])
