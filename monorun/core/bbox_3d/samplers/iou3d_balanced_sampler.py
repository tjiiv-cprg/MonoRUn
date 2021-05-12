import torch
from ..builder import IOU3D_SAMPLERS


@IOU3D_SAMPLERS.register_module()
class IoU3DBalancedSampler(object):

    def __init__(self,
                 pos_iou_thr=0.5,
                 pos_fraction_min=0.25,
                 pos_fraction_max=0.75,
                 smooth_keeprate=True,
                 min_iou=-1):
        self.pos_iou_thr = pos_iou_thr
        self.pos_fraction_min = pos_fraction_min
        self.pos_fraction_max = pos_fraction_max
        self.smooth_keeprate = smooth_keeprate
        self.min_iou = min_iou

    def sample(self, ious):

        num_total = ious.numel()
        pos_mask = ious >= self.pos_iou_thr
        num_pos = torch.sum(pos_mask)
        num_neg = num_total - num_pos
        num_pos_max = \
            self.pos_fraction_max / (1 - self.pos_fraction_max) * num_neg
        num_neg_max = \
            (1 - self.pos_fraction_min) / self.pos_fraction_min * num_pos

        if num_pos <= num_pos_max and num_neg <= num_neg_max:
            sampling_mask = torch.ones_like(ious, dtype=torch.bool)

        else:
            if num_pos > num_pos_max:
                pos_keeprate = num_pos_max / num_pos
                neg_keeprate = 1
            else:  # num_neg > num_neg_max
                pos_keeprate = 1
                neg_keeprate = num_neg_max / num_neg
            if not self.smooth_keeprate:
                keeprate = torch.full_like(ious, neg_keeprate)
                keeprate[pos_mask] = pos_keeprate
            else:
                strong_pos_thr = (self.pos_iou_thr + 1) / 2
                strong_neg_thr = self.pos_iou_thr / 2
                keeprate = (pos_keeprate - neg_keeprate
                            ) / (strong_pos_thr - strong_neg_thr
                                 ) * (ious - strong_neg_thr) + neg_keeprate
                keeprate.clamp(max=max(pos_keeprate, neg_keeprate),
                               min=min(pos_keeprate, neg_keeprate))
            sampling_mask = torch.rand_like(ious) < keeprate

        sampling_mask[ious < self.min_iou] = 0

        return sampling_mask
