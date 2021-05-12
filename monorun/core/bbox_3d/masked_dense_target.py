import torch
from mmcv.ops.roi_align import roi_align
from torch.nn.modules.utils import _pair


def masked_dense_target(pos_proposals_list, pos_assigned_gt_inds_list,
                        gt_dense_list, gt_mask_list, cfg,
                        eps=1e-4):
    targets, weights = zip(*map(
        lambda a, b, c, d: masked_dense_target_single(
            a, b, c, d, cfg, eps=eps),
        pos_proposals_list,
        pos_assigned_gt_inds_list,
        gt_dense_list,
        gt_mask_list))
    targets = torch.cat(targets, dim=0)
    weights = torch.cat(weights, dim=0)
    weights_mean = torch.mean(weights)
    weights = weights / weights_mean.clamp(min=eps)

    return targets, weights


def masked_dense_target_single(pos_proposals, pos_assigned_gt_inds, gt_dense,
                               gt_mask, cfg, eps=1e-4):
    dense_size = _pair(cfg.dense_size)
    num_pos = pos_proposals.size(0)

    if num_pos > 0:
        maxh, maxw = gt_dense.shape[-2:]
        pos_proposals_clip = torch.empty_like(pos_proposals)
        pos_proposals_clip[:, [0, 2]] = pos_proposals[:, [0, 2]].clamp(0, maxw)
        pos_proposals_clip[:, [1, 3]] = pos_proposals[:, [1, 3]].clamp(0, maxh)

        rois = torch.cat(
            [pos_assigned_gt_inds[:, None].to(pos_proposals_clip.dtype),
             pos_proposals_clip], dim=1)
        targets = roi_align(
            gt_dense, rois, dense_size, 1.0, 0, 'avg', True
        ).permute(0, 2, 3, 1)  # (n, 28, 28, 3)
        mask = roi_align(
            gt_mask, rois, dense_size, 1.0, 0, 'avg', True
        ).permute(0, 2, 3, 1)  # (n, 28, 28, 1)
        weights = mask.squeeze(-1) > eps  # (n, 28, 28)
        targets[weights] /= mask[weights]

        # (n, 3, 28, 28)
        targets = targets.permute(0, 3, 1, 2)
        # (n, 1, 28, 28)
        weights = weights.unsqueeze(1).to(targets.dtype)

    else:
        targets = pos_proposals.new_zeros((0, 3) + dense_size)
        weights = pos_proposals.new_zeros((0, 1) + dense_size)

    return targets, weights
