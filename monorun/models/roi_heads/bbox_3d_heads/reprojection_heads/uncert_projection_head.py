import torch
import torch.nn as nn
from mmdet.core import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss

from .....core import build_proj_error_coder


@HEADS.register_module()
class UncertProjectionHead(nn.Module):
    def __init__(self,
                 loss_proj=dict(
                     type='RobustKLLoss',
                     loss_weight=1.0,
                     show_pos=True,
                     grad_decay=True,
                     momentum=0.1),
                 z_min=0.5,
                 allowed_border=200,
                 proj_error_coder=dict(
                     type='DistanceInvarProjErrorCoder',
                     ref_length=1.6,
                     ref_focal_y=722,
                     target_std=0.15),
                 distance_mode='range'):
        super(UncertProjectionHead, self).__init__()

        self.loss_proj = build_loss(loss_proj) if loss_proj is not None else None
        self.z_min = z_min
        self.allowed_border = allowed_border
        self.proj_error_coder = build_proj_error_coder(proj_error_coder)
        self.fp16_enabled = False
        assert distance_mode in ['z-depth', 'range']
        self.distance_mode = distance_mode
        self.train_std_of_encoded_error = True

    @auto_fp16()
    def forward(self, coords_3d, proj_r_mats, proj_t_vecs, img_shapes):
        """
        Args:
            coords_3d (torch.Tensor): Shape (n, 3, h, w)
            proj_r_mats (torch.Tensor): Shape (n, 3, 3)
            proj_t_vecs (torch.Tensor): Shape (n, 3, 1)
            img_shapes (torch.Tensor): Shape (n, 2)

        Returns:
            torch.Tensor: Shape (n, 2, h, w), projected 2D coordinates
        """
        n, c, h, w = coords_3d.size()
        if n == 0:
            coords_3d = coords_3d.new_zeros((1, c, h, w)) + coords_3d.sum()
            proj_r_mats = proj_r_mats.new_zeros((1, 3, 3))
        coords_3d_proj = torch.einsum(
            'nchw,nxc->nxhw', coords_3d, proj_r_mats
        ) + proj_t_vecs.unsqueeze(-1)
        coords_2d_proj, coord_z = coords_3d_proj.split([2, 1], dim=1)
        # (n, 2, 28, 28)
        coords_2d_proj = coords_2d_proj / coord_z.clamp(min=self.z_min)

        coords_2d_min = -self.allowed_border  # Number
        # (n, 2, 1, 1) [[u_max, v_max]]
        coords_2d_max = img_shapes[:, [1, 0], None, None] + self.allowed_border
        coords_2d_proj.clamp_(min=coords_2d_min)
        coords_2d_proj = torch.min(coords_2d_proj, coords_2d_max)

        return coords_2d_proj

    def get_properties(self, sampling_results,
                       gt_proj_r_mats, gt_proj_t_vecs, gt_bboxes_3d, img_metas):
        img_shapes = []
        for img_meta, res in zip(img_metas, sampling_results):
            img_shapes += [img_meta['img_shape'][:2]] * len(res.pos_inds)
        if len(img_shapes):
            img_shapes = gt_proj_r_mats[0].new_tensor(img_shapes)  # [Npos, 2]
        else:
            img_shapes = gt_proj_r_mats[0].new_zeros((0, 2))

        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results]
        proj_r_mats = torch.cat(
            [gt_proj_r_mats_single[pos_assigned_gt_inds_single]
             for gt_proj_r_mats_single, pos_assigned_gt_inds_single in
             zip(gt_proj_r_mats, pos_assigned_gt_inds)], dim=0)
        proj_t_vecs = torch.cat(
            [gt_proj_t_vecs_single[pos_assigned_gt_inds_single]
             for gt_proj_t_vecs_single, pos_assigned_gt_inds_single in
             zip(gt_proj_t_vecs, pos_assigned_gt_inds)], dim=0)
        pos_bboxes_3d = torch.cat(
            [gt_bboxes_3d_single[pos_assigned_gt_inds_single]
             for gt_bboxes_3d_single, pos_assigned_gt_inds_single in
             zip(gt_bboxes_3d, pos_assigned_gt_inds)], dim=0)
        if self.distance_mode == 'z-depth':
            pos_distances = pos_bboxes_3d[:, 5:6]  # (Npos, 1)
        else:
            pos_distances = torch.norm(pos_bboxes_3d[:, 3:6], p=2, dim=1, keepdim=True)
        return proj_r_mats, proj_t_vecs, pos_bboxes_3d, pos_distances, img_shapes

    def get_distance(self, t_vec):
        if self.distance_mode == 'z-depth':
            distance = t_vec[:, 2]
        else:
            distance = torch.norm(t_vec, p=2, dim=1)
        return distance

    @force_fp32(apply_to=('coords_2d_proj', 'coords_2d_norm_logstd'))
    def loss(self, coords_2d_proj, coords_2d_norm_logstd,
             coords_2d_roi, distances):
        if self.loss_proj is not None:
            proj_error = self.proj_error_coder.encode(
                coords_2d_proj - coords_2d_roi, distances)
            if proj_error.size(0) == 0:
                loss_proj = proj_error.sum() + coords_2d_norm_logstd.sum()
            else:
                loss_proj = self.loss_proj(proj_error, 0, logstd=coords_2d_norm_logstd)
            losses = dict(loss_proj=loss_proj)
        else:
            losses = dict()
        return losses

    def loss_empty(self, device):
        if self.loss_proj is not None:
            return dict(
                loss_proj=torch.zeros(1, device=device, dtype=torch.float32))
        else:
            return dict()
