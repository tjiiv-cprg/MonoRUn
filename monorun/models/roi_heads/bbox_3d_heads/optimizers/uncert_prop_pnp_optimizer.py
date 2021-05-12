import math
import torch
import torch.nn as nn
from mmdet.core import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss
from .....core import build_rotation_coder, bbox3d_overlaps_aligned_torch
from .....ops import build_pnp

PI = math.pi


@HEADS.register_module()
class UncertPropPnPOptimizer(nn.Module):
    def __init__(self,
                 loss_rot=None,
                 loss_trans=None,
                 loss_calib=None,
                 rotation_coder=dict(type='Vec2DRotationCoder'),
                 pnp=dict(
                     type='PnPUncert',
                     z_min=0.5,
                     epnp_istd_thres=0.6,
                     inlier_opt_only=True,
                     forward_exact_hessian=False,
                     backward_exact_hessian=True),
                 allowed_border=200,
                 epnp_ransac_thres_ratio=0.2,
                 std_scale=10):
        super(UncertPropPnPOptimizer, self).__init__()

        self.pnp = build_pnp(pnp)
        self.epnp_ransac_thres_ratio = epnp_ransac_thres_ratio
        self.allowed_border = allowed_border
        self.std_scale = std_scale
        self.rotation_coder = build_rotation_coder(rotation_coder)
        self.fp16_enabled = False
        self.loss_rot = build_loss(loss_rot) if loss_rot is not None \
            else None
        self.loss_trans = build_loss(loss_trans) if loss_trans is not None \
            else None
        self.loss_calib = build_loss(loss_calib) if loss_calib is not None \
            else None

        self.cov_calib_logscale = nn.Parameter(
            torch.full((4, ), 0, dtype=torch.float))

    def init_weights(self):
        pass

    @auto_fp16()
    def forward(self, coords_2d, coords_2d_logstd, coords_3d,
                cam_intrinsic, img_shapes):
        """
        Args:
            coords_2d (Tensor): shape (Nbatch, 2, h, w)
            coords_2d_logstd (Tensor): shape (Nbatch, 2, h, w)
            coords_3d (Tensor): shape (Nbatch, 3, h, w)
            cam_intrinsic (Tensor): shape (Nbatch, 3, 3) or (1, 3, 3)
            cam_t_vecs (Tensor): shape (Nbatch, 3, 1) or (1, 3, 1)
            distances (Tensor | None): shape (Nbatch, 1)
            distances_logstd (Tensor | None): shape (Nbatch, 1)
            img_shapes (torch.Tensor): Shape (Nbatch, 2) or (1, 2)

        Returns:
            ret_val (Tensor): shape (Nbatch, 1), validity bool mask
            yaw_pred (Tensor): shape (Nbatch, 1)
            t_vec_pred (Tensor): shape (Nbatch, 3)
            pose_cov_pred (Tensor): shape (Nbatch, 4, 4), covariance matrices
                of [yaw_pred, t_vec_pred]
        """
        bn, c, h, w = coords_2d.size()

        coords_2d_istd = torch.exp(-coords_2d_logstd) / self.std_scale

        u_range = coords_2d.new_full(
            (img_shapes.size(0), 2), -self.allowed_border)
        v_range = coords_2d.new_full(
            (img_shapes.size(0), 2), -self.allowed_border)
        u_range[:, 1] = img_shapes[:, 1] + self.allowed_border
        v_range[:, 1] = img_shapes[:, 0] + self.allowed_border

        coords_2d_reshape = coords_2d.permute(0, 2, 3, 1).view(bn, h * w, 2)
        coords_2d_istd_reshape = coords_2d_istd.permute(0, 2, 3, 1).view(bn, h * w, 2)
        coords_3d_reshape = coords_3d.permute(0, 2, 3, 1).view(bn, h * w, 3)

        roi_heights = coords_2d[:, 1, -1, 0] - coords_2d[:, 1, 0, 0]
        epnp_ransac_thres = self.epnp_ransac_thres_ratio * roi_heights \
            if self.epnp_ransac_thres_ratio is not None else None

        ret_val, yaw_pred, t_vec_pred, pose_cov_pred, _ = self.pnp(
            coords_2d_reshape, coords_2d_istd_reshape,
            coords_3d_reshape,
            cam_intrinsic,
            u_range, v_range,
            epnp_ransac_thres)
        cov_calib_scale = torch.exp(self.cov_calib_logscale)
        pose_cov_calib = (cov_calib_scale * cov_calib_scale[:, None]) * pose_cov_pred

        return ret_val, yaw_pred, t_vec_pred, pose_cov_pred, pose_cov_calib

    def get_targets(self, pos_bboxes_3d):
        yaw_targets = pos_bboxes_3d[:, 6:7]
        trans_targets = pos_bboxes_3d[:, 3:6]
        return yaw_targets, trans_targets

    @force_fp32(apply_to=('ret_val', 'yaw_pred', 't_vec_pred',
                          'pose_cov', 'dimensions_pred'))
    def loss(self, ret_val, yaw_pred, t_vec_pred, pose_cov,
             dimensions_pred, yaw_targets, trans_targets, pos_bboxes_3d,
             eps=1e-6):
        loss_pose = dict()
        with torch.no_grad():
            # (n, 1)
            ious = bbox3d_overlaps_aligned_torch(
                pos_bboxes_3d[:, [3, 4, 5, 0, 1, 2, 6]],
                torch.cat((t_vec_pred,
                           dimensions_pred,
                           yaw_pred), dim=1)).unsqueeze(1)
            ious[~ret_val] = 0
        if ious.size(0) == 0:
            mean_iou = ious.sum()  # zero
        else:
            mean_iou = ious.mean()
        loss_pose['mean_iou'] = mean_iou

        yaw_pred = yaw_pred[ret_val]
        yaw_targets = yaw_targets[ret_val]
        t_vec_pred = t_vec_pred[ret_val]
        trans_targets = trans_targets[ret_val]
        pose_cov = pose_cov[ret_val]
        if self.loss_rot is not None:
            rot_pred = self.rotation_coder.encode(yaw_pred)
            rot_targets = self.rotation_coder.encode(yaw_targets)
            if rot_pred.size(0) == 0:
                loss_rot = rot_pred.sum()
            else:
                loss_rot = self.loss_rot(
                    torch.norm(rot_pred - rot_targets, dim=1, p=2, keepdim=True), -1)
            loss_pose['loss_rot'] = loss_rot
        if self.loss_trans is not None:
            if t_vec_pred.size(0) == 0:
                loss_trans = t_vec_pred.sum()
            else:
                loss_trans = self.loss_trans(t_vec_pred, trans_targets)
            loss_pose['loss_trans'] = loss_trans
        if self.loss_calib is not None:
            if pose_cov.size(0) == 0:
                loss_calib = pose_cov.sum()
            else:
                with torch.no_grad():
                    yaw_diff = ((yaw_pred - yaw_targets) + PI) % (2 * PI) - PI
                    t_vec_diff = t_vec_pred - trans_targets
                    diff = torch.cat([yaw_diff, t_vec_diff], dim=1)
                pose_cov_inv = torch.inverse(pose_cov + torch.eye(
                    pose_cov.size(-1),
                    dtype=pose_cov.dtype, device=pose_cov.device))
                loss_calib = self.loss_calib(diff, 0, inv_cov=pose_cov_inv)
            loss_pose['loss_calib'] = loss_calib
        return loss_pose, ious
