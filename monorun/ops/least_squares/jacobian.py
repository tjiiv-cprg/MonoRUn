import torch


def forward_proj(coords_2d, coords_3d, cam_mats,
                 z_min, u_range, v_range, yaw, t_vec):

    bn = coords_2d.shape[0]
    sin_yaw = torch.sin(yaw).squeeze(1)  # (b, )
    cos_yaw = torch.cos(yaw).squeeze(1)  # (b, )
    # [[ cos_yaw, 0, sin_yaw],
    #  [       0, 1,       0],
    #  [-sin_yaw, 0, cos_yaw]]
    rot_mat = cos_yaw.new_zeros((bn, 3, 3))
    rot_mat[:, 0, 0] = cos_yaw
    rot_mat[:, 2, 2] = cos_yaw
    rot_mat[:, 0, 2] = sin_yaw
    rot_mat[:, 2, 0] = -sin_yaw
    rot_mat[:, 1, 1] = 1

    k_r = torch.matmul(cam_mats, rot_mat)  # (b, 3, 3)
    # (b, 3, 1) = (b, 3, 3) * (b, 3, 1), (b, 3, 1) -> (b, 3)
    k_t = torch.matmul(cam_mats, t_vec.unsqueeze(2)).squeeze(2)
    # (b, n, 3) = (b, n, 3) + (b, 1, 3)
    uvz = torch.einsum(  # Todo: replace einsum with matmul
        'bux,bnx->bnu', k_r, coords_3d  # (b, 3, 3), (b, n, 3) -> (b, n, 3)
    ) + k_t.unsqueeze(1)
    uv, z = uvz.split([2, 1], dim=2)  # (b, n, 2), (b, n, 1)
    z_clip_mask = z < z_min  # (b, n, 1)
    z[z_clip_mask] = z_min  # (b, n, 1)
    uv /= z  # (b, n, 2) projected 2d points

    u_range = u_range.unsqueeze(1)  # (b, 1, 2)
    v_range = v_range.unsqueeze(1)  # (b, 1, 2)
    uv_lb = torch.stack((u_range[..., 0], v_range[..., 0]), dim=2)  # (b, 1, 2)
    uv_ub = torch.stack((u_range[..., 1], v_range[..., 1]), dim=2)  # (b, 1, 2)
    uv_clip_mask_lb = uv < uv_lb  # (b, n, 2)
    uv_clip_mask_ub = uv > uv_ub  # (b, n, 2)
    uv_clip_mask = uv_clip_mask_lb | uv_clip_mask_ub
    uv = torch.max(uv_lb, torch.min(uv_ub, uv))

    error_unweighted = uv - coords_2d  # (b, n, 2)

    return (uv, z, z_clip_mask, uv_clip_mask,
            sin_yaw, cos_yaw,
            error_unweighted, k_r)


def get_pose_jacobians(
        uv, z, z_clip_mask, uv_clip_mask, sin_yaw, cos_yaw,
        inlier_mask, cam_mats, coords_2d_istd, coords_3d):

    if inlier_mask is not None:
        outlier_mask = ~inlier_mask  # (b, n)
        # (b, n, 2)
        zero_mask = z_clip_mask | uv_clip_mask | outlier_mask[..., None]
    else:
        outlier_mask = None
        # (b, n, 2)
        zero_mask = z_clip_mask | uv_clip_mask

    # (b, n, 2, 2) = (b, 1, 2, 2) / (b, n, 1, 1)
    jac_t_vec_xy = cam_mats[:, None, :2, :2] / z.unsqueeze(3)
    # (b, n, 2, 1) = ((b, 1, 2, 1) - (b, n, 2, 1)) / (b, n, 1, 1)
    jac_t_vec_z = (cam_mats[:, None, :2, 2:3] - uv.unsqueeze(3)
                   ) / z.unsqueeze(3)
    # (b, n, 2, 3)
    jac_t_vec = torch.cat((jac_t_vec_xy, jac_t_vec_z), dim=3)
    # (b, n, 2, 3) *= (b, n, 2, 1)
    jac_t_vec *= coords_2d_istd.unsqueeze(3)
    jac_t_vec[zero_mask] = 0

    # [[fx, cx],   [[-sin_yaw,  cos_yaw],
    #  [ 0, cy]] *  [-cos_yaw, -sin_yaw]]
    jac_yaw_m1_l = cam_mats[:, 0:2, [0, 2]]  # (b, 2, 2)
    jac_yaw_m1_r = torch.stack(
        [torch.stack([-sin_yaw,  cos_yaw], dim=1),   # (b, 2)
         torch.stack([-cos_yaw, -sin_yaw], dim=1)],  # (b, 2)
        dim=1)  # (b, 2, 2)
    jac_yaw_m1 = torch.matmul(jac_yaw_m1_l, jac_yaw_m1_r)  # (b, 2, 2)
    jac_yaw_m2 = torch.einsum(
        'bnu,bx->bnux',
        uv,  # (b, n, 2)
        torch.stack([cos_yaw, sin_yaw], dim=1)  # (b, 2)
    )  # (b, n, 2, 2)
    # (b, n, 2, 2) = (b, 1, 2, 2) + (b, n, 2, 2)
    jac_yaw_m = jac_yaw_m1.unsqueeze(1) + jac_yaw_m2
    # (b, n, 2) = (b, n, 2) / (b, n, 1)
    jac_yaw = torch.einsum(
        'bnux,bnx->bnu',  # (b, n, 2, 2), (b, n, 2) -> (b, n, 2)
        jac_yaw_m,  # (b, n, 2, 2)
        coords_3d[..., [0, 2]]  # (b, n, 2) in [x, z] format
    ) / z
    # (b, n, 2) *= (b, n, 2)
    jac_yaw *= coords_2d_istd
    jac_yaw[zero_mask] = 0
    jac_yaw.unsqueeze_(3)  # (b, n, 2, 1)

    return jac_t_vec, jac_yaw, zero_mask, outlier_mask


def get_jacobians(
        coords_2d, coords_2d_istd,
        coords_3d,
        cam_mats,
        u_range, v_range, z_min,
        yaw, t_vec, inlier_mask):

    (uv,  # (b, n, 2) clipped
     z,   # (b, n, 1) clipped
     z_clip_mask,  # (b, n, 1) clipped are true
     uv_clip_mask,  # (b, n, 2) clipped are true
     sin_yaw,  # (b, )
     cos_yaw,  # (b, )
     error_unweighted,  # (b, n, 2) clipped
     k_r  # (b, 3, 3)
     ) = forward_proj(coords_2d, coords_3d, cam_mats,
                      z_min, u_range, v_range, yaw, t_vec)

    jac_t_vec, jac_yaw, zero_mask, outlier_mask = get_pose_jacobians(
        uv, z, z_clip_mask, uv_clip_mask, sin_yaw, cos_yaw,
        inlier_mask, cam_mats, coords_2d_istd, coords_3d)

    # point-wise jacobian
    # (b, n, 2, 3) = (b, 1, 2, 3) - (b, n, 2, 3)
    jac_pw_c3d = k_r[:, None, 0:2] - torch.einsum(
        'bnu,bx->bnux', uv, k_r[:, 2])  # (b, n, 2), (n, 3) -> (b, n, 2, 3)
    # (b, n, 2, 3) /= (b, n, 1, 1)
    jac_pw_c3d /= z.unsqueeze(3)
    # (b, n, 2, 3) *= (b, n, 2, 1)
    jac_pw_c3d *= coords_2d_istd.unsqueeze(3)
    jac_pw_c3d[zero_mask] = 0

    # element-wise jacobian
    jac_ew_istd = error_unweighted  # (b, n, 2)
    if outlier_mask is not None:
        jac_ew_istd[outlier_mask] = 0

    return jac_t_vec, jac_yaw, jac_pw_c3d, jac_ew_istd


def get_jacobian_and_error(
        coords_2d, coords_2d_istd,
        coords_3d,
        cam_mats,
        u_range, v_range, z_min,
        yaw, t_vec, inlier_mask):

    (uv,  # (b, n, 2) clipped
     z,   # (b, n, 1) clipped
     z_clip_mask,  # (b, n, 1) clipped are true
     uv_clip_mask,  # (b, n, 2) clipped are true
     sin_yaw,  # (b, )
     cos_yaw,  # (b, )
     error_unweighted,  # (b, n, 2) clipped
     k_r  # (b, 3, 3)
     ) = forward_proj(coords_2d, coords_3d, cam_mats,
                      z_min, u_range, v_range, yaw, t_vec)

    jac_t_vec, jac_yaw, _, _ = get_pose_jacobians(
        uv, z, z_clip_mask, uv_clip_mask, sin_yaw, cos_yaw,
        inlier_mask, cam_mats, coords_2d_istd, coords_3d)

    # compute weighted error
    # (b, n, 2) = (b, n, 2) * (b, n, 2)
    error = error_unweighted * coords_2d_istd

    return jac_t_vec, jac_yaw, error
