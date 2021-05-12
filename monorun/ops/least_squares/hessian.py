import torch
from .jacobian import get_jacobian_and_error


def exact_hessian(
        coords_2d, coords_2d_istd,
        coords_3d,
        cam_mats,
        u_range, v_range, z_min,
        yaw, t_vec, inlier_mask):
    bn, pn = coords_2d.shape[0:2]

    coords_2d_ = coords_2d.detach().repeat(1, 4, 1).view(bn * 4, pn, 2)
    coords_2d_istd_ = coords_2d_istd.detach().repeat(1, 4, 1).view(bn * 4, pn, 2)
    coords_3d_ = coords_3d.detach().repeat(1, 4, 1).view(bn * 4, pn, 3)
    if cam_mats.size(0) == 1 < bn:
        cam_mats_ = cam_mats.detach().clone()
    else:
        cam_mats_ = cam_mats.detach().repeat(1, 4, 1).view(bn * 4, 3, 3)
    if u_range.size(0) == 1 < bn:
        u_range_ = u_range
    else:
        u_range_ = u_range.repeat(1, 4).view(bn * 4, 2)
    if v_range.size(0) == 1 < bn:
        v_range_ = v_range
    else:
        v_range_ = v_range.repeat(1, 4).view(bn * 4, 2)
    inlier_mask_ = inlier_mask.detach().repeat(1, 4).view(bn * 4, pn) \
        if inlier_mask is not None else None

    torch.set_grad_enabled(True)
    # (b, 1), (b, 3) -> (b, 4) -> (b, 4*4) -> (b*4, 4)
    pose = torch.cat([yaw.detach(), t_vec.detach()], dim=1
                     ).repeat(1, 4).view(bn * 4, 4).requires_grad_()
    yaw_ = pose[:, :1]
    t_vec_ = pose[:, 1:]

    (jac_t_vec,  # (b*4, n, 2, 3)
     jac_yaw,  # (b*4, n, 2, 1)
     error  # (b*4, n, 2)
     ) = get_jacobian_and_error(
        coords_2d_, coords_2d_istd_,
        coords_3d_,
        cam_mats_,
        u_range_, v_range_, z_min,
        yaw_, t_vec_, inlier_mask_)

    # (b*4, n, 2, 4) -> (b*4, 2n, 4)
    jac_pose = torch.cat(
        (jac_yaw, jac_t_vec), dim=3).view(bn * 4, -1, 4)
    # (b*4, n, 2) -> (b*4, 2n, 1)
    error = error.view(bn * 4, -1, 1)

    # (b*4, 4, 1) = (b*4, 4, 2n+1) * (b*4, 2n+1, 1)
    jt_error = torch.matmul(
        jac_pose.permute(0, 2, 1), error)
    # (b, 4, 4)
    jac = jt_error.view(-1, 4, 4)
    sum_grad_jac = torch.diagonal(jac, dim1=1, dim2=2).sum()
    # (b, 4, 4)
    h = torch.autograd.grad(
        sum_grad_jac, pose)[0].view(-1, 4, 4)

    return h


def approx_hessian(
        coords_2d, coords_2d_istd,
        coords_3d,
        cam_mats,
        u_range, v_range, z_min,
        yaw, t_vec, inlier_mask):
    bn, pn = coords_2d.shape[0:2]
    (jac_t_vec,  # (b, n, 2, 3)
     jac_yaw,  # (b, n, 2, 1)
     error  # (b, n, 2)
     ) = get_jacobian_and_error(
        coords_2d, coords_2d_istd,
        coords_3d,
        cam_mats,
        u_range, v_range, z_min,
        yaw, t_vec, inlier_mask)
    # (b, n, 2, 4) -> (b, 2n, 4)
    jac_pose = torch.cat(
        (jac_yaw, jac_t_vec), dim=3).view(bn, -1, 4)
    h = torch.matmul(jac_pose.permute(0, 2, 1), jac_pose)  # may cause inf or nan
    return h
