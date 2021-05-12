import torch
from .pnp_uncert_cpu import u2d_pnp_cpu
from .builder import PNP
from .hessian import exact_hessian, approx_hessian


def pnp_uncert(coords_2d, coords_2d_istd, coords_3d,
               cam_mats, u_range, v_range, z_min=0.5,
               epnp_istd_thres=1.0, epnp_ransac_thres=None,
               inlier_opt_only=False, forward_exact_hessian=False,
               use_6dof=False):
    """
    Args:
        coords_2d (torch.Tensor): shape (Nbatch, Npoint, 2)
        coords_2d_istd (torch.Tensor): shape (Nbatch, Npoint, 2)
        coords_3d (torch.Tensor): shape (Nbatch, Npoint, 3)
        cam_mats (torch.Tensor): shape (Nbatch, 3, 3) or (1, 3, 3)
        u_range (torch.Tensor): shape (Nbatch, 2) or (1, 2)
        v_range (torch.Tensor): shape (Nbatch, 2) or (1, 2)
        z_min (float):
        epnp_istd_thres (float):
        epnp_ransac_thres (None | torch.Tensor): shape (Nbatch, )
        inlier_opt_only (bool):

    Returns:
        ret_val (Tensor): shape (Nbatch, ), validity bool mask
        r_vec (Tensor): shape (Nbatch, 1) or (Nbatch, 3)
        t_vec (Tensor): shape (Nbatch, 3)
        pose_cov (Tensor): shape (Nbatch, 4, 4), covariance matrices
            of [yaw, t_vec]
        inlier_mask (Tensor): shape (Nbatch, Npoint), inlier bool mask
    """
    with torch.no_grad():
        coords_2d_np = coords_2d.cpu().numpy()
        coords_2d_istd_np = coords_2d_istd.cpu().numpy()
        coords_3d_np = coords_3d.cpu().numpy()
        cam_mats_np = cam_mats.cpu().numpy()
        u_range_np = u_range.cpu().numpy()
        v_range_np = v_range.cpu().numpy()
        if epnp_ransac_thres is not None:
            epnp_ransac_thres_np = epnp_ransac_thres.cpu().numpy()
        else:
            epnp_ransac_thres_np = None

        ret_val, r_vec, t_vec, _, _, inlier_mask = u2d_pnp_cpu(
            coords_2d_np, coords_2d_istd_np,
            coords_3d_np,
            cam_mats_np,
            u_range_np, v_range_np, z_min=z_min,
            epnp_istd_thres=epnp_istd_thres,
            epnp_ransac_thres=epnp_ransac_thres_np,
            inlier_opt_only=inlier_opt_only,
            with_pose_cov=False)

        ret_val = coords_2d.new_tensor(ret_val, dtype=torch.bool)
        r_vec = coords_2d.new_tensor(r_vec)
        t_vec = coords_2d.new_tensor(t_vec)
        inlier_mask = coords_2d.new_tensor(inlier_mask, dtype=torch.bool)

        if ret_val.size(0) == 0:
            pose_cov = coords_2d.new_zeros((0, 4, 4))
        else:
            if forward_exact_hessian:
                h = exact_hessian(
                    coords_2d, coords_2d_istd,
                    coords_3d,
                    cam_mats,
                    u_range, v_range, z_min,
                    r_vec, t_vec, inlier_mask)
            else:
                h = approx_hessian(
                    coords_2d, coords_2d_istd,
                    coords_3d,
                    cam_mats,
                    u_range, v_range, z_min,
                    r_vec, t_vec, inlier_mask)
            try:
                pose_cov = torch.inverse(h)
            except RuntimeError:
                # (*, n)
                eigval, _ = torch.symeig(h, eigenvectors=False)
                valid_mask = eigval[:, 0] > (1e-6 * eigval[:, 3]).clamp(min=0)
                ret_val &= valid_mask
                h[~ret_val] = torch.eye(4, device=h.device, dtype=h.dtype)
                pose_cov = torch.inverse(h)

    return ret_val, r_vec, t_vec, pose_cov, inlier_mask


@PNP.register_module()
class PnPUncert(torch.nn.Module):

    def __init__(self, z_min=0.5,
                 epnp_istd_thres=0.6,
                 inlier_opt_only=True,
                 coord_istd_normalize=False,
                 forward_exact_hessian=False,
                 use_6dof=False,
                 eps=1e-6):
        """Uncertainty-2D PnP v3.

        This algorithm uses the exact derivative of L-M iteration. Instead of
        computing the derivative matrix directly, this implementation computes
        the derivative of the product of output gradient w.r.t pose and L-M
        step using auto grad, which directly yields the output gradient w.r.t.
        PnP inputs.

        Args:
            z_min (float):
            epnp_istd_thres (float): points with istd greater than (thres
                * istd_mean) will be kept as inliers
            inlier_opt_only (bool): whether to use inliers or all points for
                non-linear optimization, note that this will affect back-
                propagation
        """
        super(PnPUncert, self).__init__()
        self.z_min = z_min
        self.epnp_istd_thres = epnp_istd_thres
        self.inlier_opt_only = inlier_opt_only
        self.coord_istd_normalize = coord_istd_normalize
        self.forward_exact_hessian = forward_exact_hessian
        self.use_6dof = use_6dof
        self.eps = eps

    def forward(self,
                coords_2d, coords_2d_istd,
                coords_3d,
                cam_mats,
                u_range, v_range, epnp_ransac_thres=None):
        if self.coord_istd_normalize:
            mean = torch.mean(coords_2d_istd, dim=(1, 2), keepdim=True)
            coords_2d_istd = coords_2d_istd / mean.clamp(min=self.eps)
        return pnp_uncert(
            coords_2d, coords_2d_istd,
            coords_3d,
            cam_mats,
            u_range, v_range, z_min=self.z_min,
            epnp_istd_thres=self.epnp_istd_thres,
            epnp_ransac_thres=epnp_ransac_thres,
            inlier_opt_only=self.inlier_opt_only,
            forward_exact_hessian=self.forward_exact_hessian,
            use_6dof=self.use_6dof)
