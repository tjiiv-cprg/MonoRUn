from ._ext import lib, ffi
import numpy as np
import cv2
from mmdet.core import multi_apply

DISTANCE_MODE_CODE = {
    'range': 0,
    'z-depth': 1
}

def u2d_pnp_cpu_single(coord_2d, coord_2d_istd,
                       coord_3d,
                       istd_inlier_mask,
                       cam_mat,
                       u_range, v_range,
                       epnp_ransac_thres,
                       inlier_opt_only=False,
                       z_min=0.5,
                       dist_coeffs=None,
                       with_pose_cov=True):
    pn = coord_2d.shape[0]

    istd_inlier_count = np.count_nonzero(istd_inlier_mask)
    if istd_inlier_count > 4:
        coord_3d_inlier = coord_3d[istd_inlier_mask]
        coord_2d_inlier = coord_2d[istd_inlier_mask]
        coord_2d_istd_inlier = coord_2d_istd[istd_inlier_mask]
    else:
        coord_3d_inlier = coord_3d
        coord_2d_inlier = coord_2d
        coord_2d_istd_inlier = coord_2d_istd
        istd_inlier_mask[:] = True

    if epnp_ransac_thres is not None:
        ret_val, r_vec, t_vec, ransac_inlier_ind = cv2.solvePnPRansac(
            coord_3d_inlier,
            coord_2d_inlier,
            cam_mat,
            dist_coeffs,
            reprojectionError=epnp_ransac_thres,
            iterationsCount=30,
            flags=cv2.SOLVEPNP_EPNP)
        if ransac_inlier_ind is not None and len(ransac_inlier_ind) > 4:
            ransac_inlier_ind = ransac_inlier_ind.squeeze(1)
            ransac_inlier_mask = np.zeros(
                coord_3d_inlier.shape[0], dtype=np.bool)
            ransac_inlier_mask[ransac_inlier_ind] = True
            coord_3d_inlier = coord_3d_inlier[ransac_inlier_ind]
            coord_2d_inlier = coord_2d_inlier[ransac_inlier_ind]
            coord_2d_istd_inlier = coord_2d_istd_inlier[ransac_inlier_ind]
            istd_inlier_mask[istd_inlier_mask] = ransac_inlier_mask

    else:
        ret_val, r_vec, t_vec = cv2.solvePnP(coord_3d_inlier,
                                             coord_2d_inlier,
                                             cam_mat,
                                             dist_coeffs,
                                             flags=cv2.SOLVEPNP_EPNP)
    inlier_mask = istd_inlier_mask

    if ret_val:
        if inlier_opt_only:
            coord_3d = coord_3d_inlier
            coord_2d = coord_2d_inlier
            coord_2d_istd = coord_2d_istd_inlier
            pn = coord_2d.shape[0]

        yaw = r_vec[1:2]

        clips = np.array([z_min,
                          u_range[0],
                          u_range[1],
                          v_range[0],
                          v_range[1]], np.float64)
        coord_2d = np.ascontiguousarray(coord_2d, np.float64)
        coord_3d = np.ascontiguousarray(coord_3d, np.float64)
        coord_2d_istd = np.ascontiguousarray(coord_2d_istd, np.float64)
        cam_mat = np.ascontiguousarray(cam_mat, np.float64)
        init_pose = np.ascontiguousarray(
            np.concatenate([yaw, t_vec], axis=0), np.float64)
        clips = np.ascontiguousarray(clips, np.float64)

        ceres_dtype = 'double*'
        coord_2d_ptr = ffi.cast(ceres_dtype, coord_2d.ctypes.data)
        coord_3d_ptr = ffi.cast(ceres_dtype, coord_3d.ctypes.data)
        coord_2d_istd_ptr = ffi.cast(ceres_dtype, coord_2d_istd.ctypes.data)
        cam_mat_ptr = ffi.cast(ceres_dtype, cam_mat.ctypes.data)
        init_pose_ptr = ffi.cast(ceres_dtype, init_pose.ctypes.data)
        clips_ptr = ffi.cast(ceres_dtype, clips.ctypes.data)

        result_val = np.zeros([1], np.int32)
        result_pose = np.zeros([4], np.float64)
        result_cov = np.eye(4, dtype=np.float64) if with_pose_cov else None
        result_tr = np.zeros([1], np.float64)  # trust region radius
        result_val_ptr = ffi.cast('int*', result_val.ctypes.data)
        result_pose_ptr = ffi.cast(ceres_dtype, result_pose.ctypes.data)
        result_cov_ptr = \
            ffi.cast(ceres_dtype, result_cov.ctypes.data) if with_pose_cov \
            else ffi.NULL
        result_tr_ptr = ffi.cast(ceres_dtype, result_tr.ctypes.data)

        lib.pnp_uncert(
            coord_2d_ptr, coord_3d_ptr, coord_2d_istd_ptr,
            cam_mat_ptr, init_pose_ptr, result_val_ptr,
            result_pose_ptr, result_cov_ptr, result_tr_ptr,
            pn, clips_ptr)

        yaw_refined = result_pose[0:1].astype(np.float32)
        t_vec_refined = result_pose[1:].astype(np.float32)
        pose_cov = result_cov.astype(np.float32) if result_cov is not None else None
        tr_radius = result_tr.astype(np.float32)
        return (result_val[0] > 0,
                yaw_refined,
                t_vec_refined,
                pose_cov,
                tr_radius,
                inlier_mask)

    else:
        return (False,
                np.zeros(1, np.float32),
                np.zeros(3, np.float32),
                np.eye(4, dtype=np.float32) if with_pose_cov else None,
                np.zeros(1, np.float32),
                inlier_mask)


def u2d_pnp_cpu(coords_2d, coords_2d_istd,
                coords_3d,
                cam_mats,
                u_range, v_range, z_min=0.5,
                epnp_istd_thres=1.0,
                epnp_ransac_thres=None,
                inlier_opt_only=False,
                with_pose_cov=True):
    """
    Args:
        coords_2d (ndarray): shape (Nbatch, Npoint, 2)
        coords_2d_istd (ndarray): shape (Nbatch, Npoint, 2)
        coords_3d (ndarray): shape (Nbatch, Npoint, 3)
        cam_mats (ndarray): shape (Nbatch, 3, 3) or (1, 3, 3)
        u_range (ndarray): shape (Nbatch, 2) or (1, 2)
        v_range (ndarray): shape (Nbatch, 2) or (1, 2)
        z_min (float):
        epnp_istd_thres (float):
        epnp_ransac_thres (None | ndarray): shape (Nbatch, )
        inlier_opt_only (bool):

    Returns:
        ret_val (ndarray): shape (Nbatch, ), validity bool mask
        yaw (ndarray): shape (Nbatch, 1)
        t_vec (ndarray): shape (Nbatch, 3)
        pose_cov (ndarray): shape (Nbatch, 4, 4), covariance matrices
            of [yaw, t_vec]
        tr_radius (ndarray): shape (Nbatch, 1), trust region radius
        inlier_mask (ndarray): shape (Nbatch, Npoint), inlier bool mask
    """
    bn = coords_2d.shape[0]
    pn = coords_2d.shape[1]

    if bn > 0:
        assert coords_2d_istd.shape[1] == coords_3d.shape[1] == pn >= 4

        coord_2d_istd_mean = np.mean(
            coords_2d_istd, axis=1, keepdims=True)  # (Nbatch, 1, 2)
        # u and v istd should be greater than the threshold, (Nbatch, Npoint)
        istd_inlier_masks = np.min(
            coords_2d_istd >= epnp_istd_thres * coord_2d_istd_mean, axis=2)

        if cam_mats.shape[0] == 1 < bn:
            cam_mats = [cam_mats.squeeze(0)] * bn
        if u_range.shape[0] == 1 < bn:
            u_range = [u_range.squeeze(0)] * bn
        if v_range.shape[0] == 1 < bn:
            v_range = [v_range.squeeze(0)] * bn
        if epnp_ransac_thres is None:
            epnp_ransac_thres = [None] * bn
        dist_coeffs = np.zeros((8, 1), dtype=np.float32)  # zero distortion

        ret_val, yaw, t_vec, pose_cov, tr_radius, inlier_mask = multi_apply(
            u2d_pnp_cpu_single,
            coords_2d, coords_2d_istd,
            coords_3d,
            istd_inlier_masks,
            cam_mats,
            u_range, v_range,
            epnp_ransac_thres,
            inlier_opt_only=inlier_opt_only,
            z_min=z_min,
            dist_coeffs=dist_coeffs,
            with_pose_cov=with_pose_cov)

        ret_val = np.array(ret_val, dtype=np.bool)  # (Nbatch, )
        yaw = np.stack(yaw, axis=0)  # (Nbatch, 1)
        t_vec = np.stack(t_vec, axis=0)  # (Nbatch, 3)
        # (Nbatch, 4, 4)
        pose_cov = np.stack(pose_cov, axis=0) if with_pose_cov else None
        tr_radius = np.stack(tr_radius, axis=0)  # (Nbatch, 1)
        inlier_mask = np.stack(inlier_mask, axis=0)  # (Nbatch, Npoint)

    else:
        ret_val = np.zeros((0, ), dtype=np.bool)
        yaw = np.zeros((0, 1), dtype=np.float32)
        t_vec = np.zeros((0, 3), dtype=np.float32)
        pose_cov = np.zeros((0, 4, 4), dtype=np.float32)
        tr_radius = np.zeros((0, 1), dtype=np.float32)
        inlier_mask = np.zeros((0, pn), dtype=np.bool)

    return ret_val, yaw, t_vec, pose_cov, tr_radius, inlier_mask
