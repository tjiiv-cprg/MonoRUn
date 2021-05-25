import cv2
import numpy as np


def compute_box_bev(label):
    if isinstance(label, list):  # gt label
        ry = label[14]
        bl = label[10]
        bw = label[9]
        t = np.array([[label[11]],
                      [label[12]],
                      [label[13]]])
    else:  # det label
        ry = label[6]
        bl = label[0]
        bw = label[2]
        t = label[3:6][:, None]
    r_mat = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, +np.cos(ry)]])
    corners = np.array([[bl / 2, bl / 2, -bl / 2, -bl / 2,
                         bl / 2, bl / 2 + bw / 2, bl / 2],
                        [0, 0, 0, 0, 0, 0, 0],
                        [bw / 2, -bw / 2, -bw / 2, bw / 2,
                         bw / 2, 0, -bw / 2]])
    corners = r_mat @ corners + t
    return corners


def draw_cov(img, mean, covariance, color, thickness=1, cov_scale=8.0):
    """Draw 95% confidence ellipse of a 2-D Gaussian distribution.
    Args:
        mean : array_like
            The mean vector of the Gaussian distribution (ndim=1).
        covariance : array_like
            The 2x2 covariance matrix of the Gaussian distribution.
    """
    # chi2inv(0.95, 2) = 5.9915
    vals, vecs = np.linalg.eigh(5.9915 * cov_scale * covariance)
    indices = vals.argsort()[::-1]
    vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

    center = int(mean[0] + .5), int(mean[1] + .5)
    axes = int(vals[0] + .5), int(vals[1] + .5)
    angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
    cv2.ellipse(
        img, center, axes, angle, 0, 360, color, thickness)


def show_bev(
        img, labels, bbox_results, bbox_3d_results, oc_maps, std_maps, pose_covs,
        cali_mat, width=None, height=None, scale=10, score_thr=0.1, thickness=2,
        cov_scale=8.0):
    """
    Args:
        bbox_3d_results (list[ndarray]): multiclass results,
            in format [l, h, w, x, y, z, ry, score]
    """
    if (width is None) or (height is None):
        height, width = img.shape[:2]
    bev_img = np.full((height, width, 3), 255, dtype=np.uint8)
    origin = np.array([int(width / 16), int(height / 2)])
    # preproc
    fx = cali_mat[0, 0]
    cx = cali_mat[0, 2]
    proj_mat = np.array(
        [[0, scale],
         [scale, 0]])
    # draw FOV line
    fov_line_x_extend = (-100, 100)
    end_pt_left = np.array([fov_line_x_extend[0],
                            -(fx * fov_line_x_extend[0] / cx)])
    end_pt_right = np.array([fov_line_x_extend[1],
                             -(fx * fov_line_x_extend[1] / (cx - img.shape[1] + 1))])
    cv2.line(bev_img, tuple(origin.astype(np.int)),
             tuple((np.round(proj_mat @ end_pt_left) + origin).astype(np.int)),
             (127, 127, 127))
    cv2.line(bev_img, tuple(origin.astype(np.int)),
             tuple((np.round(proj_mat @ end_pt_right) + origin).astype(np.int)),
             (127, 127, 127))
    # draw gt boxes
    if labels is not None:
        for label in labels:
            if label[0] in ['Car', 'Pedestrian', 'Cyclist']:
                corners = compute_box_bev(label)
                corners_bev = (proj_mat @ corners[[0, 2], :]).T + origin
                corners_bev = np.round(corners_bev).astype(np.int32)
                if label[2] == 0:
                    color = (10, 180, 10)
                elif label[2] == 1:
                    color = (160, 160, 10)
                elif label[2] == 2:
                    color = (210, 10, 10)
                else:
                    color = (190, 10, 190)
                cv2.polylines(bev_img, corners_bev.astype(np.int32)[None, ...], False,
                              color, thickness=thickness)
    # draw det results:
    for i in range(len(bbox_3d_results)):
        oc_map = oc_maps[i].transpose((0, 2, 3, 1))  # (n, h, w, 3)
        std_map = std_maps[i].mean(axis=1)  # (n, h, w)
        std_mean = std_map.mean(axis=(1, 2))  # (n, )
        mask = std_map < 2 * std_mean[:, None, None]  # (n, h, w)
        for j in range(len(oc_map)):
            bbox_3d_result = bbox_3d_results[i][j]
            if bbox_3d_result[-1] < score_thr:
                continue
            ry = bbox_3d_result[6]
            r_mat = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                              [0, 1, 0],
                              [-np.sin(ry), 0, +np.cos(ry)]])
            t = bbox_3d_result[3:6]
            oc_pts = oc_map[j][mask[j]]  # (np, 3)
            sort_idx = np.argsort(oc_pts[:, 1])[::-1]
            oc_pts = oc_pts[sort_idx]
            pts = oc_pts @ r_mat.T + t  # (np, 3)
            pts_bev = pts[:, [0, 2]] @ proj_mat.T + origin
            # get rgb
            x1, y1, x2, y2, _ = bbox_results[i][j]
            img_roi = img[round(y1):round(y2), round(x1):round(x2)]
            rgb = cv2.resize(img_roi, (28, 28))[mask[j]][sort_idx]  # (np, 3)
            if thickness > 1:
                pts_bev = np.concatenate(
                    [pts_bev + np.array([-0.5, -0.5]),
                     pts_bev + np.array([-0.5, 0.5]),
                     pts_bev + np.array([0.5, 0.5]),
                     pts_bev + np.array([0.5, -0.5])], axis=1).reshape((-1, 2))
                rgb = np.repeat(rgb, 4, axis=0)
            inlier_mask = (pts_bev[:, 0] > 0) & (pts_bev[:, 0] < width - 1) & (
                    pts_bev[:, 1] > 0) & (pts_bev[:, 1] < height - 1)
            pts_bev = pts_bev[inlier_mask]
            rgb = rgb[inlier_mask]
            # draw pts
            bev_img[
                np.round(pts_bev[:, 1]).astype(np.int), np.round(pts_bev[:, 0]).astype(
                    np.int)] = rgb
            # draw boxes
            corners = compute_box_bev(bbox_3d_result)
            corners_bev = (proj_mat @ corners[[0, 2], :]).T + origin
            cv2.polylines(bev_img, corners_bev.astype(np.int32)[None, ...], False,
                          (10, 60, 240), thickness=thickness)
            # draw covariances
            cov_bev = pose_covs[i][j, [3, 1]][:, [3, 1]] * (scale * scale)
            mean = t[[2, 0]] * scale + origin
            draw_cov(bev_img, mean, cov_bev, (10, 60, 240), thickness=thickness,
                     cov_scale=cov_scale)
    return bev_img


def compute_box_3d(label):
    edge_idx = np.array([[0, 1],
                         [1, 2],
                         [2, 3],
                         [3, 0],
                         [4, 5],
                         [5, 6],
                         [6, 7],
                         [7, 4],
                         [0, 4],
                         [1, 5],
                         [2, 6],
                         [3, 7]])
    corners = np.array([[ 0.5,  0,  0.5],
                        [ 0.5,  0, -0.5],
                        [-0.5,  0, -0.5],
                        [-0.5,  0,  0.5],
                        [ 0.5, -1,  0.5],
                        [ 0.5, -1, -0.5],
                        [-0.5, -1, -0.5],
                        [-0.5, -1,  0.5]])
    if isinstance(label, list):  # gt label
        ry = label[14]
        bl = label[10]
        bw = label[9]
        bh = label[8]
        t = np.array([label[11], label[12], label[13]])
    else:  # det label
        ry = label[6]
        bl = label[0]
        bw = label[2]
        bh = label[1]
        t = label[3:6]
    r_mat = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, +np.cos(ry)]])
    corners *= [bl, bh, bw]
    corners = corners @ r_mat.T + t
    return corners, edge_idx


def draw_box_3d_pred(image, bbox_3d_results, cam_intrinsic, score_thr=0.1, z_clip=0.1,
                     color=(10, 60, 240), thickness=2):
    """
    Args:
        bbox_3d_results (list[ndarray]): multiclass results,
            in format [l, h, w, x, y, z, ry, score]
    """
    bbox_3d_results = np.concatenate(bbox_3d_results, axis=0)
    sort_idx = np.argsort(bbox_3d_results[:, 5])[::-1]
    bbox_3d_results = bbox_3d_results[sort_idx]
    for bbox_3d in bbox_3d_results:
        if bbox_3d[-1] < score_thr:
            continue
        corners, edge_idx = compute_box_3d(bbox_3d)
        corners_in_front = corners[:, 2] >= z_clip
        edges_0_in_front = corners_in_front[edge_idx[:, 0]]
        edges_1_in_front = corners_in_front[edge_idx[:, 1]]
        edges_in_front = edges_0_in_front & edges_1_in_front
        edge_idx_in_front = edge_idx[edges_in_front]
        # project to image
        corners_2d = (proj_to_img(corners, cam_intrinsic, z_clip=z_clip)
                      * 8).astype(np.int)
        if np.any(edges_in_front):
            lines = np.stack([corners_2d[edge_idx_single]
                              for edge_idx_single in edge_idx_in_front],
                             axis=0)  # (n, 2, 2)
            cv2.polylines(image, lines, False, color,
                          thickness=thickness, shift=3)
        # compute intersection
        edges_clipped = edges_0_in_front ^ edges_1_in_front
        if np.any(edges_clipped):
            edge_idx_to_clip = edge_idx[edges_clipped]
            edges_0 = corners[edge_idx_to_clip[:, 0]]
            edges_1 = corners[edge_idx_to_clip[:, 1]]
            z0 = edges_0[:, 2]
            z1 = edges_1[:, 2]
            weight_0 = z1 - z_clip
            weight_1 = z_clip - z0
            intersection = (edges_0 * weight_0[:, None] + edges_1 * weight_1[:, None]
                            ) * (1 / (z1 - z0)).clip(min=-1e6, max=1e6)[:, None]
            keep_idx = np.where(z0 > z_clip,
                                edge_idx_to_clip[:, 0],
                                edge_idx_to_clip[:, 1])
            intersection_2d = (proj_to_img(intersection, cam_intrinsic, z_clip=z_clip)
                               * 8).astype(np.int)  # (n, 2)
            keep_2d = corners_2d[keep_idx]  # (n, 2)
            lines = np.stack([keep_2d, intersection_2d], axis=1)  # (n, 2, 2)
            cv2.polylines(image, lines, False, color,
                          thickness=thickness, shift=3)
    return


def proj_to_img(pts, proj_mat, z_clip=1e-4):
    pts_2d = pts @ proj_mat.T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:].clip(min=z_clip)
    return pts_2d
