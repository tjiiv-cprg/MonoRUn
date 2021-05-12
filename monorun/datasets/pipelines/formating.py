import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.formating import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):

    def __call__(self, results):
        results = super().__call__(results)
        for key in ['gt_bboxes_3d', 'gt_pose_r_mats', 'gt_pose_t_vecs',
                    'gt_proj_r_mats', 'gt_proj_t_vecs', 'cali_k_mat', 'cam_t_vec']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_coords_3d' in results:
            gt_coords_3d = results['gt_coords_3d']
            gt_coords_3d = [
                np.ascontiguousarray(gt_coord_3d.transpose(2, 0, 1))
                for gt_coord_3d in gt_coords_3d]
            results['gt_coords_3d'] = DC(to_tensor(gt_coords_3d))
        if 'gt_coords_3d_mask' in results:
            gt_coords_3d_mask = results['gt_coords_3d_mask']
            gt_coords_3d_mask = [
                np.ascontiguousarray(gt_coord_3d_mask.transpose(2, 0, 1))
                if len(gt_coord_3d_mask.shape) == 3
                else gt_coord_3d_mask[None, ...]
                for gt_coord_3d_mask in gt_coords_3d_mask]
            results['gt_coords_3d_mask'] = DC(to_tensor(gt_coords_3d_mask))
        if 'coord_2d' in results:
            coord_2d = results['coord_2d']
            coord_2d = np.ascontiguousarray(coord_2d.transpose(2, 0, 1))
            results['coord_2d'] = DC(to_tensor(coord_2d), stack=True)
        if 'depth' in results:
            depth = results['depth'][None, ...]
            results['depth'] = DC(to_tensor(depth), stack=True)
        return results
