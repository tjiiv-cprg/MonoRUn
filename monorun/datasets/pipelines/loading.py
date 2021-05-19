import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class LoadAnnotations3D(object):

    def __init__(self,
                 with_bbox_3d=True,
                 with_coord_3d=True,
                 with_coord_2d=True,
                 with_truncation=False,
                 with_depth=False,
                 depth_mean=0.0,
                 depth_std=1.0):
        self.with_bbox_3d = with_bbox_3d
        self.with_coord_3d = with_coord_3d
        self.with_coord_2d = with_coord_2d
        self.with_truncation = with_truncation
        self.with_depth = with_depth
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    @staticmethod
    def _load_coord_3d(results):
        assert results['img_shape'] == results['ori_shape']
        h, w = results['img_shape'][:2]
        oc_dict = mmcv.load(
            osp.join(results['coord_3d_prefix'], results['ann_info']['coord_3d']))
        gt_coords_3d = []
        gt_coords_3d_mask = []
        for i, bbox_3d in zip(results['ann_info']['object_ids'],
                              results['ann_info']['bboxes_3d']):
            uv = np.round(oc_dict['uv_list'][i]).astype(np.int)
            oc = oc_dict['oc_list'][i].astype(np.float32)
            coord_3d = np.zeros((h, w, 3), dtype=np.float32)
            coord_3d_mask = np.zeros((h, w, 1), dtype=np.float32)
            coord_3d[uv[:, 1], uv[:, 0]] = oc
            coord_3d_mask[uv[:, 1], uv[:, 0]] = 1.0
            gt_coords_3d.append(coord_3d)
            gt_coords_3d_mask.append(coord_3d_mask)
        results['gt_coords_3d'] = gt_coords_3d
        results['gt_coords_3d_mask'] = gt_coords_3d_mask
        results['dense_fields'].append('gt_coords_3d')
        results['dense_fields'].append('gt_coords_3d_mask')
        return results

    @staticmethod
    def _load_bboxes_3d(results):
        results['gt_bboxes_3d'] = results['ann_info']['bboxes_3d']
        results['bbox_3d_fields'].append('gt_bboxes_3d')
        results['gt_proj_r_mats'] = results['ann_info']['gt_proj_r_mats']
        results['gt_proj_t_vecs'] = results['ann_info']['gt_proj_t_vecs']
        return results

    def _load_depth(self, results):
        depth = mmcv.imread(
            osp.join(results['depth_prefix'],
                     results['ann_info']['depth']),
            flag='unchanged')[..., None]  # (H, W, 1)
        results['depth'] = (depth.astype(np.float32) - self.depth_mean) / self.depth_std
        results['dense_fields'].append('depth')
        return results

    @staticmethod
    def _gen_coord_2d(results):
        assert results['img_shape'] == results['ori_shape']
        h, w = results['img_shape'][:2]
        coord_2d = np.mgrid[:h, :w].astype(np.float32)
        coord_2d[[1, 0]] = coord_2d[[0, 1]]  # to [u, v]
        results['coord_2d'] = np.moveaxis(coord_2d, 0, -1)  # (H, W, 2)
        if 'dense_fields' in results:
            results['dense_fields'].append('coord_2d')
        else:
            results['dense_fields'] = ['coord_2d']
        return results

    def __call__(self, results):
        if 'ann_info' in results and 'cam_intrinsic' in results['ann_info']:
            results['cam_intrinsic'] = results['ann_info']['cam_intrinsic']
        elif 'img_info' in results and 'cam_intrinsic' in results['img_info']:
            results['cam_intrinsic'] = results['img_info']['cam_intrinsic']
        else:
            raise ValueError('cam_intrinsic not found')
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_coord_3d:
            results = self._load_coord_3d(results)
        if self.with_coord_2d:
            results = self._gen_coord_2d(results)
        if self.with_truncation:
            results['truncation'] = results['ann_info']['truncation']
        if self.with_depth:
            results = self._load_depth(results)
        return results
