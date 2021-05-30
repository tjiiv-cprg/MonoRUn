import numpy as np
import mmcv
from mmdet.models import DETECTORS, TwoStageDetector
from monorun.core import draw_box_3d_pred, show_bev


@DETECTORS.register_module()
class MonoRUnDetector(TwoStageDetector):

    def simple_test(self, img, img_metas, proposals=None, rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale, **kwargs)

    def show_result(self,
                    img,
                    cam_intrinsic,
                    result,
                    score_thr=0.3,
                    cov_scale=5.0,
                    bev_scale=25,
                    thickness=2,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    views=['camera', 'bev']):
        img = mmcv.imread(img)
        img_show = []
        if 'camera' in views:
            img_pred_3d = img.copy()
            draw_box_3d_pred(
                img_pred_3d,
                result['bbox_3d_results'],
                cam_intrinsic,
                score_thr=score_thr)
            img_show.append(img_pred_3d)
        if 'bev' in views:
            viz_bev = show_bev(
                img, None, result['bbox_results'], result['bbox_3d_results'],
                cam_intrinsic, width=img.shape[1], height=img.shape[0], scale=bev_scale,
                oc_maps=result.get('oc_maps'),
                std_maps=result.get('std_maps'),
                pose_covs=result.get('pose_covs'),
                cov_scale=cov_scale,
                score_thr=score_thr, thickness=2)
            img_show.append(viz_bev)
        if len(img_show) == 1:
            img_show = img_show[0]
        elif len(img_show) == 2:
            img_show = np.concatenate(img_show, axis=0)
        else:
            raise ValueError('no view to show')
        if show:
            mmcv.imshow(img_show, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img_show, out_file)
        if not (show or out_file):
            return img_show
