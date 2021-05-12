from mmdet.models import DETECTORS, TwoStageDetector


@DETECTORS.register_module()
class TwoStageDetectorMod(TwoStageDetector):

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
