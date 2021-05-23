import numpy as np
import torch
from packaging import version

from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmcv.ops.roi_align import roi_align
from mmdet import __version__
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead


@HEADS.register_module()
class MonoRUnRoIHead(StandardRoIHead):

    def __init__(self,
                 noc_roi_extractor=None,
                 noc_head=None,
                 global_head=None,
                 projection_head=None,
                 pose_head=None,
                 score_head=None,
                 debug=False,
                 **kwargs):
        super(MonoRUnRoIHead, self).__init__(**kwargs)
        assert not self.with_mask
        assert not self.with_shared_head
        if noc_head is not None:
            self.init_noc_head(noc_roi_extractor, noc_head)
        if global_head is not None:
            self.global_head = build_head(global_head)
        if projection_head is not None:
            self.projection_head = build_head(projection_head)
        if pose_head is not None:
            self.pose_head = build_head(pose_head)
        if score_head is not None:
            score_head.update(train_cfg=self.train_cfg)
            self.score_head = build_head(score_head)
        self.debug = debug
        self.new_version = version.parse(__version__) >= version.parse('2.4.0')

    @property
    def with_noc(self):
        return hasattr(self, 'noc_head') and self.noc_head is not None

    @property
    def with_reg(self):
        return hasattr(self, 'global_head') and self.global_head is not None

    @property
    def with_projection(self):
        return hasattr(
            self, 'projection_head') and self.projection_head is not None

    @property
    def with_pose(self):
        return hasattr(self, 'pose_head') and self.pose_head is not None

    @property
    def with_score(self):
        return hasattr(self, 'score_head') and self.score_head is not None

    def init_noc_head(self, noc_roi_extractor, noc_head):
        assert noc_roi_extractor is not None
        self.noc_roi_extractor = build_roi_extractor(noc_roi_extractor)
        self.noc_head = build_head(noc_head)

    def init_assigner_sampler(self):
        self.bbox_assigner = self.bbox_sampler = None
        self.bbox_refined_assigner = self.bbox_refined_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.bbox_assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.bbox_sampler, context=self)
            if hasattr(self.train_cfg, 'bbox_refined_assigner'):
                self.bbox_refined_assigner = build_assigner(
                    self.train_cfg.bbox_refined_assigner)
            if hasattr(self.train_cfg, 'bbox_refined_sampler'):
                self.bbox_refined_sampler = build_sampler(
                    self.train_cfg.bbox_refined_sampler, context=self)

    def init_weights(self, pretrained):
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_noc:
            self.noc_head.init_weights()
            self.noc_roi_extractor.init_weights()
        if self.with_reg:
            self.global_head.init_weights()
        if self.with_pose:
            self.pose_head.init_weights()
        if self.with_score:
            self.score_head.init_weights()

    def forward_dummy(self, x, proposals):
        raise NotImplementedError

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_bboxes_3d=None,
                      gt_coords_3d=None,
                      gt_coords_3d_mask=None,
                      coord_2d=None,
                      cam_intrinsic=None,
                      gt_proj_r_mats=None,
                      gt_proj_t_vecs=None):
        assert self.with_bbox and self.with_noc and self.with_reg

        num_imgs = len(img_metas)

        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        bbox_results = self._bbox_forward_train(x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                img_metas)
        losses.update(bbox_results['loss_bbox'])

        roi_labels = bbox_results['bbox_targets'][0]
        # refine bboxes
        if self.bbox_refined_assigner is not None \
                and self.bbox_refined_sampler is not None:
            # refine bboxes like Cascade R-CNN
            pos_is_gts = [res.pos_is_gt for res in sampling_results]
            with torch.no_grad():
                roi_labels = torch.where(
                    roi_labels == self.bbox_head.num_classes,
                    bbox_results['cls_score'][:, :-1].argmax(1),
                    roi_labels)
                proposal_list = self.bbox_head.refine_bboxes(
                    bbox_results['rois'], roi_labels,
                    bbox_results['bbox_pred'], pos_is_gts, img_metas)

            # assign and sample for 3D R-CNN
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_refined_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_refined_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        else:
            # only refine pos bboxes (without filtering gts) and do not resample
            with torch.no_grad():
                for i, img_meta in enumerate(img_metas):
                    mask = bbox_results['rois'][:, 0] == i
                    roi_labels_i = roi_labels[mask]
                    pos_keep = roi_labels_i < self.bbox_head.num_classes
                    pos_rois = bbox_results['rois'][mask, 1:][pos_keep]
                    pos_bbox_pred = bbox_results['bbox_pred'][mask][pos_keep]
                    pos_label = roi_labels_i[pos_keep]
                    pos_bboxes = self.bbox_head.regress_by_class(
                        pos_rois, pos_label, pos_bbox_pred, img_meta)
                    sampling_results[i].bboxes[pos_keep] = pos_bboxes
                    sampling_results[i].pos_bboxes = pos_bboxes

        # reg head forward and loss
        reg_results = self._reg_forward_train(
            x, sampling_results, gt_bboxes_3d, gt_labels)
        losses.update(reg_results['loss_reg'])

        # noc head forward and loss
        flip = []
        for img_meta, res in zip(img_metas, sampling_results):
            flip += [img_meta['flip']] * len(res.pos_inds)
        noc_results = self._noc_forward_train(
            x, sampling_results, reg_results['latent_pred'],
            reg_results['latent_var'], gt_coords_3d, gt_coords_3d_mask,
            gt_bboxes_3d, img_metas, flip)
        losses.update(noc_results['loss_noc'])

        # decode coords_3d and get coords_2d
        if self.with_projection or self.with_pose:
            dimensions_pred, dimensions_var = \
                self.global_head.dim_coder.decode(
                    reg_results['dim_pred'],
                    reg_results['dim_var'],
                    reg_results['pos_gt_labels'])
            coords_3d_pred, coords_3d_var = self.noc_head.coord_coder.decode(
                noc_results['noc_pred'], noc_results['noc_var'],
                dimensions_pred, dimensions_var, flip)
            if noc_results['rois'].size(0) == 0:
                coords_2d_roi = noc_results['noc_pred'].new_zeros(
                    (0, 2, ) + noc_results['noc_pred'].shape[-2:])
            else:
                coords_2d_roi = roi_align(
                    coord_2d, noc_results['rois'],
                    noc_results['noc_pred'].shape[-2:], 1.0, 0, 'avg', True)
        else:
            coords_3d_pred = coords_3d_var = \
                dimensions_pred = coords_2d_roi = None

        # projection head forward and loss
        if self.with_projection:
            projection_results = self._projection_forward_train(
                coords_3d_pred,
                noc_results['proj_logstd'],
                coords_3d_var, sampling_results,
                coords_2d_roi, gt_proj_r_mats, gt_proj_t_vecs,
                gt_bboxes_3d, img_metas)
            losses.update(projection_results['loss_proj'])
        else:
            projection_results = dict(proj_logstd=noc_results['proj_logstd'])

        if self.with_pose:
            pose_results = self._pose_forward_train(
                coords_2d_roi, projection_results['proj_logstd'],
                coords_3d_pred,
                dimensions_pred,
                projection_results['img_shapes'],
                sampling_results,
                projection_results['pos_bboxes_3d'], cam_intrinsic)
            losses.update(pose_results['loss_pose'])

            if self.with_score:
                score_results = self._score_forward_train(
                    reg_results['reg_fc_out'],
                    pose_results['ret_val'],
                    pose_results['yaw_pred'],
                    pose_results['t_vec_pred'],
                    pose_results['pose_cov_calib'] if getattr(
                        self.train_cfg, 'calib_scoring', False
                    ) else pose_results['pose_cov_pred'],
                    dimensions_pred,
                    pose_results['ious'])
                losses.update(score_results['loss_score'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _reg_forward(self, x, rois, labels):
        reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        dim_latent_pred, dim_latent_var, distance_pred, distance_logstd, reg_fc_out = \
            self.global_head(reg_feats)
        dim_pred, dim_var, latent_pred, latent_var = self.global_head.slice_pred(
            dim_latent_pred, dim_latent_var, labels)
        reg_results = dict(
            dim_pred=dim_pred,
            dim_var=dim_var,
            latent_pred=latent_pred,
            latent_var=latent_var,
            distance_pred=distance_pred,
            distance_logstd=distance_logstd,
            roi_feats=reg_feats,
            reg_fc_out=reg_fc_out)
        return reg_results

    def _reg_forward_train(self, x, sampling_results,
                           gt_bboxes_3d, gt_labels):

        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        pos_gt_labels = torch.cat(
            [res.pos_gt_labels for res in sampling_results], dim=0)
        reg_results = self._reg_forward(x, pos_rois, pos_gt_labels)

        reg_targets = self.global_head.get_targets(
            sampling_results, gt_bboxes_3d, gt_labels)

        loss_reg = self.global_head.loss(
            reg_results['dim_pred'],
            reg_results['distance_pred'],
            reg_results['distance_logstd'],
            *reg_targets)

        if reg_results['distance_pred'] is not None \
                and reg_results['distance_pred'].size(1):
            distance_pred = reg_results['distance_pred']
            distance_logstd = reg_results['distance_logstd']
            if not self.global_head.reg_decoded_distance:
                distance_pred, distance_logstd = \
                    self.global_head.distance_coder.decode(
                        distance_pred, distance_logstd)
        else:
            distance_pred = distance_logstd = None

        reg_results.update(
            loss_reg=loss_reg,
            distance_pred=distance_pred,
            distance_logstd=distance_logstd,
            pos_gt_labels=pos_gt_labels)
        if self.debug:
            reg_results['dim_pred'] = reg_targets[0]

        return reg_results

    def _noc_forward(self, x, rois, latent_pred, latent_var, labels, flip):
        # unlike 2d bboxes, class label must be pre-determined before
        # 3d localization (for efficiency)
        noc_feats = self.noc_roi_extractor(
            x[:self.noc_roi_extractor.num_inputs], rois)
        noc_pred, noc_var, proj_logstd, regular_params = self.noc_head(
            noc_feats, latent_pred, latent_var, labels, flip=flip)
        noc_results = dict(
            noc_pred=noc_pred,
            noc_var=noc_var,
            proj_logstd=proj_logstd,
            regular_params=regular_params,
            rois=rois)
        return noc_results

    def _noc_forward_train(
            self, x, sampling_results, latent_pred, latent_var,
            gt_coords_3d, gt_coords_3d_mask, gt_bboxes_3d, img_metas, flip):
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        pos_labels = torch.cat(
            [res.pos_gt_labels for res in sampling_results])
        noc_results = self._noc_forward(
            x, pos_rois, latent_pred, latent_var, pos_labels, flip)
        if self.noc_head.loss_noc is not None:
            noc_targets, noc_weights = self.noc_head.get_targets(
                sampling_results, gt_coords_3d, gt_coords_3d_mask,
                gt_bboxes_3d, self.train_cfg, img_metas)
            loss_noc = self.noc_head.loss(
                noc_results['noc_pred'], noc_targets, noc_weights)
            if self.debug:
                noc_results['noc_pred'] = noc_targets
                noc_results['proj_logstd'] = (
                    1 / noc_weights.clamp(min=1e-6, max=1e6)
                ).log().repeat(1, 2, 1, 1)
        else:
            loss_noc = dict()
        noc_results.update(loss_noc=loss_noc)
        return noc_results

    def _projection_decode(self, proj_logstd, coords_3d_var, distances=None):
        """Decode logstd of encoded proj in test time."""
        if self.projection_head.train_std_of_encoded_error:
            proj_logstd = self.projection_head.proj_error_coder.decode_logstd(
                proj_logstd, coords_3d_var, distances)
        return dict(proj_logstd=proj_logstd)

    def _projection_forward_train(
            self, coords_3d_pred, proj_logstd, coords_3d_var, sampling_results,
            coords_2d_roi, gt_proj_r_mats, gt_proj_t_vecs,
            gt_bboxes_3d, img_metas):
        (proj_r_mats,
         proj_t_vecs,
         pos_bboxes_3d,
         pos_distances,
         img_shapes) = self.projection_head.get_properties(
            sampling_results, gt_proj_r_mats, gt_proj_t_vecs,
            gt_bboxes_3d, img_metas)
        coords_2d_proj = self.projection_head(
            coords_3d_pred, proj_r_mats, proj_t_vecs, img_shapes)
        loss_proj = self.projection_head.loss(
            coords_2d_proj, proj_logstd, coords_2d_roi, pos_distances)
        projection_results = dict(loss_proj=loss_proj,
                                  img_shapes=img_shapes,
                                  pos_bboxes_3d=pos_bboxes_3d)
        projection_results.update(
            self._projection_decode(
                proj_logstd, coords_3d_var, pos_distances))
        return projection_results

    def _pose_forward(self, coords_2d, proj_logstd, coords_3d_pred,
                      cam_intrinsic, img_shapes):
        ret_val, yaw_pred, t_vec_pred, pose_cov_pred, pose_cov_calib = \
            self.pose_head(coords_2d, proj_logstd, coords_3d_pred, cam_intrinsic,
                           img_shapes)
        pose_results = dict(
            ret_val=ret_val,
            yaw_pred=yaw_pred,
            t_vec_pred=t_vec_pred,
            pose_cov_pred=pose_cov_pred,
            pose_cov_calib=pose_cov_calib)
        return pose_results

    def _pose_forward_train(
            self, coords_2d, proj_logstd, coords_3d_pred, dimensions_pred, img_shapes,
            sampling_results, pos_bboxes_3d, cam_intrinsic):
        pos_cam_intrinsic = []
        for cam_intrinsic_single, res in zip(
                cam_intrinsic, sampling_results):
            pos_cam_intrinsic += [cam_intrinsic_single] * len(res.pos_inds)
        if len(pos_cam_intrinsic):
            pos_cam_intrinsic = torch.stack(pos_cam_intrinsic, dim=0)
        else:
            pos_cam_intrinsic = coords_2d.new_zeros((0, 3, 3))
        pose_results = self._pose_forward(
            coords_2d, proj_logstd, coords_3d_pred, pos_cam_intrinsic, img_shapes)
        yaw_targets, trans_targets = self.pose_head.get_targets(
            pos_bboxes_3d)
        loss_pose, ious = self.pose_head.loss(
            pose_results['ret_val'],
            pose_results['yaw_pred'],
            pose_results['t_vec_pred'],
            pose_results['pose_cov_calib'],
            dimensions_pred,
            yaw_targets, trans_targets, pos_bboxes_3d)
        pose_results.update(loss_pose=loss_pose, ious=ious)
        return pose_results

    def _score_forward(self, reg_fc_out, yaw, t_vec, pose_cov, dimensions):
        scores = self.score_head(reg_fc_out, yaw, t_vec, pose_cov, dimensions)
        return dict(scores=scores)

    def _score_forward_train(self, reg_fc_out, ret_val, yaw, t_vec,
                             pose_cov, dimensions, ious):
        score_results = self._score_forward(
            reg_fc_out[ret_val], yaw[ret_val], t_vec[ret_val], pose_cov[ret_val],
            dimensions[ret_val])
        loss_score = self.score_head.loss(score_results['scores'], ious[ret_val])
        score_results.update(loss_score=loss_score)
        return score_results

    async def async_simple_test(self, *args, **kwargs):
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    coord_2d=None,
                    cam_intrinsic=None,
                    rescale=False):
        assert self.with_bbox and self.with_noc \
               and self.with_pose and self.with_score
        assert len(img_metas) == 1, 'batch inference is not supported yet'

        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        rcnn_test_cfg = self.test_cfg
        coord_2d = coord_2d[0]
        cam_intrinsic = cam_intrinsic[0][0][None, ...]

        rois = bbox2roi(proposal_list)
        bbox_results = self._bbox_forward(x, rois)
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)

        if det_bboxes.shape[0] > 0:
            # if det_bboxes is rescaled to the original image size, we need
            # to rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * det_bboxes.new_tensor(scale_factor)
                if rescale else det_bboxes)
            bbox_3d_rois = bbox2roi([_bboxes])
        else:
            bbox_3d_rois = None

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head.num_classes)

        # bbox_3d test
        if bbox_3d_rois is None:
            bbox_3d_result = [np.zeros((0, 8), dtype=np.float32)
                              for _ in range(self.noc_head.num_classes)]
        else:
            # reg forward
            reg_results = self._reg_forward(x, bbox_3d_rois, det_labels)
            # decode distance
            if reg_results['distance_pred'] is not None \
                    and reg_results['distance_pred'].size(1):
                distance_pred = reg_results['distance_pred']
                distance_logstd = reg_results['distance_logstd']
                if not self.global_head.reg_decoded_distance:
                    distance_pred, distance_logstd = \
                        self.global_head.distance_coder.decode(
                            distance_pred, distance_logstd)
            else:
                distance_pred = distance_logstd = None
            # decode dimension
            dimensions_pred, dimensions_var = \
                self.global_head.dim_coder.decode(
                    reg_results['dim_pred'],
                    reg_results['dim_var'],
                    det_labels)
            # noc forward
            noc_results = self._noc_forward(
                x, bbox_3d_rois, reg_results['latent_pred'],
                reg_results['latent_var'], det_labels, img_metas[0]['flip'])
            # coord decode
            coords_3d_pred, coords_3d_var = self.noc_head.coord_coder.decode(
                noc_results['noc_pred'], noc_results['noc_var'],
                dimensions_pred, dimensions_var, img_metas[0]['flip'])
            proj_logstd = self._projection_decode(
                noc_results['proj_logstd'],
                coords_3d_var,
                distance_pred)['proj_logstd']
            # prepare coords_2d
            coords_2d_roi = roi_align(
                coord_2d, bbox_3d_rois,
                noc_results['noc_pred'].shape[-2:], 1.0, 0, 'avg', True)
            # pose forward (PnP)
            pose_results = self._pose_forward(
                coords_2d_roi, proj_logstd,
                coords_3d_pred,
                cam_intrinsic,
                cam_intrinsic.new_tensor(img_shape)[None, ...])
            if rcnn_test_cfg.cov_correction:
                distance = self.projection_head.get_distance(pose_results['t_vec_pred'])
                pose_results['pose_cov_calib'] = \
                    self.projection_head.proj_error_coder.cov_correction(
                        pose_results['pose_cov_calib'], distance)
            # score forward
            score_results = self._score_forward(
                reg_results['reg_fc_out'],
                pose_results['yaw_pred'],
                pose_results['t_vec_pred'],
                pose_results['pose_cov_calib'] if getattr(
                    rcnn_test_cfg, 'calib_scoring', False
                ) else pose_results['pose_cov_pred'],
                dimensions_pred)
            if self.score_head.pre_sigmoid:
                score_results['scores'].sigmoid_()
            score_results['scores'][~pose_results['ret_val']] = 0
            # multiply cls with loc score
            scores = det_bboxes[:, -1] * score_results['scores'] \
                if rcnn_test_cfg.mult_2d_score \
                else score_results['scores']
            # get bbox_3d_result
            bbox_3d_result = self.get_bbox_3d_result(
                dimensions_pred,
                pose_results['yaw_pred'],
                pose_results['t_vec_pred'],
                scores,
                det_labels,
                to_np=False)
            # do 3D NMS and select 2D boxes
            bbox_3d_result, keep_inds_3d = self.multiclass_3d_result_nms(
                bbox_3d_result, rcnn_test_cfg.nms_3d_thr, to_np=True)
            bbox_result = [
                bbox_result_single[keep_inds_3d_single]
                for bbox_result_single, keep_inds_3d_single in zip(
                    bbox_result, keep_inds_3d)]
        results = dict(bbox_results=bbox_result, bbox_3d_results=bbox_3d_result)
        # return noc and std maps
        if getattr(rcnn_test_cfg, 'debug', False):
            if det_bboxes.shape[0] == 0:
                oc_maps = [np.zeros(
                    (0, 3, 0, 0), dtype=np.float32)] * self.bbox_head.num_classes
                std_maps = [np.zeros(
                    (0, self.noc_head.uncert_channels, 0, 0),
                    dtype=np.float32)] * self.bbox_head.num_classes
                pose_covs = [np.zeros(
                    (0, 4, 4), dtype=np.float32)] * self.bbox_head.num_classes
                latent_vecs = [np.zeros(
                    (0, 16), dtype=np.float32)] * self.bbox_head.num_classes
            else:
                oc_pred = coords_3d_pred.cpu().numpy()
                std = torch.exp(proj_logstd).cpu().numpy()
                pose_cov = pose_results['pose_cov_calib'].cpu().numpy()
                det_labels = det_labels.cpu().numpy()
                latent = reg_results['latent_pred'].cpu().numpy()
                oc_maps = []
                std_maps = []
                pose_covs = []
                latent_vecs = []
                for i in range(self.bbox_head.num_classes):
                    class_mask = det_labels == i
                    keep_inds_3d_single = keep_inds_3d[i]
                    oc_maps.append(
                        oc_pred[class_mask][keep_inds_3d_single])
                    std_maps.append(
                        std[class_mask][keep_inds_3d_single])
                    pose_covs.append(
                        pose_cov[class_mask][keep_inds_3d_single])
                    latent_vecs.append(
                        latent[class_mask][keep_inds_3d_single])
            results.update(oc_maps=oc_maps, std_maps=std_maps,
                           pose_covs=pose_covs, latent_vecs=latent_vecs)
        if self.new_version:
            return [results]
        else:
            return results

    def aug_test(self, *args, **kwargs):
        raise NotImplementedError

    def get_bbox_3d_result(self, dimensions, yaw, t_vec, scores, labels,
                           to_np=False):
        bboxes_3d = torch.cat(
            (dimensions, t_vec, yaw, scores.unsqueeze(1)), dim=1)
        if to_np:
            bboxes_3d = bboxes_3d.cpu().numpy()
            labels = labels.cpu().numpy()
        return [bboxes_3d[labels == i] for i in range(self.bbox_head.num_classes)]

    def multiclass_3d_result_nms(self, bbox_3d_result, nms_thr=0.25, to_np=True):
        """
        Args:
            bbox_3d_result (list[Tensor]): tensor shape (N, 8),
                in format [l, h, w, x, y, z, ry, score]
            nms_thr (float):
            to_np (bool):

        Returns:
            bbox_3d_result_out (list[Tensor | ndarray]):
            keep_inds_3d (list[Tensor | ndarray]):
        """
        bbox_3d_result_out = []
        keep_inds_3d = []
        for bbox_3d_single in bbox_3d_result:
            n = bbox_3d_single.size(0)
            if n > 1:
                boxes_for_nms = self.xywhr2xyxyr(
                    bbox_3d_single[:, [3, 5, 0, 2, 6]])
                keep_inds_single = nms_gpu(
                    boxes_for_nms, bbox_3d_single[:, 7], nms_thr)
                if to_np:
                    bbox_3d_result_out.append(
                        bbox_3d_single[keep_inds_single].cpu().numpy())
                    keep_inds_3d.append(keep_inds_single.cpu().numpy())
                else:
                    bbox_3d_result_out.append(bbox_3d_single[keep_inds_single])
                    keep_inds_3d.append(keep_inds_single)
            else:
                if to_np:
                    bbox_3d_result_out.append(bbox_3d_single.cpu().numpy())
                    keep_inds_3d.append(np.zeros(n, dtype=np.int64))
                else:
                    bbox_3d_result_out.append(bbox_3d_single)
                    keep_inds_3d.append(
                        bbox_3d_single.new_zeros((n, ), dtype=torch.int64))
        return bbox_3d_result_out, keep_inds_3d

    @staticmethod
    def xywhr2xyxyr(boxes_xywhr):
        """Convert a rotated boxes in XYWHR format to XYXYR format.

        Args:
            boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

        Returns:
            torch.Tensor: Converted boxes in XYXYR format.
        """
        boxes = torch.zeros_like(boxes_xywhr)
        half_w = boxes_xywhr[:, 2] / 2  # l in bbox_3d
        half_h = boxes_xywhr[:, 3] / 2  # w in bbox_3d
        # x in cam coord
        boxes[:, 0] = boxes_xywhr[:, 0] - half_w
        # z in cam coord, mirrored_direction
        boxes[:, 1] = boxes_xywhr[:, 1] - half_h
        boxes[:, 2] = boxes_xywhr[:, 0] + half_w
        boxes[:, 3] = boxes_xywhr[:, 1] + half_h
        boxes[:, 4] = boxes_xywhr[:, 4]
        return boxes
