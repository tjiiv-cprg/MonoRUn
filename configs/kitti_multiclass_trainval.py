model = dict(
    type='MonoRUnDetector',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPNplus',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        num_lower_outs=1),
    rpn_head=dict(
        type='RPNHeadMod',
        in_channels=256,
        feat_channels=256,
        starting_level=1,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[5],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='MonoRUnRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[2, 4, 8, 16, 32],
            finest_scale=20),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        global_head=dict(
            type='FCExtractorMonteCarlo',
            with_dim=True,
            with_latent_vec=True,
            latent_channels=16,
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            num_classes=3,
            roi_feat_size=7,
            latent_class_agnostic=False,
            loss_dim=dict(
                type='SmoothL1LossMod', loss_weight=1.0, beta=1.0),
            dim_coder=dict(
                type='MultiClassNormDimCoder',
                target_means=[
                    (3.89, 1.53, 1.62),  # car
                    (0.82, 1.78, 0.63),  # pedestrian
                    (1.77, 1.72, 0.57)],  # cyclist
                target_stds=[
                    (0.44, 0.14, 0.11),
                    (0.25, 0.13, 0.12),
                    (0.15, 0.10, 0.14)]),
            dropout_rate=0.5,
            dropout2d_rate=0.2),
        noc_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[2, 4, 8, 16, 32],
            finest_scale=28),
        noc_head=dict(
            type='FCNNOCDecoder',
            num_convs=3,
            roi_feat_size=14,
            in_channels=256,
            conv_kernel_size=3,
            conv_out_channels=256,
            num_classes=3,
            class_agnostic=False,
            upsample_cfg=dict(type='carafe', scale_factor=2),
            num_convs_upsampled=1,
            loss_noc=None,
            noc_channels=3,
            uncert_channels=2,
            dropout2d_rate=0.2,
            flip_correction=True,
            coord_coder=dict(
                type='NOCCoder',
                target_means=(-0.1, -0.5, 0.0),
                target_stds=(0.35, 0.23, 0.34),
                eps=1e-5),
            latent_channels=16),
        projection_head=dict(
            type='UncertProjectionHead',
            loss_proj=dict(
                type='RobustKLLoss',
                loss_weight=1.0,
                momentum=0.1),
            proj_error_coder=dict(
                type='DistanceInvarProjErrorCoder',
                ref_length=1.6,
                ref_focal_y=722,
                target_std=0.15)),
        pose_head=dict(
            type='UncertPropPnPOptimizer',
            pnp=dict(
                type='PnPUncert',
                z_min=0.5,
                epnp_istd_thres=0.6,
                inlier_opt_only=True,
                forward_exact_hessian=False),
            rotation_coder=dict(type='Vec2DRotationCoder'),
            allowed_border=200,
            epnp_ransac_thres_ratio=0.2),
        score_head=dict(
            type='MLPScoreHead',
            reg_fc_out_channels=1024,
            num_pose_fcs=1,
            pose_fc_out_channels=1024,
            fusion_type='add',
            num_fused_fcs=1,
            fc_out_channels=256,
            use_pose_norm=True,
            loss_score=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),)
    ))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=0.5),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.75,
        min_bbox_size=0),
    rcnn=dict(
        bbox_assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.6,
            neg_iou_thr=0.6,
            min_pos_iou=0.6,
            match_low_quality=True,
            ignore_iof_thr=0.6),
        bbox_sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        iou3d_sampler=dict(
            type='IoU3DBalancedSampler',
            pos_iou_thr=0.5,
            pos_fraction_min=0.25,
            pos_fraction_max=0.75,
            smooth_keeprate=True),
        dense_size=28,
        pos_weight=-1,
        calib_scoring=True,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.75,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=100,
        nms_3d_thr=0.01,
        mult_2d_score=True,
        calib_scoring=True,
        cov_correction=True))
dataset_type = 'KITTI3DDataset'
train_data_root = 'data/kitti/training/'
test_data_root = 'data/kitti/testing/'
img_norm_cfg = dict(
    mean=[95.80, 98.72, 93.82], std=[83.11, 81.65, 80.54], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_coord_3d=False,
         with_coord_2d=True),
    dict(type='RandomFlip3D', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),  # use default args
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad3D', size_divisor=32),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_bboxes_3d',
               'coord_2d', 'cam_intrinsic']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='LoadAnnotations3D',
                 with_bbox_3d=False,
                 with_coord_3d=False,
                 with_coord_2d=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad3D', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'coord_2d']),
            dict(type='ToTensor', keys=['cam_intrinsic']),
            dict(type='ToDataContainer', fields=(
                dict(key='cam_intrinsic'), )),
            dict(type='Collect', keys=[
                'img', 'coord_2d', 'cam_intrinsic']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=train_data_root + 'trainval_list.txt',
        img_prefix=train_data_root + 'image_2/',
        label_prefix=train_data_root + 'label_2/',
        calib_prefix=train_data_root + 'calib/',
        meta_prefix=train_data_root + 'img_metas/',
        pipeline=train_pipeline,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        ann_file=test_data_root + 'test_list.txt',
        img_prefix=test_data_root + 'image_2/',
        calib_prefix=test_data_root + 'calib/',
        meta_prefix=test_data_root + 'img_metas/',
        pipeline=test_pipeline,
        filter_empty_gt=False))
# optimizer
optimizer = dict(type='AdamW', lr=2.0e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0.0)
total_epochs = 32
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'faster_rcnn/faster_rcnn_r101_fpn_2x_coco/' \
            'faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True  # prevent distributed deadlock when there's no gt

custom_hooks = [
    dict(
        type='LossUpdaterHook',
        step=[100],
        loss_cfgs=[[
            dict(attr='roi_head.pose_head.loss_calib',
                 type='KLLossMV',
                 loss_weight=0.01)
        ]],
        by_epoch=False),
]
