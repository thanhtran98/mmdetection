# the new config inherits the base configs to highlight the necessary modification
_base_ = './rtmdet-ins_tiny_8xb32-300e_coco.py'

checkpoint = 'hubmap_hv/pretrained/mmdet/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(
        type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    bbox_head=dict(
        num_classes=1,
    )
)


# Imgsz 1024
img_size = (1024,1024)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='CachedMosaic',
        img_scale=img_size,
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(img_size[0]*2, img_size[1]*2),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_size),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=img_size,
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=img_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=img_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    dict(type='Pad', size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]


dataset_type = 'CocoDataset'
classes = ('blood_vessel',)
data_root='/mmdetection/hubmap_hv'

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        # type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        # ann_file='repos/coco_label/bv_only_dataset12/train.json',
        ann_file='repos/coco_label/bv_only_dataset12/train_dilated.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train'),
        pipeline=train_pipeline
        )
    )

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        # type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        # ann_file='repos/coco_label/bv_only_dataset12/val.json',
        ann_file='repos/coco_label/bv_only_dataset12/val_dilated.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train'),
        pipeline=test_pipeline
        )
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        # ann_file='repos/coco_label/bv_only_dataset12/val.json',
        ann_file='repos/coco_label/bv_only_dataset12/val_dilated.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train'),
        pipeline=test_pipeline
        )
    )

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/repos/coco_label/bv_only_dataset12/val.json',
    metric=[
        'bbox',
        'segm',
    ],
    )
test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/repos/coco_label/bv_only_dataset12/val.json',
    metric=[
        'bbox',
        'segm',
    ],
    )

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/mmdetection/hubmap_hv/pretrained/mmdet/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'