_base_ = ['./mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco.py']

model = dict(
    panoptic_head=dict(
        num_things_classes=1,
        loss_cls=dict(class_weight=[1.0] * 1 + [0.1])
        ),
    panoptic_fusion_head=dict(
        num_things_classes=1,
        ),
    )

dataset_type = 'CocoDataset'
classes = ('blood_vessel',)
data_root='/mmdetection/hubmap_hv'
fold = 0

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        # type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        # ann_file='repos/coco_label/bv_only_dataset12/train.json',
        # ann_file='repos/coco_label/bv_only_dataset12/train_dilated.json',
        ann_file=f'repos/coco_label/bv_only_dataset12_5fold/train_fold{fold}.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train'),
        # pipeline=train_pipeline
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
        # ann_file='repos/coco_label/bv_only_dataset12/val_dilated.json',
        ann_file=f'repos/coco_label/bv_only_dataset12_5fold/val_fold{fold}.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train'),
        # pipeline=test_pipeline
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
        # ann_file='repos/coco_label/bv_only_dataset12/val_dilated.json',
        ann_file=f'repos/coco_label/bv_only_dataset12_5fold/val_fold{fold}.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train'),
        # pipeline=test_pipeline
        )
    )

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/repos/coco_label/bv_only_dataset12_5fold/val_fold{fold}.json',
    metric=[
        'bbox',
        'segm',
    ],
    )
test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/repos/coco_label/bv_only_dataset12_5fold/val_fold{fold}.json',
    metric=[
        'bbox',
        'segm',
    ],
    )

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/mmdetection/hubmap_hv/pretrained/mmdet/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco_20220508_091649-01b0f990.pth'