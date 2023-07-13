# the new config inherits the base configs to highlight the necessary modification
_base_ = './rtmdet-ins_m_8xb32-300e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1,
    )
)

dataset_type = 'COCODataset'
classes = ('blood_vessel',)
data_root='/mmdetection/hubmap_hv'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        # type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='repos/coco_label/bv_only_dataset12/train.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train')
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        # type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='repos/coco_label/bv_only_dataset12/val.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train')
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
        ann_file='repos/coco_label/bv_only_dataset12/val.json',
        data_prefix=dict(img='hubmap-hacking-the-human-vasculature/train')
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