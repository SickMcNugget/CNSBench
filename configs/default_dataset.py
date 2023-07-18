img_dir = "yolo"
ann_dir = "masks"
crop_size = (512,512)
scale = (1024,1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
         scale=scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True
        ),
    dict(
        type='RandomCrop', 
         crop_size=crop_size, 
        cat_max_ratio=0.75
        ),
    dict(
        type='RandomFlip', 
        prob=0.5
       ),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
#   batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
#       type=dataset_type,
#       data_root=data_root,
        data_prefix=dict(
            img_path=f'{img_dir}/train',
            seg_map_path=f'{ann_dir}/train/'),
        pipeline=train_pipeline))
val_dataloader = dict(
#   batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
#       type=dataset_type,
#       data_root=data_root,
        data_prefix=dict(
            img_path=f'{img_dir}/val', 
            seg_map_path=f'{ann_dir}/val/'),
        pipeline=test_pipeline))
test_dataloader = dict(
#   batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
#       type=dataset_type,
#       data_root=data_root,
        data_prefix=dict(
            img_path=f'{img_dir}/test', 
            seg_map_path=f'{ann_dir}/test/'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
