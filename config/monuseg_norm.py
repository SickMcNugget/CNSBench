dataset_type = "MonusegDataset"
data_root = "" #change this
img_dir = "yolo/segment"
ann_dir = "masks/segment"
crop_size = (512,512)
scale = (1024,1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
   # dict(
   #     type='CLAHE',
   #     clip_limit=45.0, # default at 40.0. Higher clip limit is for more aggressive thresholding
   #     ), # Works really well for resolving under segmentation issues
    dict(
        type='RandomResize',
        scale=scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True
        ),
    #dict(
    #    type='RandomRotate', 
    #    prob=0.1, 
    #    degree=(- 20, 20)
    #    ), #Testing this # Did not test well
    
  #  dict(   # Testing this one, has both rotation and flip built in with respective probabilities
   #     type='RandomRotFlip',
    #    rotate_prob=0.5, 
     #   flip_prob=0.5, 
      #  degree=(- 20, 20)
       # ),
    dict(
        type='RandomCrop', 
        crop_size=crop_size, 
        cat_max_ratio=0.75
        ),
   # dict(
   #     type='RandomFlip', 
   #     prob=0.5
   #    ),
    
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=f'{img_dir}/train',
            seg_map_path=f'{ann_dir}/train/png'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=f'{img_dir}/val', 
            seg_map_path=f'{ann_dir}/val/png'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=f'{img_dir}/test', 
            seg_map_path=f'{ann_dir}/test/png'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
