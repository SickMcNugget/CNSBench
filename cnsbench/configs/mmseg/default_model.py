crop_size = (512, 512)
norm_cfg = dict(type="BN", requires_grad=True)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode="whole"),
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg, num_classes=2),
    auxiliary_head=dict(norm_cfg=norm_cfg, num_classes=2),
)
