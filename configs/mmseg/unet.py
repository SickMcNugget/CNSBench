_base_ = ['mmseg::_base_/models/fcn_unet_s5-d16.py']

crop_size = (512,512)
model = dict(
    test_cfg = dict(mode="slide", crop_size=crop_size, stride=(170,170))
)