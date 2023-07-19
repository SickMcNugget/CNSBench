img_dir = "yolo_sn"

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path=f'{img_dir}/train')))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path=f'{img_dir}/val')))
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path=f'{img_dir}/test')))