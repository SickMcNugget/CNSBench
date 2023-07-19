import argparse
import datetime
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import Config
from mmengine.runner import Runner
from pathlib import Path

def register_datasets():
    @DATASETS.register_module()
    class MoNuSegDataset(BaseSegDataset):
        """MoNuSeg dataset."""
        METAINFO = dict(
            classes = ('background', 'nucleus'),
            palette = [[0, 0, 0], [0, 255, 0]] # This has been changed, lets see what it does
        )
    
        def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
            super().__init__(
                img_suffix=img_suffix, 
                seg_map_suffix=seg_map_suffix,
                reduce_zero_label=reduce_zero_label,
                **kwargs)
    
    @DATASETS.register_module()
    class MoNuSACDataset(BaseSegDataset):
        """MoNuSAC dataset."""
        METAINFO = dict(
            classes = ('background', 'nucleus'),
            palette = [[0, 0, 0], [0, 255, 0]] # This has been changed, lets see what it does
        )
    
        def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
            super().__init__(
                img_suffix=img_suffix, 
                eg_map_suffix=seg_map_suffix,
                reduce_zero_label=reduce_zero_label,
                **kwargs)
        
    @DATASETS.register_module()
    class CryoNuSegDataset(BaseSegDataset):
        """CryoNuSeg dataset."""
        METAINFO = dict(
            classes = ('background', 'nucleus'),
            palette = [[0, 0, 0], [0, 255, 0]] # This has been changed, lets see what it does
        )
    
        def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
            super().__init__(
                img_suffix=img_suffix, 
                seg_map_suffix=seg_map_suffix,
                reduce_zero_label=reduce_zero_label,
                **kwargs)
    
    @DATASETS.register_module()
    class TNBCDataset(BaseSegDataset):
        """TNBC dataset."""
        METAINFO = dict(
            classes = ('background', 'nucleus'),
            palette = [[0, 0, 0], [0, 255, 0]] # This has been changed, lets see what it does
        )
    
        def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
            super().__init__(
                img_suffix=img_suffix, 
                seg_map_suffix=seg_map_suffix,
                reduce_zero_label=reduce_zero_label,
                **kwargs)

def get_filename(args: argparse.Namespace):
    if args.model == "unet":
        model = "unet_s5-d16"
    elif args.model == "deeplabv3plus":
        model = "deeplabv3plus_r50_d8"

    return f"{model}_1xb{args.batch}-{args.iterations//1000}k_{args.dataset.lower()}-512x512.py"

def main(args: argparse.Namespace):
    # make sure the datasets are registered
    register_datasets() 

    configs_dir = Path("configs") / "mmseg"
    # -- grab the config location based on arguments -- #
    default_cfg = Config.fromfile(configs_dir / "default.py")
    dataset_cfg = Config.fromfile(configs_dir / "default_dataset.py")
    default_model_cfg = Config.fromfile(configs_dir / "default_model.py")
    model_cfg = Config.fromfile(configs_dir / f"{args.model.lower()}.py")
    if args.normalised:
        normalised_cfg = Config.fromfile(configs_dir / "normalised_dataset.py")

    cfg = Config()
    cfg.merge_from_dict(default_cfg.to_dict())
    cfg.merge_from_dict(dataset_cfg.to_dict())
    cfg.merge_from_dict(default_model_cfg.to_dict())
    cfg.merge_from_dict(model_cfg.to_dict())
    if args.normalised:
        cfg.merge_from_dict(normalised_cfg.to_dict())
    
    # -- Fill in dataset settings -- #
    cfg.dataset_type = f"{args.dataset}Dataset"
    cfg.data_root = str(args.dataset_root / args.dataset)
    cfg.train_dataloader.dataset.type = cfg.dataset_type
    cfg.train_dataloader.dataset.data_root = cfg.data_root

    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root

    cfg.test_dataloader.dataset.type = cfg.dataset_type
    cfg.test_dataloader.dataset.data_root = cfg.data_root

    # -- set batch size for model -- #
    cfg.train_dataloader.batch_size = args.batch

    # -- Create the work directory -- #
    now = datetime.datetime.now()
    work_dir = f"./work_dirs/{args.dataset}/{args.model}/{now.year}-{now.month:02d}-{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}/"
    cfg.work_dir = work_dir

    # -- Hooks -- #
    logger = dict(type="LoggerHook", interval=args.log_interval, log_metric_by_epoch=False)
    checkpoint = dict(type="CheckpointHook",
                      interval=args.checkpoint_interval,
                      by_epoch=False,
                      max_keep_ckpts=5,
                      save_last=True,
                      save_best="mIoU",
                      rule="greater",
                      published_keys=["meta", "state_dict"])

    cfg.default_hooks.logger = logger
    cfg.default_hooks.checkpoint = checkpoint

    if args.wandb:
        cfg.vis_backends = [dict(
            type='WandbVisBackend',
            init_kwargs = dict(project=args.dataset, name=get_filename(args)))]
    
        cfg.visualizer = dict(
            type='SegLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer')


    # -- Setup other model config info -- #
    cfg.train_cfg = dict(
        type="IterBasedTrainLoop",
        max_iters=args.iterations,
        val_interval=args.val_interval
    )
    
    # -- Set seed to facilitate reproducing the result -- #
    cfg['randomness'] = dict(seed=0)

    # -- Let's have a look at the final config used for training -- #
    print(f'Config:\n{cfg.pretty_text}')

    # -- Workaround to set a custom config name -- #
    cfg.dump(get_filename(args))
    cfg = Config.fromfile(get_filename(args))
    Path(get_filename(args)).unlink()

    # -- Train a model -- #
    runner = Runner.from_cfg(cfg)
    runner.train()

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    MODELS = ["deeplabv3plus", "unet"]

    parser = argparse.ArgumentParser("Training script for training models on the mmsegmentation architecture")
    #parser.add_argument("--project", type=str, required=True, help="The name to use for saving project data")
    parser.add_argument("--batch", type=int, default=2, help="The batch size to use during training")
    parser.add_argument("--wandb", action="store_true", help="Enables Weights and Biases for logging results")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    parser.add_argument("--model", type=str, required=True, choices=MODELS, help="The model to use for training")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"), help="The base directory for datasets")
    parser.add_argument("--normalised", action='store_true', help="Whether to train on stain normalised images")
    parser.add_argument("--iterations", type=int, default=20000, help="The maximum iterations for training")
    parser.add_argument("--val-interval", type=int, default=1000, help="Validation interval during training in iterations")
    parser.add_argument("--log-interval", type=int, default=25, help="Logging interval during training in iterations")
    parser.add_argument("--checkpoint-interval", type=int, default=2000, help="Checkpointing interval in iterations")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    # Ensure old profiling data is cleaned up
    main(args)