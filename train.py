import argparse
from pathlib import Path
import datetime
import importlib
import mmcv
import mmseg
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import Config
from mmengine.runner import Runner

def get_config(dataset: str, normalised: bool, model: str):
    config = ""
    match dataset.lower():
        case 'monuseg':
            match model.lower():
                case 'deeplabv3plus':
                    if normalised:
                        config = "config/deeplabv3plus_r50-d8_1xb1-20k_monuseg_norm-512x512.py"
                    else:
                        config = "config/deeplabv3plus_r50-d8_1xb1-20k_monuseg-512x512.py"
                case 'unet':
                    if normalised:
                        config = "3"
                    else:
                        config = "4"
                case default:
                    return "Model chosen does not exist"
        case 'monusac':
            match model.lower():
                case 'deeplabv3plus':
                    if normalised:
                        config = "config/deeplabv3plus_r50-d8_1xb1-20k_monusac_norm-512x512.py"
                    else:
                        config = "config/deeplabv3plus_r50-d8_1xb1-20k_monusac-512x512.py"
                case 'unet':
                    if normalised:
                        config = "7"
                    else:
                        config = "8"
                case default:
                    return "Model chosen does not exist"
        case 'cryonuseg':
            match model.lower():
                case 'deeplabv3plus':
                    if normalised:
                        config = "config/deeplabv3plus_r50-d8_1xb1-20k_cryonuseg_norm-512x512.py"
                    else:
                        config = "config/deeplabv3plus_r50-d8_1xb1-20k_cryonuseg-512x512.py"
                case 'unet':
                    if normalised:
                        config = "11"
                    else:
                        config = "12"
                case default:
                    return "Model chosen does not exist"
        case 'tnbc':
            match model.lower():
                case 'deeplabv3plus':
                    if normalised:
                        config = "config/deeplabv3plus_r50-d8_1xb1-20k_tnbc_norm-512x512.py"
                    else:
                        config = "config/deeplabv3plus_r50-d8_1xb1-20k_tnbc-512x512.py"
                case 'unet':
                    if normalised:
                        config = "15"
                    else:
                        config = "16"
                case default:
                    return "Model chosen does not exist"
        case default:
            return "Dataset chosen does not exist"
    return config

def register_datasets():
    @DATASETS.register_module()
    class MonusegDataset(BaseSegDataset):
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
    class MonusacDataset(BaseSegDataset):
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
    class CryonusegDataset(BaseSegDataset):
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
    class TnbcDataset(BaseSegDataset):
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

def main(args: argparse.Namespace):
    
    # make sure the datasets are registered
    register_datasets() 
    # -- grab the config location based on arguments -- #
    config_location = get_config(args.dataset, args.normalised, args.model)
    cfg = Config.fromfile(config_location)
    print('[*] Config loaded: ' + config_location)
    
    # -- set batch size for model -- #
    cfg.train_dataloader.batch_size = args.batch

    # -- Create the work directory -- #
    now = datetime.datetime.now()
    work_dir = f"./work_dirs/{args.dataset}/{args.model}/{now.year}/{now.month}/{now.day}/{now.hour}/{now.minute}/"
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
            init_kwargs = dict(project=args.dataset, name=config_location))]
    
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

    # -- Train a model -- #
    runner = Runner.from_cfg(cfg)
    runner.train()

def get_mmseg_root():
    mmseg = importlib.util.find_spec("mmseg")
    root = Path(mmseg.submodule_search_locations[0])
    while "mmsegmentation" not in root.stem and not root.stem == "":
        root = root.parent

    return root

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    MODELS = ["deeplabv3plus", "unet"]

    parser = argparse.ArgumentParser("Training script for training models on the mmsegmentation architecture")
    #parser.add_argument("--project", type=str, required=True, help="The name to use for saving project data")
    parser.add_argument("--batch", type=int, default=2, help="The batch size to use during training")
    parser.add_argument("--wandb", action="store_true", help="Enables Weights and Biases for logging results")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    parser.add_argument("--model", type=str, required=True, choices=MODELS, help="The model to use for training")
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

