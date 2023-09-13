from mmengine import Config
from mmseg.apis import init_model, inference_model
from mmseg.structures import SegDataSample
import mmcv
import argparse
from pathlib import Path
from cnsbench.results import export_raw_prediction


def get_source(args: argparse.Namespace):
    source = args.dataset_root / args.dataset
    if args.normalised:
        source = source / "yolo_sn"
    else:
        source = source / "yolo"
    source = source / "test"
    return source


def get_save_path(args: argparse.Namespace):
    save_path = args.out_dir / "export"
    if args.normalised:
        save_path = save_path / "stainnorm"
    else:
        save_path = save_path / "nostainnorm"

    save_path = save_path / args.dataset
    if (
        "deeplabv3plus" in args.config.name
        or "deeplabv3plus" in args.checkpoint.parents
    ):
        save_path = save_path / "deeplabv3plus"
    elif "unet" in args.config.name or "unet" in args.checkpoint.parents:
        save_path = save_path / "unet"
    elif "stdc" in args.config.name or "stdc" in args.checkpoint.parents:
        save_path = save_path / "stdc"

    return save_path


def get_config(args: argparse.Namespace):
    checkpoint_dir = args.checkpoint.parent
    config = list(checkpoint_dir.glob("*.py"))
    if len(config) == 1:
        return config[0]
    else:
        print("Error getting config")
        return None


def main(args: argparse.Namespace):
    if args.config is None:
        args.config = get_config(args)

    save_path = get_save_path(args)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    cfg = Config.fromfile(args.config)

    # Init the model from the config and the checkpoint
    model = init_model(cfg, str(args.checkpoint), "cuda:0")

    export_all(get_source(args), model, save_path)  # classes, palette)


def export_all(path: Path, model, out_dir: Path):
    assert path.exists(), "path does not exist"
    images = sorted(path.glob("*.png"))
    for image in images:
        # img = mmcv.imread(image, channel_order='rgb')
        result: SegDataSample = inference_model(model, image)
        export_raw_prediction(result, str(out_dir / image.name))


def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "CryoNuSeg", "TNBC"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=DATASETS,
        help="The folder containing images for inference",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets"),
        help="The path to where datasets are stored",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("work_dirs"),
        help="The folder to place results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="The configuration file to use for inference (.py)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="The model checkpoint to use for inference (.pth)",
    )
    parser.add_argument(
        "--normalised",
        action="store_true",
        help="Whether to train on stain normalised images",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    main(args)

