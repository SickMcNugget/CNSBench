from pathlib import Path
import argparse
import importlib
from cnsbench.datasets import Unzipper

def main(args: argparse.Namespace):
    for dataset in ["CryoNuSeg"]:#args.datasets:
        print(f"-- {dataset} --")
        get_dataset(dataset, args)

def get_dataset(dataset: str, args: argparse.Namespace):
    print(f"Attempting to download {dataset}")
    downloader_cls = get_dataset_class(dataset, "Downloader")
    zip_paths = downloader_cls().download()
    if zip_paths is None:
        return None

    print(f"\nUnzipping {dataset}")
    unzip_paths = Unzipper(zip_paths).unzip()

    print(f"\nOrganising {dataset}")
    mover_cls = get_dataset_class(dataset, "Mover")
    mover_cls(args.dataset_root, unzip_paths).move_all()

    print(f"\nGenerate masks for {dataset}")
    mask_generator_cls = get_dataset_class(dataset, "MaskGenerator")
    mask_generator_cls(args.dataset_root).generate_masks()

    print(f"\nCreating YOLO compatible training data for {dataset}\n")
    yolofier_cls = get_dataset_class(dataset, "Yolofier")
    yolofier_cls(args.dataset_root).yolofy()

def get_dataset_class(dataset: str, class_type: str):
    module = importlib.import_module("cnsbench.datasets")
    dataset_class = getattr(module, f"{dataset}{class_type}")
    return dataset_class

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="append", choices=DATASETS, help="The datasets available for download and preparation")
    parser.add_argument("--dataset-root", default=Path("."), type=Path, help="The path to output the dataset")
    # parser.add_argument("--mask-path", type=Path, help="The path to store generated masks")
    # parser.add_argument("--profile-path", type=Path, default="profile_stats", help="Where to save profiling data")
    # parser.add_argument("--yolofy-path", type=Path, help="The path to store the yolofied version of the dataset")
    # parser.add_argument("--label-value", type=int, default=255, help="The grayscale value for mask values")
    # parser.add_argument("-e", "--ext", action="append", choices=["tif", "jpg", "png"], help="The file extensions to use for mask saving.")
    # parser.add_argument("-p", "--profile", action="store_true", help="Enables the creation of profiling statistics for use with 'show_stats.py'")
    # parser.add_argument("-g", "--get-monuseg", action="store_true", help="Gets MoNuSeg (this is generally only needed for the first run)")
    # parser.add_argument("-y", "--yolofy", action="store_true", help="Creates a version of the dataset that uses yolo-style segmentation labels with .png images")
    # parser.add_argument("-m", "--masks", action="store_true", help="Creates segmentation masks for the dataset")
    # parser.add_argument("--yolofy-det", action="store_true", help="Creates a version of the dataset that uses yolo-style detection labels with .png images")
    # parser.add_argument("--masks-det", action="store_true", help="Creates detection masks for the dataset")

    args = parser.parse_args()

    # Allows .tif to serve as the default mask extension
    if args.datasets is None:
        args.datasets = DATASETS
    # argparse does not have an option for forcing unique values, so this is done 'manually'.
    args.datasets = set(args.datasets)
    
    return args

if __name__=="__main__":
    args = get_args()

    # Ensure old profiling data is cleaned up
    main(args)