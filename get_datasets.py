from pathlib import Path
import argparse
import importlib
from cnsbench.datasets import Unzipper, DownloaderError

def main(args: argparse.Namespace):
    for dataset in args.datasets:
        print(f"-- {dataset} --")
        get_dataset(dataset, args)

def get_dataset(dataset: str, args: argparse.Namespace):
    print(f"Attempting to download {dataset}")
    downloader_cls = get_dataset_class(dataset, "Downloader")

    try:
        zip_paths = downloader_cls().download()
    
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
    except DownloaderError as e:
        print(e)

def get_dataset_class(dataset: str, class_type: str):
    module = importlib.import_module("cnsbench.datasets")
    dataset_class = getattr(module, f"{dataset}{class_type}")
    return dataset_class

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="append", choices=DATASETS, help="The datasets available for download and preparation")
    parser.add_argument("--dataset-root", default=Path("datasets"), type=Path, help="The path to output the dataset")

    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = DATASETS
    # argparse does not have an option for forcing unique values, so this is done 'manually'.
    args.datasets = set(args.datasets)
    
    return args

if __name__=="__main__":
    args = get_args()

    # Ensure old profiling data is cleaned up
    main(args)