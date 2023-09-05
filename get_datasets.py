from pathlib import Path
import argparse
import shutil
from cnsbench.datasets import (
    make_directories,
    get_dataset_type,
    files_exist,
    missing_files,
    Unzipper,
    StainExtractor,
    StainNormaliser,
)


def main(args: argparse.Namespace):
    for dataset in args.datasets:
        make_directories(args.dataset_root, dataset)
        dataset = get_dataset_type(dataset_name=dataset)
        dataset = dataset(args.dataset_root)

        zip_folder = args.dataset_root / "zips"
        unzip_folder = args.dataset_root / "unzips"

        if files_exist(zip_folder, dataset.zip_names):
            print(f"No need to download {dataset.name}...")
        else:
            print(f"Downloading {dataset.name}...")
            for zip_name, zip_source in zip(dataset.zip_names, dataset.zip_sources):
                if dataset.download(zip_source):
                    shutil.move(zip_name, zip_folder)
                else:
                    print(f"Error downloading {zip_name} using source {zip_source}")
                    return

        zip_paths = [zip_folder / zip_name for zip_name in dataset.zip_names]

        unzipper = Unzipper()
        if files_exist(unzip_folder, dataset.unzip_names):
            print(f"No need to unzip {dataset.name}...")
        else:
            print(f"Unzipping {dataset.name}...")
            for zip_path in zip_paths:
                unzipper.unzip(zip_path, (unzip_folder / zip_path.stem))

        unzip_paths = [(unzip_folder / zip_path.stem) for zip_path in zip_paths]

        if not missing_files(dataset.original_paths):
            print(f"No need to organise {dataset.name}...")
        else:
            print(f"Organising {dataset.name}...")
            dataset.organise(unzip_paths)

        if not missing_files(dataset.mask_paths):
            print(f"No need to generate masks for {dataset.name}...")
        else:
            print(f"Generating masks for {dataset.name}...")
            dataset.generate_masks()

        if not missing_files(dataset.yolo_paths):
            print(f"No need to yolofy {dataset.name}...")
        else:
            print(f"Yolofying {dataset.name}...")
            dataset.yolofy()

        if not missing_files(dataset.stain_paths):
            print(f"No need to extract stains for {dataset.name}...")
        else:
            print(f"Extracting stains for {dataset.name}...")
            stain_extractor = StainExtractor(dataset.fit_image)
            stain_extractor.extract(dataset.yolo_paths, dataset.stain_paths)

        if not missing_files(dataset.yolosn_paths):
            print(f"No need to stain normalise {dataset.name}...")
        else:
            # fit_image = yolo_paths[0] / "TCGA-A7-A13E-01Z-00-DX1.png"
            print(f"Stain normalising {dataset.name}...")
            stain_normaliser = StainNormaliser("reinhard", dataset.fit_image)
            stain_normaliser.normalise(dataset.yolo_paths, dataset.yolosn_paths)


def get_args() -> argparse.Namespace:
    DATASETS = {"MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        action="append",
        choices=DATASETS,
        help="The datasets available for download and preparation",
    )
    parser.add_argument(
        "--dataset-root",
        default=Path("datasets"),
        type=Path,
        help="The path to output the dataset",
    )

    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = DATASETS

    args.datasets = set(args.datasets)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
