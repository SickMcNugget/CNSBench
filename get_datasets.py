from pathlib import Path
import argparse
from cnsbench.datasets import DatasetManager
from cnsbench.datasets import make_directories, get_monuseg


def main(args: argparse.Namespace):
    # manager = DatasetManager(args.dataset_root)

    for dataset in args.datasets:
        make_directories(args.dataset_root, dataset)
        if dataset == "MoNuSeg":
            get_monuseg(args.dataset_root)


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

    # Ensure old profiling data is cleaned up
    main(args)
