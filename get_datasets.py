from pathlib import Path
import argparse
from cnsbench.datasets import MoNuSegDownloader, Unzipper

def main(args: argparse.Namespace):
    if "MoNuSeg" in args.datasets:
        get_monuseg(args)
    
    if "MoNuSeg" in args.datasets:
        ...
    if "MoNuSeg" in args.datasets:
        ...
    if "MoNuSeg" in args.datasets:
        ...

def get_monuseg(args: argparse.Namespace):
    zip_paths = MoNuSegDownloader().download()
    unzip_paths = Unzipper(zip_paths).unzip()
    # mover = MoNuSegMover()


def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNUSAC", "TNBC", "CryoNuSeg"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="append", choices=DATASETS, help="The datasets available for download and preparation")
    # parser.add_argument("--out-path", type=Path, help="The path to store the output data")
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