import argparse
from pathlib import Path
import time
import importlib

def main(args: argparse.Namespace):
    

def get_mmseg_root():
    mmseg = importlib.util.find_spec("mmseg")
    root = Path(mmseg.submodule_search_locations[0])
    while "mmsegmentation" not in root.stem and not root.stem == "":
        root = root.parent

    return root

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, help="The configuration file to use for training")
    parser.add_argument("--project", type=str, help="The name to use for saving project data")
    parser.add_argument("--batch-size", type=int, default=2, help="The batch size to use during training")
    parser.add_argument("--wandb", action="store_true", help="Enables Weights and Biases for logging results")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()

    # Ensure old profiling data is cleaned up
    main(args)