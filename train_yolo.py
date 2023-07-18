import argparse
from pathlib import Path
import datetime
import importlib
import os

def get_config(dataset: str, normalised: bool):
    config = ""
    match dataset.lower():
        case 'monuseg':
            if normalised:
                config = "yaml/monuseg_norm.yaml"
            else:
                config = "yaml/monuseg.yaml"
        case 'monusac':
            if normalised:
                config = "yaml/monusac_norm.yaml"
            else:
                config = "yaml/monusac.yaml"
        case 'cryonuseg':
            if normalised:
                config = "yaml/cryonuseg_norm.yaml"
            else:
                config = "yaml/cryonuseg.yaml"
        case 'tnbc':
            if normalised:
                config = "yaml/tnbc_norm.yaml"
            else:
                config = "yaml/tnbc.yaml"
        case default:
            return "Dataset chosen does not exist"
    return config


def main(args: argparse.Namespace):
    # -- Grab the dataset config for model -- #
    data = get_config(args.dataset, args.normalised)
    print('[*] Config loaded: ' + data)

    # -- Model to load -- #
    model = "yolov8l-seg.pt"

    # -- full loader -- #
    cuda = "CUDA_VISIBLE_DEVICES='0'"
    name = "yolov8_1xb"+str(args.batch)+"-"+str(args.epochs)+"e_"+args.dataset.lower()+"-640x640"
    loader =cuda+" yolo segment train"+" data="+data+" model="+model+" epochs="+str(args.epochs)+" lr0="+str(args.lr)+" max_det="+str(args.det)+" batch="+str(args.batch)+" cache="+str(args.cache)+" project="+args.dataset+" name="+name

    # -- Run the script -- #
    #print(loader) 
    os.system(loader)


def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    parser = argparse.ArgumentParser("Training script for the Ultralytics YOLOv8 model")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    parser.add_argument("--batch", type=int, default=2, help="number of images per batch")
    parser.add_argument("--normalised", action='store_true', help="Whether to train on stain normalised images")
    parser.add_argument("--lr", type=int, default=0.0003, help="initial learning rate (i.e. SGD=1E-2, Adam=1E-3)")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train for")
    parser.add_argument("--cache", action='store_true', help="True/ram, disk or False. Use cache for data loading")
    parser.add_argument("--det", type=int, default=1000, help="maximum number of detections per image")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    # Ensure old profiling data is cleaned up
    main(args)