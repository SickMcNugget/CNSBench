import argparse
from ultralytics import YOLO
from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks

def get_config(dataset: str, normalised: bool):
    config = f"configs/yamls/{dataset.lower()}"
    if normalised:
        config = f"{config}_norm"
    config += ".yaml"
    return config

def get_name(args: argparse.Namespace):
    name = f"yolov8l_1xb{args.batch}-{args.epochs}e_{args.dataset.lower()}"
    if args.normalised:
        name += "_norm"
    name += "-640x640"
    return name

def main(args: argparse.Namespace):
    # -- Grab the dataset config for model -- #
    data = get_config(args.dataset, args.normalised)
    print('[*] Config loaded: ' + data)

    # -- Load model -- #
    model = YOLO("yolov8l-seg.pt", task="segment")
    name = get_name(args)
    add_wandb_callbacks(model,
                        run_name=name,
                        project=args.dataset,
                        dir=f"./{args.out_dir}")
    model.train(data=data, 
                lr0=args.lr,
                epochs=args.epochs, 
                batch=args.batch,
                cache=args.cache,
                project=f"{args.out_dir}/{args.dataset}",
                name=f"train_{get_name(args)}")

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    parser = argparse.ArgumentParser("Training script for the Ultralytics YOLOv8 model")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    parser.add_argument("--batch", type=int, default=2, help="number of images per batch")
    parser.add_argument("--normalised", action='store_true', help="Whether to train on stain normalised images")
    parser.add_argument("--lr", type=int, default=0.0003, help="initial learning rate (i.e. SGD=1E-2, Adam=1E-3)")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train for")
    parser.add_argument("--cache", action='store_false', help="Disable dataset caching (on by default)")
    parser.add_argument("--out-dir", type=str, default="yolov8_work_dirs", help="The base directory for saving yolov8 training results")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    # Ensure old profiling data is cleaned up
    main(args)