from mmengine import Config
from mmseg.apis import init_model, inference_model
from mmseg.structures import SegDataSample
import mmcv
import argparse
from pathlib import Path
import numpy as np
import cv2
import cnsbench

def main(args: argparse.Namespace):
    if not args.output.exists():
        args.output.mkdir(parents=True)

    cfg = Config.fromfile(args.config)

    # Init the model from the config and the checkpoint
    model = init_model(cfg, str(args.checkpoint), 'cuda:0')

    export_all(args.source, model, args.output) #classes, palette)

def export_all(path: Path, model, out_dir: Path):
    assert path.exists(), "path does not exist"
    images = sorted(path.glob("*.png"))
    for image in images:
        img = mmcv.imread(image, channel_order='bgr')
        result: SegDataSample = inference_model(model, img)
        cnsbench.results.export_raw_prediction(result, str(out_dir / image.name))

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=Path, help="The folder containing images for inference")
    parser.add_argument("-o", "--output", type=Path, default="export", help="The folder to place results")
    parser.add_argument("--config", type=Path, help="The configuration file to use for inference (.py)")
    parser.add_argument("--checkpoint", type=Path, help="The model checkpoint to use for inference (.pth)")

    args = parser.parse_args()

    return args


if __name__=="__main__":
    args = get_args()

    main(args)