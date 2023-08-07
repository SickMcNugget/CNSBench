from mmengine import Config
from mmseg.apis import init_model, inference_model
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
import mmcv
import argparse
from pathlib import Path
import numpy as np
import cv2
import datetime
import os

from cnsbench.results import numpy_from_result

def get_config(args: argparse.Namespace):
    checkpoint_dir = args.checkpoint.parent
    config = list(checkpoint_dir.glob("*.py"))
    if len(config) == 1:
        return config[0]
    else:
        print("Error getting config")
        return None

def get_images(args: argparse.Namespace):
    images = args.dataset_root / args.dataset
    if args.normalised:
        images = images / "yolo_sn"
    else:
        images /= "yolo"
    images /= "test"
    return sorted(images.glob("*.png"))

def get_mask(image_path: Path):
    image_name = image_path.name
    return image_path.parents[2] / "masks" / "test" / image_name

def main(args: argparse.Namespace):
    if args.config is None:
        args.config = get_config(args)

    images = get_images(args)
    if len(images) == 0:
        print("No images found!")
        return

    cfg = Config.fromfile(args.config)
    model = init_model(cfg, str(args.checkpoint), 'cuda:0')
    classes = model.dataset_meta['classes']
    palette = model.dataset_meta['palette']
    now = datetime.datetime.now()
    if args.save is not None:
        save_dir = f"{args.save}/{now.year}-{now.month:02d}-{now.day:02d}_{now.hour:02d}_{now.minute:02d}"
    try:
        os.mkdir(save_dir)
    except OSError as error:
        print(error)
    i = 0
    for image in images:
        if args.number is not None:
            if i < args.number:
                process_images(image, model, classes, palette, args, now)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == ord('q'):
                    break
                i = i + 1
        else:
            process_images(image, model, classes, palette, args, now)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break

def process_images(image, model, classes, palette, args, now):
    img = mmcv.imread(image, channel_order='bgr')
    ground_truth = mmcv.imread(get_mask(image), flag="grayscale")
    result: SegDataSample = inference_model(model, img)

    if args.binary:
        # Under the hood, this function converts the result to a numpy array and displays it
        original = cv2.equalizeHist(ground_truth)
        prediction = cv2.equalizeHist(numpy_from_result(result))
    else:
        original = mask_overlay(img, ground_truth, classes, palette, args.outline)
        prediction = prediction_overlay(img, result, classes, palette, args.outline)
    
    if args.save is not None:
        save_name = f"{args.save}/{now.year}-{now.month:02d}-{now.day:02d}_{now.hour:02d}_{now.minute:02d}/{image.name}"
        print("[*] Saving to: "+save_name)
        cv2.imwrite(save_name, prediction)
    else:
        cv2.imshow(f"{image.stem} - Ground Truth", original)
        cv2.imshow(f"{image.stem} - Prediction", prediction)
    

def prediction_overlay(src: np.ndarray,
                             result: SegDataSample,
                             classes: "list[str]", 
                             palette: "list[list[int, int, int]]",
                             outline: bool,
                             alpha: float = 0.5) -> np.ndarray:     
    mask = numpy_from_result(result)
    dest = mask_overlay(src, mask, classes, palette, outline=outline, alpha=alpha)
    return dest

def mask_overlay(src: np.ndarray,
                 mask: np.ndarray,
                 classes: "list[str]", 
                 palette: "list[list[int, int, int]]", 
                 outline: bool,
                 alpha: float = 0.5) -> np.ndarray:
    dest = src.copy()
    labels = np.unique(mask)

    for label in labels:
        # skipping background (ugly in visualisations)
        if classes[label].lower() in ["background", "bg"]:
            continue

        binary_mask = (mask == label)
        colour_mask = np.zeros_like(src)
        
        if outline:
            contours = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            colour_mask = cv2.drawContours(colour_mask, contours, -1, palette[label], thickness=2)
            binary_mask = np.nonzero(colour_mask)
            alpha = 1
        else:
            colour_mask[...] = palette[label]
        
        dest[binary_mask] = cv2.addWeighted(src, 1 - alpha, colour_mask, alpha, 0)[binary_mask]

    return dest

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "CryoNuSeg", "TNBC"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, choices=DATASETS, help="The folder containing images for inference")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"), help="The path to where datasets are stored")
    parser.add_argument("--config", type=Path, default=None, help="The configuration file to use for inference (.py)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="The model checkpoint to use for inference (.pth)")
    parser.add_argument("-b", "--binary", action='store_true', help="Whether black and white binary masks should be produced")
    parser.add_argument("--outline", action='store_true', default=True, help="Whether overlays should be drawn as an outline instead of filled in")
    parser.add_argument("--normalised", action='store_true', help="Whether to train on stain normalised images")
    parser.add_argument("--number", type=int, help="Limit how many images to process visuals for")
    parser.add_argument("--save", type=Path, help="directory to save predictions to")
    args = parser.parse_args()

    return args


if __name__=="__main__":
    args = get_args()

    main(args)