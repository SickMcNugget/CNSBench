from mmengine import Config
from mmseg.apis import init_model, inference_model
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
import mmcv
import argparse
from pathlib import Path
import numpy as np
import cv2

def main(args: argparse.Namespace):
    cfg = Config.fromfile(args.config)

    # Init the model from the config and the checkpoint
    model = init_model(cfg, str(args.checkpoint), 'cuda:0')

    classes = model.dataset_meta['classes']
    palette = model.dataset_meta['palette']

    img = mmcv.imread(args.source, channel_order='bgr')
    if args.ground_truth:
        ground_truth = mmcv.imread(args.ground_truth, channel_order='bgr')

    result: SegDataSample = inference_model(model, img)

    if args.binary:
        # Under the hood, this function converts the result to a numpy array and displays it
        show_raw_prediction(result)
    else:
        prediction_overlay = get_overlayed_prediction(img, result, classes, palette, args.outline)
        
        if args.ground_truth:
            ground_truth_overlay = overlay_mask(img, ground_truth, classes, palette, args.outline)
        
        cv2.imshow("Original", img)
        cv2.imshow("Prediction", prediction_overlay)
        if args.ground_truth:
            cv2.imshow("Ground Truth", ground_truth_overlay)
        cv2.waitKey(0)

def get_overlayed_prediction(src: np.ndarray,
                             result: SegDataSample,
                             classes: "list[str]", 
                             palette: "list[list[int, int, int]]",
                             outline: bool,
                             alpha: float = 0.5) -> np.ndarray:     
    mask = numpy_from_result(result)
    dest = overlay_mask(src, mask, classes, palette, outline=outline, alpha=alpha)
    return dest

def overlay_mask(src: np.ndarray,
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

def show_raw_prediction(result: SegDataSample):
    """Uses OpenCV to show segmentation results

    Parameters
    ----------
    result : SegDataSample
        The predictions made by an mmsegmentation model

    """    
    prediction = numpy_from_result(result)
    cv2.imshow("Prediction", prediction)
    cv2.waitKey(0)

def numpy_from_result(result: SegDataSample, squeeze: bool = True, as_uint: bool = True) -> np.ndarray:
    """Converts an mmsegmentation inference result into a numpy array (for exporting and visualisation)

    Parameters
    ----------
    result : SegDataSample
        The segmentation results to extract the numpy array from
    squeeze : bool, optional
        Squeezes down useless dimensions (mainly for binary segmentation), by default True
    as_uint : bool, optional
        Converts the array to uint8 format, instead of (usually) int64, by default True

    Returns
    -------
    np.ndarray
        The extracted numpy array
    """    
    array: np.ndarray = result.pred_sem_seg.cpu().numpy().data
    if squeeze:
        array = array.squeeze()
    if as_uint:
        array = array.astype(np.uint8)
    return array

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=Path, required=True, help="The image to perform inference on")
    parser.add_argument("--ground-truth", type=Path, help="The ground truth mask for the selected image")
    parser.add_argument("--config", type=Path, required=True, help="The configuration file to use for inference (.py)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="The model checkpoint to use for inference (.pth)")
    parser.add_argument("-b", "--binary", action='store_true', help="Whether black and white binary masks should be produced")
    parser.add_argument("--outline", action='store_true', default=True, help="Whether overlays should be drawn as an outline instead of filled in")

    args = parser.parse_args()

    return args


if __name__=="__main__":
    args = get_args()

    main(args)