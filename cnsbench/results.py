import numpy as np
from mmseg.structures import SegDataSample
import cv2
import torch
import torchvision.transforms as T

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

def export_raw_prediction(result: SegDataSample, image: str):
    """Uses OpenCV to export segmentation results

    Parameters
    ----------
    result : SegDataSample
        The predictions made by an mmsegmentation model
    image : str
        The name of the image for saving

    """    
    prediction = numpy_from_result(result)
    cv2.imwrite(image, prediction)

def yolo_to_numpy(result):
    if result.masks is None:
        return np.zeros(result.orig_img.shape[:2])
    
    array: torch.Tensor = result.masks.data
    array = torch.any(array, dim=0).to(torch.uint8)
    # add two empty dimensions for resize
    array = array.reshape(1, 1, *array.shape)

    array = T.Resize(size=result.orig_img.shape[:2], 
                     interpolation=T.InterpolationMode.NEAREST, 
                     antialias=False)(array).squeeze()
    return array.cpu().numpy()

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