import numpy as np
from mmseg.structures import SegDataSample
import cv2

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