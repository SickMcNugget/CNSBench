import cv2
import numpy as np


def calc_hausdorff(X: np.ndarray, y: np.ndarray, class_idx=1):
    X_contours = cv2.findContours(X, 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)[0]
    y_contours = cv2.findContours(y, 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)[0]

    X_centroids = contours_to_centroids(X_contours)
    y_centroids = contours_to_centroids(y_contours)


def contours_to_centroids(contours: tuple):
    centroids = np.empty((len(contours),2), dtype=np.uint16)
    for i in range(len(contours)):
        contour = contours[i]
        M = cv2.moments(contour)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        centroids[i,0] = centroid_x
        centroids[i,1] = centroid_y
    return centroids


def calc_precision(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    if predicted.sum() == 0:
        return 0.0
    else:
        return (np.logical_and(ground_truth, predicted).sum() / predicted.sum())

def calc_accuracy(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)
    not_ground_truth = np.logical_not(ground_truth)
    not_predicted = np.logical_not(predicted)

    tp = np.logical_and(ground_truth, predicted).sum()
    tn = np.logical_and(not_ground_truth, not_predicted).sum()

    # ground_truth.size represents TN + TP + FN + FP, as it contains all pixels.
    return ((tp + tn) / ground_truth.size)

def calc_recall(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    if ground_truth.sum() == 0:
        return 0.0
    else:
        return (np.logical_and(ground_truth, predicted).sum() / ground_truth.sum())

def calc_f1(precision: float, recall: float):
    return 2 * ((precision * recall) / (precision + recall))

def calc_iou(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    intersection = np.logical_and(ground_truth, predicted).sum()
    union = predicted.sum() + ground_truth.sum() - intersection

    if union == 0:
        return 0.0
    else:
        return intersection / union
