import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

def general_hausdorff_distance(A: np.ndarray, B: np.ndarray):
    """Retrieved from the scipy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html#scipy.spatial.distance.directed_hausdorff
    """
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])

def general_object_hausdorff_legacy(gt: np.ndarray, pred: np.ndarray):
    if gt.max() == 0 and pred.max() == 0:
        return 0
    elif gt.max() == 0 or pred.max() == 0:
        return np.inf

    gt_to_pred = directed_object_hausdorff_legacy(gt, pred)
    pred_to_gt = directed_object_hausdorff_legacy(pred, gt)

    return (gt_to_pred + pred_to_gt) / 2


def general_object_hausdorff(A: np.ndarray, B: np.ndarray):
    if A.max() == 0 and B.max() == 0:
        return 0
    elif A.max() == 0 or B.max() == 0:
        return np.inf
    
    A_inst = cv2.connectedComponents(A)[1]
    B_inst = cv2.connectedComponents(B)[1]
    
    A_to_B = directed_object_hausdorff(A_inst, B_inst)
    B_to_A = directed_object_hausdorff(B_inst, A_inst)

    return (A_to_B + B_to_A) / 2

def directed_object_hausdorff_legacy(A: np.ndarray, B: np.ndarray):
    A_area = np.sum(np.greater(A, 0), dtype=np.float32)

    A_inst = cv2.connectedComponents(A)[1]
    B_inst = cv2.connectedComponents(B)[1]

    A_unique = np.unique(A_inst)
    A_unique = A_unique[A_unique != 0]
    A_total_unique = len(A_unique)

    B_unique = np.unique(B_inst)
    B_unique = B_unique[B_unique != 0]
    B_total_unique = len(B_unique)

    # Allocate look up tables for object coordinates
    # This algorithm is slow, so reducing memory allocations saves huge time
    obj_dist = np.zeros(B_total_unique)
    B_objs_coords = []
    for B_label in range(B_total_unique):
        B_obj_coords = np.transpose(np.nonzero(np.equal(B_inst, B_unique[B_label])))
        B_objs_coords.append(B_obj_coords)

    A_to_B = 0.0
    for A_label in range(A_total_unique):
        A_obj = np.equal(A_inst, A_unique[A_label])
        A_obj_coords = np.transpose(np.nonzero(A_obj))

        intersection = B_inst[A_obj]
        intersection = intersection[intersection != 0]

        # If they overlap, just grab the MOST overlapping prediction object
        # Otherwise, calculate distance from all predictions to the ground truth and choose the closest
        if len(intersection) > 0:
            B_label = np.argmax(np.bincount(intersection))
            B_obj_coords = np.transpose(np.nonzero(np.equal(B_inst, B_label.item())))
        else:
            for B_label in range(B_total_unique):
                obj_dist[B_label] = general_hausdorff_distance(A_obj_coords, B_objs_coords[B_label])

            B_obj_label = np.argmin(obj_dist)
            B_obj_coords = np.transpose(np.nonzero(np.equal(B_inst, B_unique[B_obj_label])))

        gamma = np.sum(A_obj, dtype=np.float32) / A_area
        A_to_B += gamma * general_hausdorff_distance(A_obj_coords, B_obj_coords)
    return A_to_B

def precision(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    if predicted.sum() == 0:
        return 0.0
    else:
        return (np.logical_and(ground_truth, predicted).sum() / predicted.sum())

def accuracy(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)
    not_ground_truth = np.logical_not(ground_truth)
    not_predicted = np.logical_not(predicted)

    tp = np.logical_and(ground_truth, predicted).sum()
    tn = np.logical_and(not_ground_truth, not_predicted).sum()

    # ground_truth.size represents TN + TP + FN + FP, as it contains all pixels.
    return ((tp + tn) / ground_truth.size)

def recall(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    if ground_truth.sum() == 0:
        return 0.0
    else:
        return (np.logical_and(ground_truth, predicted).sum() / ground_truth.sum())

def f1(precision: float, recall: float):
    if (precision + recall) == 0:
        return 0.0

    return 2 * ((precision * recall) / (precision + recall))

def iou(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    intersection = np.logical_and(ground_truth, predicted).sum()
    union = predicted.sum() + ground_truth.sum() - intersection

    if union == 0:
        return 0.0
    else:
        return intersection / union

def directed_object_hausdorff(A: np.ndarray, B: np.ndarray):
    A_area = np.sum(np.greater(A, 0), dtype=np.float32)

    A_unique = np.unique(A)
    A_unique = A_unique[A_unique != 0]
    A_total_unique = len(A_unique)

    B_unique = np.unique(B)
    B_unique = B_unique[B_unique != 0]
    B_total_unique = len(B_unique)

    # Allocate look up tables for object coordinates
    # This algorithm is slow, so reducing memory allocations saves huge time
    obj_dist = np.zeros(B_total_unique)
    B_objs_coords = []
    for B_label in range(B_total_unique):
        B_obj_coords = np.transpose(np.nonzero(np.equal(B, B_unique[B_label])))
        B_objs_coords.append(B_obj_coords)

    A_to_B = 0.0
    for A_label in range(A_total_unique):
        A_obj = np.equal(A, A_unique[A_label])
        A_obj_coords = np.transpose(np.nonzero(A_obj))

        intersection = B[A_obj]
        intersection = intersection[intersection != 0]

        # If they overlap, just grab the MOST overlapping prediction object
        # Otherwise, calculate distance from all predictions to the ground truth and choose the closest
        if len(intersection) > 0:
            B_label = np.argmax(np.bincount(intersection))
            B_obj_coords = np.transpose(np.nonzero(np.equal(B, B_label.item())))
        else:
            for B_label in range(B_total_unique):
                obj_dist[B_label] = general_hausdorff_distance(A_obj_coords, B_objs_coords[B_label])

            B_obj_label = np.argmin(obj_dist)
            B_obj_coords = np.transpose(np.nonzero(np.equal(B, B_unique[B_obj_label])))

        gamma = np.sum(A_obj, dtype=np.float32) / A_area
        A_to_B += gamma * general_hausdorff_distance(A_obj_coords, B_obj_coords)
    return A_to_B

def precision(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    if predicted.sum() == 0:
        return 0.0
    else:
        return (np.logical_and(ground_truth, predicted).sum() / predicted.sum())

def accuracy(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)
    not_ground_truth = np.logical_not(ground_truth)
    not_predicted = np.logical_not(predicted)

    tp = np.logical_and(ground_truth, predicted).sum()
    tn = np.logical_and(not_ground_truth, not_predicted).sum()

    # ground_truth.size represents TN + TP + FN + FP, as it contains all pixels.
    return ((tp + tn) / ground_truth.size)

def recall(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    if ground_truth.sum() == 0:
        return 0.0
    else:
        return (np.logical_and(ground_truth, predicted).sum() / ground_truth.sum())

def f1(precision: float, recall: float):
    if (precision + recall) == 0:
        return 0.0

    return 2 * ((precision * recall) / (precision + recall))

def iou(X: np.ndarray, y: np.ndarray, class_idx=1):
    ground_truth = np.equal(X, class_idx)
    predicted = np.equal(y, class_idx)

    intersection = np.logical_and(ground_truth, predicted).sum()
    union = predicted.sum() + ground_truth.sum() - intersection

    if union == 0:
        return 0.0
    else:
        return intersection / union