import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

gt_labels, gt_instances = cv2.connectedComponents(gt)
pred_labels, pred_instances = cv2.connectedComponents(pred)

gt_masks = np.zeros((gt_labels, *gt_instances.shape))
pred_masks = np.zeros((pred_labels, *pred_instances.shape))


TP = 0
FP = 0
FN = 0
sum_IOU = 0
matched_instances = {}# Create a dictionary to save ground truth indices in keys and predicted matched instances as velues
                    # It will also save IOU of the matched instance in [indx][1]

# Find matched instances and save it in a dictionary
for i in tqdm(np.unique(gt_instances)):
    if i == 0:
        pass
    else:
        temp_image = np.array(gt_instances)
        temp_image = temp_image == i
        matched_image = temp_image * pred_instances
    
        for j in np.unique(matched_image):
            if j == 0:
                pass
            else:
                pred_temp = pred_instances == j
                intersection = sum(sum(temp_image*pred_temp))
                union = sum(sum(temp_image + pred_temp))
                IOU = intersection/union
                sum_IOU += IOU
                if IOU> 0.5:
                    matched_instances [i] = j, IOU 

print(sum_IOU)

# print(gt_masks.shape, pred_masks.shape)
sum_IOU = 0

for i in tqdm(np.unique(gt_instances)):
    if i == 0:
        continue

    gt_mask = np.equal(gt_instances, i)
    overlap_gt_pred_indices = np.logical_and(gt_mask, pred_instances)

    # Multiple nuclei instances may intersect
    pred_overlap = pred_instances[overlap_gt_pred_indices]
    for j in np.unique(pred_overlap):
        if j == 0:
            continue

        pred_mask = np.equal(pred_instances, j)
        intersection = np.sum(np.logical_and(gt_mask, pred_mask))
        union = np.sum(np.logical_or(gt_mask, pred_mask))

        iou = intersection/union
        sum_IOU += iou

print(sum_IOU)
        # print(intersection / union)
    # masked_instance = gt_instances == i
    # matched_image = masked_instance * pred_instances

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(gt_instances * 255)
    # ax[1].imshow(matched_image * 255)
    # plt.show()

# cv2.imshow("GT", gt*255)
# cv2.imshow("pred", pred*255)
# cv2.waitKey()