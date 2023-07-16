import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff


def main():
    gt = cv2.imread("/home/joren/localhonours/code/CNSBench/MoNuSeg/masks/test/TCGA-2Z-A9J9-01A-01-TS1.png", cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread("/home/joren/localhonours/code/NucleiSegmentation_mmseg/export/TCGA-2Z-A9J9-01A-01-TS1.png", cv2.IMREAD_GRAYSCALE)
    gt_instances = cv2.connectedComponents(gt)[1]
    pred_instances = cv2.connectedComponents(pred)[1]

    print(object_level_hausdorff3(gt_instances, pred_instances))

def general_hausdorff_distance(A: np.ndarray, B: np.ndarray):
    """Retrieved from the scipy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html#scipy.spatial.distance.directed_hausdorff
    """
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])

def object_level_hausdorff3(S, G):
    tempS = S > 0
    totalAreaS = np.sum(tempS, dtype=np.float32)

    tempG = G > 0
    totalAreaG = np.sum(tempG, dtype=np.float32)

    # 获取S中object列表及非0元素个数
    listLabelS = np.unique(S)
    listLabelS = listLabelS[listLabelS != 0]
    numS = len(listLabelS)
    # 获取G中object列表及非0元素个数
    listLabelG = np.unique(G)
    listLabelG = listLabelG[listLabelG != 0]
    numG = len(listLabelG)

    # 记录omega_i*H(G_i,S_i)
    temp1 = 0.0

    for iLabelS in tqdm(range(numS)):
        # Si为S中值为iLabelS的区域, boolean矩阵
        Si = S == listLabelS[iLabelS]
        # 找到G中对应区域并去除背景
        intersectlist = G[Si]
        intersectlist = intersectlist[intersectlist != 0]

        if len(intersectlist) != 0:
            indexGi = np.argmax(np.bincount(intersectlist))
            Gi = G == indexGi.item()
        else:
            tempDist = np.zeros(numG)

            for iLabelG in range(numG):
                Gi = G == listLabelG[iLabelG]
                tempDist[iLabelG] = general_hausdorff_distance(Gi, Si)

            minIdx = np.argmin(tempDist)
            Gi = G == listLabelG[minIdx]

        omegai = np.sum(Si, dtype=np.float32) / totalAreaS
        temp1 = temp1 + omegai * general_hausdorff_distance(Gi, Si)

    # 记录tilde_omega_i*H(tilde_G_i,tilde_S_i)
    temp2 = 0.0

    for iLabelG in tqdm(range(numG)):
        # Si为S中值为iLabelS的区域, boolean矩阵
        tildeGi = G == listLabelG[iLabelG]
        # 找到G中对应区域并去除背景
        intersectlist = S[tildeGi]
        intersectlist = intersectlist[intersectlist != 0]

        if len(intersectlist) != 0:
            indextildeSi = np.argmax(np.bincount(intersectlist))
            tildeSi = S == indextildeSi.item()
        else:
            tempDist = np.zeros(numS)

            for iLabelS in range(numS):
                tildeSi = S == listLabelS[iLabelS]
                tempDist[iLabelS] = general_hausdorff_distance(tildeGi, tildeSi)

            minIdx = np.argmin(tempDist)
            tildeSi = S == listLabelS[minIdx]

        tildeOmegai = np.sum(tildeGi, dtype=np.float32) / totalAreaG
        temp2 = temp2 + tildeOmegai * general_hausdorff_distance(tildeGi, tildeSi)

    objHaus = (temp1 + temp2) / 2
    return objHaus


def object_level_hausdorff2(gt, pred):
    first = haus_a_to_b(gt, pred)
    second = haus_a_to_b(pred, gt)
    object_level_haus = (first + second) / 2
    return object_level_haus

def haus_a_to_b(gt, pred):
    # These will be used for weighting objects based on area
    gt_area = np.sum(gt > 0, dtype=np.float32)
    pred_area = np.sum(pred > 0, dtype=np.float32)

    # I assume instance are not labelled, i.e semantic labelling
    gt_instances = cv2.connectedComponents(gt)[1]
    pred_instances = cv2.connectedComponents(pred)[1]

    # All labels will be used in the case where objects cannot be directly
    # matched with another object.
    gt_unique_labels = np.unique(gt_instances)
    gt_unique_labels = gt_unique_labels[gt_unique_labels != 0]
    gt_num_labels = len(gt_unique_labels)

    pred_unique_labels = np.unique(pred_instances)
    pred_unique_labels = pred_unique_labels[pred_unique_labels != 0]
    pred_num_labels = len(pred_unique_labels)

    haus_gt_to_pred = 0.0
    # Start by comparing all instances in the ground truth with instances in predictions
    for gt_label_index in tqdm(range(gt_num_labels)):
        # mask the ground truth so only the instance of interest is visible
        gt_object = np.equal(gt_instances, gt_unique_labels[gt_label_index])

        intersection = pred_instances[gt_object]
        intersection = intersection[intersection != 0]

        if len(intersection) > 0:
            # Count the number of pixels corresponding to all objects that overlap
            pred_label_index = np.argmax(np.bincount(intersection))
            # whichever object had the most pixels, select it as the comparison of choice
            # the below line may need to be swapped for:
            # pred_object = np.equal(pred_instances, pred_unique_labels[pred_label_index])
            pred_object = np.equal(pred_instances, pred_label_index.item())
        else:
            distance_to_closest_object = np.zeros(pred_num_labels)

            for pred_label_index in range(pred_num_labels):
                pred_object = np.equal(pred_instances, pred_unique_labels[pred_label_index])
                # Order of parameters to general hausdorff does not matter, both directions are calculated
                distance_to_closest_object[pred_label_index] = general_hausdorff_distance(gt_object, pred_object)

            pred_closest_object_label = np.argmin(distance_to_closest_object)
            pred_object = np.equal(pred_instances, pred_unique_labels[pred_closest_object_label])
            
        # gamma is the previously mentioned weighting 
        gamma = np.sum(gt_object, dtype=np.float32) / gt_area
        haus_gt_to_pred = haus_gt_to_pred + gamma * general_hausdorff_distance(gt_object, pred_object)
    return haus_gt_to_pred

def object_level_hausdorff(gt: np.ndarray, pred: np.ndarray):
    # These will be used for weighting objects based on area
    gt_area = np.sum(gt > 0, dtype=np.float32)
    pred_area = np.sum(pred > 0, dtype=np.float32)

    # I assume instance are not labelled, i.e semantic labelling
    gt_instances = cv2.connectedComponents(gt)[1]
    pred_instances = cv2.connectedComponents(pred)[1]

    # All labels will be used in the case where objects cannot be directly
    # matched with another object.
    gt_unique_labels = np.unique(gt_instances)
    gt_unique_labels = gt_unique_labels[gt_unique_labels != 0]
    gt_num_labels = len(gt_unique_labels)

    pred_unique_labels = np.unique(pred_instances)
    pred_unique_labels = pred_unique_labels[pred_unique_labels != 0]
    pred_num_labels = len(pred_unique_labels)

    haus_gt_to_pred = 0.0
    # Start by comparing all instances in the ground truth with instances in predictions
    for gt_label_index in tqdm(range(gt_num_labels)):
        # mask the ground truth so only the instance of interest is visible
        gt_object = np.equal(gt_instances, gt_unique_labels[gt_label_index])

        intersection = pred_instances[gt_object]
        intersection = intersection[intersection != 0]

        if len(intersection) > 0:
            # Count the number of pixels corresponding to all objects that overlap
            pred_label_index = np.argmax(np.bincount(intersection))
            # whichever object had the most pixels, select it as the comparison of choice
            # the below line may need to be swapped for:
            # pred_object = np.equal(pred_instances, pred_unique_labels[pred_label_index])
            pred_object = np.equal(pred_instances, pred_label_index.item())
        else:
            distance_to_closest_object = np.zeros(pred_num_labels)

            for pred_label_index in range(pred_num_labels):
                pred_object = np.equal(pred_instances, pred_unique_labels[pred_label_index])
                # Order of parameters to general hausdorff does not matter, both directions are calculated
                distance_to_closest_object[pred_label_index] = general_hausdorff_distance(gt_object, pred_object)

            pred_closest_object_label = np.argmin(distance_to_closest_object)
            pred_object = np.equal(pred_instances, pred_unique_labels[pred_closest_object_label])
            
        # gamma is the previously mentioned weighting 
        gamma = np.sum(gt_object, dtype=np.float32) / gt_area
        haus_gt_to_pred = haus_gt_to_pred + gamma * general_hausdorff_distance(gt_object, pred_object)
    
    haus_pred_to_gt = 0.0
    for pred_label_index in tqdm(range(pred_num_labels)):
        # mask the ground truth so only the instance of interest is visible
        pred_object = np.equal(pred_instances, pred_unique_labels[pred_label_index])

        intersection = gt_instances[pred_object]
        intersection = intersection[intersection != 0]

        if len(intersection) > 0:
            # Count the number of pixels corresponding to all objects that overlap
            gt_label_index = np.argmax(np.bincount(intersection))
            # whichever object had the most pixels, select it as the comparison of choice
            # the below line may need to be swapped for:
            # gt_object = np.equal(gt_instances, gt_unique_labels[gt_label_index])
            gt_object = np.equal(gt_instances, gt_label_index.item())
        else:
            distance_to_closest_object = np.zeros(gt_num_labels)

            for gt_label_index in range(gt_num_labels):
                gt_object = np.equal(gt_instances, gt_unique_labels[gt_label_index])
                # Order of parameters to general hausdorff does not matter, both directions are calculated
                distance_to_closest_object[gt_label_index] = general_hausdorff_distance(gt_object, pred_object)

            gt_closest_object_label = np.argmin(distance_to_closest_object)
            gt_object = np.equal(gt_instances, gt_unique_labels[gt_closest_object_label])
            
        # gamma is the previously mentioned weighting 
        sigma = np.sum(pred_object, dtype=np.float32) / pred_area
        haus_pred_to_gt = haus_pred_to_gt + sigma * general_hausdorff_distance(gt_object, pred_object)



    object_level_haus = (haus_gt_to_pred + haus_pred_to_gt) / 2
    return object_level_haus

if __name__=="__main__":
    main()