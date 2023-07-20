import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cnsbench import metrics

def main():
    gt = cv2.imread("/home/joren/localhonours/code/CNSBench/datasets/MoNuSeg/masks/test/TCGA-2Z-A9J9-01A-01-TS1.png", cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread("/home/joren/localhonours/code/CNSBench/yolov8_work_dirs/export/stainnorm/MoNuSeg/yolov8/TCGA-2Z-A9J9-01A-01-TS1.png", cv2.IMREAD_GRAYSCALE)

    print(metrics.general_object_hausdorff(gt, pred))

if __name__=="__main__":
    main()