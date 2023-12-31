import cv2
from cnsbench import metrics
import cProfile, pstats

def main():
    control_gt = cv2.imread("/home/joren/localhonours/code/CNSBench/datasets/MoNuSeg/masks/test/TCGA-2Z-A9J9-01A-01-TS1.png", cv2.IMREAD_GRAYSCALE)
    control_pred = cv2.imread("/home/joren/localhonours/code/CNSBench/yolov8_work_dirs/export/stainnorm/MoNuSeg/yolov8/TCGA-2Z-A9J9-01A-01-TS1.png", cv2.IMREAD_GRAYSCALE)

    # No profiling for expected results
   #result_orig = metrics.general_object_hausdorff(control_gt, control_pred)
   #result_new = metrics.general_object_hausdorff_faster(control_gt, control_pred)
   #print(result_orig, result_new)

    test_gt = cv2.imread("/home/joren/localhonours/code/CNSBench/datasets/MoNuSAC/masks/test/TCGA-2Z-A9JG-01Z-00-DX1_1.png", cv2.IMREAD_GRAYSCALE)
    test_pred = cv2.imread("/home/joren/localhonours/code/CNSBench/work_dirs/export/stainnorm/MoNuSAC/deeplabv3plus/TCGA-2Z-A9JG-01Z-00-DX1_1.png", cv2.IMREAD_GRAYSCALE)

    with cProfile.Profile() as pr:
        result_orig = metrics.general_object_hausdorff_legacy(test_gt, test_pred)
        stats_orig = pstats.Stats(pr)

    with cProfile.Profile() as pr:
        result_new = metrics.general_object_hausdorff(test_gt, test_pred)
        stats_new = pstats.Stats(pr)

    print(result_orig, result_new)

    stats_orig.sort_stats(pstats.SortKey.CUMULATIVE)
    stats_new.sort_stats(pstats.SortKey.CUMULATIVE)
    stats_orig.print_stats(20)
    stats_new.print_stats(20)


if __name__=="__main__":
    main()