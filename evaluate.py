import argparse
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
from cnsbench import metrics
import cv2

def main(args: argparse.Namespace):
    # args.compare_root = "final_preds/nostainnorm/MoNuSAC/unet/"
    # args.dataset_root = "MoNuSAC/"
    maskdir = get_maskdir(args)
    gt_paths = sorted(maskdir.glob("*.png"))
    pred_paths = sorted(args.compare_root.glob("*.png"))

    results = {}

    pooldata = list(zip(gt_paths, pred_paths))
    print(pooldata)

    with Pool() as pool:
        comparisons = pool.starmap(compare_masks, pooldata)

    for comparison in comparisons:
        for key, value in comparison.items():
            if not key in results:
                results[key] = []
            results[key].append(value)

    df = pd.DataFrame.from_dict(results)
    df.set_index("name", inplace=True)
        
    print(df)
    print(df.groupby("split").mean().loc[:, "f1":"hausdorff"].round(3))

def compare_masks(gt_path: Path, pred_path: Path):
    if gt_path.name != pred_path.name:
        print("Woah there buddy something seems to have gone awry")

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

    comparisons = {}
    comparisons["name"] = gt_path.stem

    parts = gt_path.parts
    if "train" in parts:
        comparisons["split"] = "train"
    elif "val" in parts:
        comparisons["split"] = "val"
    elif "test" in parts:
        comparisons["split"] = "test"

    comparisons["accuracy"] = metrics.accuracy(gt, pred)
    comparisons["precision"] = metrics.precision(gt, pred)
    comparisons["recall"] = metrics.recall(gt, pred)
    comparisons["f1"] = metrics.f1(comparisons["precision"], comparisons["recall"])
    comparisons["iou"] = metrics.iou(gt, pred)
    comparisons["hausdorff"] = metrics.general_object_hausdorff(gt, pred)

    return comparisons

def get_maskdir(args: argparse.Namespace):
    maskdir = args.dataset_root / args.dataset / "masks" / "test"
    return Path(maskdir)

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "CryoNuSeg", "TNBC"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS, required=False, help="The dataset to use when comparing masks")
    parser.add_argument("--compare-root", type=Path, required=False, help="The path where predictions lie")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"), help="The path where datasets are stored")
    args = parser.parse_args()

    args.dataset="MoNuSeg"
    args.compare_root = Path("yolov8_work_dirs/export/stainnorm/MoNuSeg/yolov8/")
    return args

if __name__=="__main__":
    args = get_args()

    # Ensure old profiling data is cleaned up
    main(args)
