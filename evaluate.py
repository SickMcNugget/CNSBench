import argparse
from pathlib import Path

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
    for gt_path, pred_path in zip(gt_paths, pred_paths):
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

        comparisons["accuracy"] = metrics.calc_accuracy(gt, pred)
        comparisons["precision"] = metrics.calc_precision(gt, pred)
        comparisons["recall"] = metrics.calc_recall(gt, pred)
        comparisons["f1"] = metrics.calc_f1(comparisons["precision"], comparisons["recall"])
        comparisons["iou"] = metrics.calc_iou(gt, pred)
        # comparisons["hausdorff"] = metrics.calc_hausdorff(gt, pred)

        for key, value in comparisons.items():
            if not key in results:
                results[key] = []
            results[key].append(value)

        df = pd.DataFrame.from_dict(results)
        df.set_index("name", inplace=True)
        
        print(df)
        print(df.groupby("split").mean().loc[:, "f1":"iou"].round(3))

#def evaluate_dataset(args: argparse.Namespace):
#    comparer = Comparer(args.dataset_root)
#    # df = comparer.compare("../NucleiSegmentation_mmseg/export")
#    df = comparer.compare(args.compare_root)
#    print(df)
#    print(df.groupby("split").mean().loc[:, "f1":"iou"].round(3))

def get_maskdir(args: argparse.Namespace):
    maskdir = args.dataset_root / args.dataset / "masks" / "test"
    return Path(maskdir)

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "CryoNuSeg", "TNBC"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS, required=True, help="The dataset to use when comparing masks")
    parser.add_argument("--compare-root", type=Path, required=True, help="The path where predictions lie")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"), help="The path where datasets are stored")
    args = parser.parse_args()
    
    return args

if __name__=="__main__":
    args = get_args()

    # Ensure old profiling data is cleaned up
    main(args)