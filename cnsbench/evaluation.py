from __future__ import annotations
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import cnsbench.metrics as metrics

import cv2

class Comparer:
    def __init__(self, dataset_root: str | Path):
        if isinstance(dataset_root, str):
            dataset_root = Path(dataset_root)
        self.dataset_root = dataset_root

    def get_gt_from_preds(self, pred_masks: list[Path]) -> list[Path]:
        all_gt_masks = self.dataset_root.glob(f"**/masks/**/*.png")
        
        gt_masks = []
        for pred_mask in pred_masks:
            for gt_mask in all_gt_masks:
                if pred_mask.name == gt_mask.name:
                    gt_masks.append(gt_mask)
                    break
        return gt_masks

    def compare(self, prediction: str | Path) -> pd.DataFrame:
        """prediction can be a folder or a single image"""
        if isinstance(prediction, str):
            prediction = Path(prediction)

        if prediction.is_dir():
            pred_masks = list(prediction.glob("**/*.png"))
        elif prediction.is_file():
            pred_masks = [prediction]
        gt_masks = self.get_gt_from_preds(pred_masks)
        
        pooldata = [(gt_mask, pred_mask) for gt_mask, pred_mask in zip(gt_masks, pred_masks)]
        
        results = {}
        with Pool() as pool:
            comparisons = pool.starmap(self._compare, pooldata)
            for comparison in comparisons:
                for key, value in comparison.items():
                    if not key in results:
                        results[key] = []
                    results[key].append(value)
        
        df = pd.DataFrame.from_dict(results)
        df.set_index("name", inplace=True)
        return df
        
    def _compare(self, gt_mask: Path, pred_mask: Path) -> dict:
        gt = cv2.imread(str(gt_mask), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_mask), cv2.IMREAD_GRAYSCALE)

        comparisons = {}
        comparisons["name"] = gt_mask.stem

        parts = gt_mask.parts
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
        return comparisons

    # def _compare()