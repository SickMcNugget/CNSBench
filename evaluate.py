import argparse
from pathlib import Path
# import cnsbench.metrics
from cnsbench.evaluation import Comparer

def main(args: argparse.Namespace):
    # args.compare_root = "final_preds/nostainnorm/MoNuSAC/unet/"
    # args.dataset_root = "MoNuSAC/"
    evaluate_dataset(args)

def evaluate_dataset(args: argparse.Namespace):
    comparer = Comparer(args.dataset_root)
    # df = comparer.compare("../NucleiSegmentation_mmseg/export")
    df = comparer.compare(args.compare_root)
    print(df)
    print(df.groupby("split").mean().loc[:, "f1":"iou"].round(3))

        # zip_paths = downloader_cls().download()
    
        # print(f"\nUnzipping {dataset}")
        # unzip_paths = Unzipper(zip_paths).unzip()

        # print(f"\nOrganising {dataset}")
        # mover_cls = get_dataset_class(dataset, "Mover")
        # mover_cls(args.dataset_root, unzip_paths).move_all()

        # print(f"\nGenerate masks for {dataset}")
        # mask_generator_cls = get_dataset_class(dataset, "MaskGenerator")
        # mask_generator_cls(args.dataset_root).generate_masks()

        # print(f"\nCreating YOLO compatible training data for {dataset}\n")
        # yolofier_cls = get_dataset_class(dataset, "Yolofier")
        # yolofier_cls(args.dataset_root).yolofy()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-root", default=Path("."), type=Path, help="The path where predictions lie")
    parser.add_argument("--dataset-root", default=Path("."), type=Path, help="The path to output the dataset")
    args = parser.parse_args()
    
    return args

if __name__=="__main__":
    args = get_args()

    # Ensure old profiling data is cleaned up
    main(args)