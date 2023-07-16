import cv2
import numpy as np
import argparse
from pathlib import Path
from skimage.draw import polygon
import imagesize
from multiprocessing import Pool

OUT_DIR = Path("yolo_to_mask")

def main(args: argparse.Namespace):
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)

    annotation_files: list[Path] = sorted(args.predictions.glob("*.txt"))
    assert len(annotation_files) > 0, "No annotation files found"

    pooldata = [(args.dataset_root, annotation_file) for annotation_file in annotation_files]
    with Pool() as pool:
        pool.starmap(_yolo_to_mask, pooldata)
    # for annotation_file in annotation_files:
    #     _yolo_to_mask(args.dataset_root, annotation_file)

def _yolo_to_mask(dataset_path: Path, annotation_file: Path):
    """Convert yolo annotations into binary masks using the shape of the images that the predictions are based on."""
    with open(annotation_file) as f:
        dump = f.read()

    associated_img = sorted(dataset_path.glob(f"**/{annotation_file.with_suffix('.png').name}"))
    assert len(associated_img) == 1, len(associated_img)
    shape = imagesize.get(associated_img[0])

    annotations = dump.splitlines()

    try:
        nuclei = read_yolo_segmentations(annotations, shape)
    except ValueError:
        print(f"Bad annotation in file: {annotation_file.name}")
    
    mask = np.zeros((shape[1], shape[0]), dtype=np.uint8)
    for nucleus in nuclei:
        # Polygon objects allow for greater flexibility when drawing the binary masks
        # mask_polygon = Polygon(blob)
        # coords = np.asarray(mask_polygon.exterior.coords)
        # coordinates needed to be swapped to line up with original image
        rr, cc = polygon(nucleus[:,1], nucleus[:,0], (shape[1], shape[0]))
        mask[rr, cc] = 1

    cv2.imwrite(str(OUT_DIR / annotation_file.with_suffix(".png").name), mask)

def read_yolo_segmentations(annotations: list[str], shape: tuple[int, int]):
    annotations = [annotation[2:] for annotation in annotations]

    nuclei = []
    for annotation in annotations:
        if len(annotation) == 0:
            continue

        split_annotation = annotation.split(" ")
        split_annotation = [float(value) for value in split_annotation]
        split_annotation = [(split_annotation[i], split_annotation[i+1]) for i in range(0, len(split_annotation), 2)]

        nucleus = np.zeros((len(split_annotation), 2))
        for i, (x, y) in enumerate(split_annotation):
            # rounded_x = round(x * shape[0])
            # rounded_y = round(y * shape[1])
            nucleus[i, 0] = x * shape[0]
            nucleus[i, 1] = y * shape[1]
        nuclei.append(nucleus)

    return nuclei

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("MoNuSeg/yolo/train"), help="The path containing yolo segmentation results")
    parser.add_argument("--dataset-root", type=Path, default=Path("."), help="The root folder where datasets are stored")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = get_args()
    main(args)