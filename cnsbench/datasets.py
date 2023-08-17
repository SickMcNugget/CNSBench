from __future__ import annotations

# builtins
from abc import ABC, abstractmethod
from pathlib import Path
import sys
import subprocess
from zipfile import ZipFile
import shutil
from multiprocessing import Pool

# Linalg, parsing, image manip
import xml.etree.ElementTree as ET
from skimage.draw import polygon
import numpy as np
import cv2
from openslide import OpenSlide
import torchstain
from torchvision import transforms

# Progress
from tqdm import tqdm

# Dataset downloading
import gdown


class DownloaderError(Exception):
    """Raised when the downloader cannot download a file"""

    pass


class Downloader(ABC):
    @abstractmethod
    def download(self, zip_source: str, *args) -> Path:
        """Downloads a zip file from it's online id and returns the downloaded path"""


class GDownloader(Downloader):
    def __init__(self) -> None:
        super().__init__()

    def download(self, zip_source: str) -> Path:
        zip_path = gdown.download(id=zip_source)

        if zip_path is None:
            raise DownloaderError(f"Error downloading {zip_source} from google drive")

        return Path(zip_path)


class ZenodoDownloader(Downloader):
    def __init__(self) -> None:
        super().__init__()

    def download(self, zip_source: str) -> Path:
        try:
            subprocess.run(["zenodo_get", zip_source])
        except FileNotFoundError:
            pass


class KaggleDownloader(Downloader):
    def __init__(self) -> None:
        super().__init__()

    def download(self, zip_source: str) -> Path:
        try:
            ret = subprocess.run(
                ["kaggle", "datasets", "download", "-d", zip_source],
                stderr=subprocess.DEVNULL,
            )
            ret.check_returncode()
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise DownloaderError(
                "Please configure ~/.kaggle/kaggle.json. Follow the instructions"
                + " on their website."
            )


class Unzipper:
    def __init__(self) -> None:
        pass

    def unzip(self, zip_path: Path, unzip_folder: Path) -> Path:
        with ZipFile(zip_path, "r") as zfile:
            zfile.extractall(unzip_folder)

        return unzip_folder / zip_path.stem


class Organiser(ABC):
    def __init__(self) -> None:
        super().__init__()

    def copy_folder(self, folder: Path, dest: Path):
        for image in folder.iterdir():
            self.copy_if_new(image, dest)

    def copy_if_new(self, src: Path, dest: Path):
        if not (dest / src.name).exists():
            shutil.copy(src, dest)

    @abstractmethod
    def organise(self, unzip_paths: list[Path], destination_paths: list[Path], *args):
        pass


class MoNuSegOrganiser(Organiser):
    def __init__(self) -> None:
        super().__init__()

    def organise(self, unzip_paths: list[Path], destination_paths: list[Path]):
        # [0] is train/val zip file
        xmls = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/*.xml"))
        tifs = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/*.tif"))

        split_idx = -7
        train_xmls, train_tifs = xmls[:split_idx], tifs[:split_idx]
        val_xmls, val_tifs = xmls[split_idx:], tifs[split_idx:]

        train_path = destination_paths[0]
        for train_xml, train_tif in zip(train_xmls, train_tifs):
            self.copy_if_new(train_xml, train_path)
            self.copy_if_new(train_tif, train_path)

        val_path = destination_paths[1]
        for val_xml, val_tif in zip(val_xmls, val_tifs):
            self.copy_if_new(val_xml, val_path)
            self.copy_if_new(val_tif, val_path)

        test_files = (unzip_paths[1] / unzip_paths[1].stem).glob("*")
        test_path = destination_paths[2]
        for file in test_files:
            self.copy_if_new(file, test_path)


class MoNuSACOrganiser(Organiser):
    def __init__(self) -> None:
        super().__init__()

    def organise(self, unzip_paths: list[Path], destination_paths: list[Path]):
        # [0] is train/val zip file
        patients = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("*"))

        split_idx = -11
        train_patients = patients[:split_idx]
        val_patients = patients[split_idx:]

        pooldata = [(patient, destination_paths[0]) for patient in train_patients]
        with Pool() as pool:
            for _ in tqdm(
                pool.istarmap(self.copy_folder, pooldata), total=len(pooldata)
            ):
                pass

        pooldata = [(patient, destination_paths[1]) for patient in val_patients]
        with Pool() as pool:
            for _ in tqdm(
                pool.istarmap(self.copy_folder, pooldata), total=len(pooldata)
            ):
                pass

        test_patients = sorted((unzip_paths[1] / unzip_paths[1].stem).glob("*"))

        pooldata = [(patient, destination_paths[2]) for patient in test_patients]
        with Pool() as pool:
            for _ in tqdm(
                pool.istarmap(self.copy_folder, pooldata), total=len(pooldata)
            ):
                pass


class CryoNuSegOrganiser(Organiser):
    def __init__(self) -> None:
        super().__init__()

    def organise(
        self,
        unzip_paths: list[Path],
        destination_paths: list[Path],
        mask_paths: list[Path],
    ):
        # [0] is train/val zip file
        train_tifs = sorted(unzip_paths[0].glob("tissue images/*[12].tif"))
        val_test_tifs = sorted(unzip_paths[0].glob("tissue images/*3.tif"))
        val_tifs = val_test_tifs[:5]
        test_tifs = val_test_tifs[5:]

        for train_tif in train_tifs:
            self.copy_if_new(train_tif, destination_paths[0])
        for val_tif in val_tifs:
            self.copy_if_new(val_tif, destination_paths[1])
        for test_tif in test_tifs:
            self.copy_if_new(test_tif, destination_paths[2])

        train_mask_tifs = sorted(
            unzip_paths[0].glob("Annotator 2*/**/mask binary without border/*[12].png")
        )
        val_test_mask_tifs = sorted(
            unzip_paths[0].glob("Annotator 2*/**/mask binary without border/*3.png")
        )
        val_mask_tifs = val_test_mask_tifs[:5]
        test_mask_tifs = val_test_mask_tifs[5:]

        for train_mask_tif in train_mask_tifs:
            self.copy_if_new(train_mask_tif, mask_paths[0])
        for val_mask_tif in val_mask_tifs:
            self.copy_if_new(val_mask_tif, mask_paths[1])
        for test_mask_tif in test_mask_tifs:
            self.copy_if_new(test_mask_tif, mask_paths[2])


class TNBCOrganiser(Organiser):
    def __init__(self) -> None:
        super().__init__()

    def organise(
        self,
        unzip_paths: list[Path],
        destination_paths: list[Path],
        mask_paths: list[Path],
    ):
        # [0] is train/val zip file
        folders = sorted(unzip_paths[0].glob(f"**/Slide_*"))
        train_folders = folders[:6]
        val_folders = [folders[i] for i in [6, 9, 10]]
        test_folders = [folders[i] for i in [7, 8]]

        for train_folder in train_folders:
            self.copy_folder(train_folder, destination_paths[0])
        for val_folder in val_folders:
            self.copy_folder(val_folder, destination_paths[1])
        for test_folder in test_folders:
            self.copy_folder(test_folder, destination_paths[2])

        mask_folders = sorted((unzip_paths[0] / unzip_paths[0].stem).glob(f"**/GT_*"))
        train_mask_folders = mask_folders[:6]
        val_mask_folders = [mask_folders[i] for i in [6, 9, 10]]
        test_mask_folders = [mask_folders[i] for i in [7, 8]]

        for train_mask_folder in train_mask_folders:
            self.copy_folder(train_mask_folder, mask_paths[0])
        for val_mask_folder in val_mask_folders:
            self.copy_folder(val_mask_folder, mask_paths[1])
        for test_mask_folder in test_mask_folders:
            self.copy_folder(test_mask_folder, mask_paths[2])


class MaskGenerator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate_masks(self, original_paths: list[Path], mask_paths: list[Path], *args):
        pass

    @abstractmethod
    def get_pooldata(self, original_path: Path, mask_path: Path, *args):
        pass

    @abstractmethod
    def generate_mask(self, source_data: Path, *args):
        pass


class MoNuSegMaskGenerator(MaskGenerator):
    def __init__(self, xml_parser: XMLParser) -> None:
        super().__init__()
        self.xml_parser = xml_parser

    def generate_masks(self, original_paths, mask_paths):
        for original_path, mask_path in zip(original_paths, mask_paths):
            pooldata = self.get_pooldata(original_path, mask_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.istarmap(self.generate_mask, pooldata), total=len(pooldata)
                ):
                    pass

    def get_pooldata(self, original_path: Path, mask_path: Path):
        patients = sorted(original_path.glob("*.xml"))
        return [(patient, mask_path) for patient in patients]

    def generate_mask(self, patient: Path, mask_path: Path):
        patient_image = patient.with_suffix(".tif")
        if not patient_image.exists():
            return

        height, width = cv2.imread(str(patient_image)).shape[:2]

        nuclei = self.xml_parser.parse(patient)

        # skimage.draw.polygon uses the point-in-polygon test for accuracy!
        # DO NOT use opencv to draw polygons that are defined with real coordinates
        mask = np.zeros((height, width))
        for nucleus in nuclei:
            rr, cc = polygon(nucleus[:, 1], nucleus[:, 0], mask.shape)
            mask[rr, cc] = 1

        # Perform morphological opening to remove erroneous pixels.
        # These exist as the original annotations contain loops in the vertices.
        # These loops result in non-convex shapes that create "holes"
        # in the corners of the nuclei in masks which should be removed.
        # NOTE: I cannot do this, as it would damage the annotations.
        # This must simply be left be, as fixing the problem causes more problems,
        # due to the fact that nuclei are so close to one another and so small.
        # An example of the phenomenon: masks/test/TCGA-2Z-A9J9, coord (x,y): (335,73)

        filename = str((mask_path / patient.stem).with_suffix(".png"))
        cv2.imwrite(filename, mask)


class MoNuSACMaskGenerator(MaskGenerator):
    def __init__(self, xml_parser: XMLParser) -> None:
        super().__init__()
        self.xml_parser = xml_parser

    def generate_masks(self, original_paths, mask_paths):
        for original_path, mask_path in zip(original_paths, mask_paths):
            pooldata = self.get_pooldata(original_path, mask_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.istarmap(self.generate_mask, pooldata), total=len(pooldata)
                ):
                    pass

    def get_pooldata(self, original_path: Path, mask_path: Path):
        patients = sorted(original_path.glob("*.xml"))
        return [(patient, mask_path) for patient in patients]

    def generate_mask(self, patient: Path, mask_path: Path):
        patient_image = patient.with_suffix(".svs")
        if not patient_image.exists():
            return

        slide = OpenSlide(str(patient_image))
        width, height = slide.level_dimensions[0]
        slide.close()
        nuclei = self.xml_parser.parse(patient)

        # skimage.draw.polygon uses the point-in-polygon test for accuracy!
        # DO NOT use opencv to draw polygons that are defined with real coordinates
        mask = np.zeros((height, width))
        for nucleus in nuclei:
            rr, cc = polygon(nucleus[:, 1], nucleus[:, 0], mask.shape)
            mask[rr, cc] = 1

        # Perform morphological opening to remove erroneous pixels.
        # These exist as the original annotations contain loops in the vertices.
        # These loops result in non-convex shapes that create "holes"
        # in the corners of the nuclei in masks which should be removed.
        # NOTE: I cannot do this, as it would damage the annotations.
        # This must simply be left be, as fixing the problem causes more problems,
        # due to the fact that nuclei are so close to one another and so small.
        # An example of the phenomenon: masks/test/TCGA-2Z-A9J9, coord (x,y): (335,73)

        filename = str((mask_path / patient.stem).with_suffix(".png"))
        cv2.imwrite(filename, mask)


class CryoNuSegMaskGenerator(MaskGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate_masks(self, original_paths, mask_paths):
        for original_path, mask_path in zip(original_paths, mask_paths):
            pooldata = self.get_pooldata(original_path, mask_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.imap(self.generate_mask, pooldata), total=len(pooldata)
                ):
                    pass

    def get_pooldata(self, original_path: Path, mask_path: Path):
        masks = list(mask_path.glob("*.png"))
        return [mask for mask in masks]

    def generate_mask(self, mask: Path):
        img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        img[img == 255] = 1
        cv2.imwrite(str(mask), img)


class TNBCMaskGenerator(MaskGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate_masks(self, original_paths, mask_paths):
        for original_path, mask_path in zip(original_paths, mask_paths):
            pooldata = self.get_pooldata(original_path, mask_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.imap(self.generate_mask, pooldata), total=len(pooldata)
                ):
                    pass

    def get_pooldata(self, original_path: Path, mask_path: Path):
        masks = list(mask_path.glob("*.png"))
        return [mask for mask in masks]

    def generate_mask(self, mask: Path):
        img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        img[img == 255] = 1
        cv2.imwrite(str(mask), img)


# class MoNuSACMaskGenerator(MaskGenerator):
#     def __init__(self, dataset_root: str | Path):
#         super().__init__(dataset_root)
#         self.xml_parser = MoNuSACXMLParser()
#         self.label_map = {'Epithelial':1,
#                           'Lymphocyte':2,
#                           'Neutrophil':3,
#                           'Macrophage':4,
#                           'Ambiguous':5,}
#         self.masks_1cls = self.dataset_root / "masks_1cls"
#         for mask_path in [self.masks_1cls / "train", self.masks_1cls / "val", self.masks_1cls / "test"]:
#             if not mask_path.exists():
#                 mask_path.mkdir(parents=True)

#     def _get_pooldata(self, mask_path: Path, original_path: Path):
#         patients = sorted(original_path.glob("*.xml"))
#         return [(mask_path, patient) for patient in patients]

#     def _generate_mask(self, mask_path: Path, patient: Path):
#         patient_image = patient.with_suffix(".svs")
#         if not patient_image.exists():
#             return

#         slide = OpenSlide(str(patient_image))
#         width, height = slide.level_dimensions[0]
#         slide.close()
#         nuclei = self.xml_parser.parse(patient)

#         # skimage.draw.polygon uses the point-in-polygon test for accuracy!
#         # DO NOT use opencv to draw polygons that are defined with real coordinates
#         mask = np.zeros((height, width))
#         for nucleus, class_label in nuclei:
#             rr, cc = polygon(nucleus[:,1], nucleus[:,0], mask.shape)
#             mask[rr,cc] = self.label_map[class_label]

#         filename = str((mask_path / patient.stem).with_suffix(".png"))
#         cv2.imwrite(filename, mask)

#         mask_1cls = np.zeros((height, width))
#         for nucleus, class_label in nuclei:
#             rr, cc = polygon(nucleus[:,1], nucleus[:,0], mask_1cls.shape)
#             mask_1cls[rr,cc] = 1

#         filename = str((self.masks_1cls / mask_path.stem / patient.stem).with_suffix(".png"))
#         cv2.imwrite(filename, mask_1cls)


class XMLParser(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def parse(self, xml: Path) -> list[np.ndarray]:
        pass


class MoNuSegXMLParser(XMLParser):
    def __init__(self):
        super().__init__()

    def parse(self, xml: Path) -> list[np.ndarray]:
        tree = ET.parse(xml)
        root = tree.getroot()

        nuclei = []
        regions = root.findall("Annotation/Regions/Region")
        for region in regions:
            vertices = region.findall("Vertices/Vertex")

            # Invalid region by definition
            if len(vertices) < 3:
                continue

            nucleus = np.zeros((len(vertices), 2))
            for i, vertex in enumerate(vertices):
                nucleus[i, 0] = float(vertex.get("X"))
                nucleus[i, 1] = float(vertex.get("Y"))
            nuclei.append(nucleus)

        return nuclei


class MoNuSACXMLParser(XMLParser):
    def __init__(self):
        super().__init__()

    def parse(self, xml: Path) -> list[np.ndarray]:
        tree = ET.parse(xml)
        root = tree.getroot()

        nuclei = []
        regions = root.findall("Annotation/Regions/Region")
        for region in regions:
            vertices = region.findall("Vertices/Vertex")

            # Invalid region by definition
            if len(vertices) < 3:
                continue

            nucleus = np.zeros((len(vertices), 2))
            for i, vertex in enumerate(vertices):
                nucleus[i, 0] = float(vertex.get("X"))
                nucleus[i, 1] = float(vertex.get("Y"))
            nuclei.append(nucleus)

        return nuclei


# This multi-class MoNuSAC XMLParser is deprecated. MoNuSAC is treated binary for this project
# class MoNuSACXMLParser(XMLParser):
#     def __init__(self):
#         super().__init__()

#     def parse(self, xml: Path) -> list[np.ndarray]:
#         tree = ET.parse(xml)
#         root = tree.getroot()

#         nuclei = []
#         annotations = root.findall("Annotation")
#         for annotation in annotations:
#             class_label = annotation.find("Attributes/Attribute").get("Name")
#             regions = annotation.findall("Regions/Region")
#             if len(regions) == 0: # Some errors in annotations
#                 continue
#             for region in regions:
#                 vertices = region.findall("Vertices/Vertex")

#                 # Invalid region by definition
#                 if len(vertices) < 3:
#                     continue

#                 nucleus = np.zeros((len(vertices), 2))
#                 for i, vertex in enumerate(vertices):
#                     nucleus[i,0] = float(vertex.get("X"))
#                     nucleus[i,1] = float(vertex.get("Y"))
#                 nuclei.append((nucleus, class_label))

#         return nuclei


class Yolofier(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def yolofy_files(self, original_paths: list[Path], yolofy_paths: list[Path], *args):
        pass

    @abstractmethod
    def get_pooldata(self, original_path: Path, yolofy_path: Path, *args):
        pass

    @abstractmethod
    def yolofy_file(self, source_data: Path, *args):
        pass


class MoNuSegYolofier(Yolofier):
    def __init__(self, xml_parser: XMLParser) -> None:
        super().__init__()
        self.xml_parser = xml_parser

    def yolofy_files(self, original_paths, yolofy_paths):
        for original_path, yolofy_path in zip(original_paths, yolofy_paths):
            pooldata = self.get_pooldata(original_path, yolofy_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.istarmap(self.yolofy_file, pooldata), total=len(pooldata)
                ):
                    pass

    def get_pooldata(self, original_path: Path, yolofy_path: Path):
        patients = sorted(original_path.glob("*.xml"))
        return [(patient, yolofy_path) for patient in patients]

    def yolofy_file(self, patient_xml: Path, yolofy_path: Path):
        patient_image = patient_xml.with_suffix(".tif")

        if not patient_image.exists():
            return

        # YOLOv8 requires .png images
        patient_img = cv2.imread(str(patient_image))
        filename = str((yolofy_path / patient_image.stem).with_suffix(".png"))
        cv2.imwrite(filename, patient_img)

        height, width = patient_img.shape[:2]
        nuclei = self.xml_parser.parse(patient_xml)

        normalised_nuclei = []
        for nucleus in nuclei:
            nucleus[:, 0] /= width
            nucleus[:, 1] /= height
            normalised_nuclei.append(nucleus)

        annotations = self.nuclei_to_annotations(normalised_nuclei)

        filename = (yolofy_path / patient_image.stem).with_suffix(".txt")
        with open(filename, "w") as f:
            f.write(annotations)

    def nuclei_to_annotations(self, nuclei: np.ndarray) -> str:
        annotations = ""

        for nucleus in nuclei:
            annotations += "0"
            for vertex in nucleus:
                annotations += f" {vertex[0]} {vertex[1]}"
            # close off the nucleus polygon (using first vertex of contour)
            annotations += f" {nucleus[0, 0]} {nucleus[0, 1]}\n"

        return annotations


class MoNuSACYolofier(Yolofier):
    def __init__(self, xml_parser: XMLParser) -> None:
        super().__init__()
        self.xml_parser = xml_parser

    def yolofy_files(self, original_paths, yolofy_paths):
        for original_path, yolofy_path in zip(original_paths, yolofy_paths):
            pooldata = self.get_pooldata(original_path, yolofy_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.istarmap(self.yolofy_file, pooldata), total=len(pooldata)
                ):
                    pass

    def get_pooldata(self, original_path: Path, yolofy_path: Path):
        patients = sorted(original_path.glob("*.xml"))
        return [(patient, yolofy_path) for patient in patients]

    def yolofy_file(self, patient_xml: Path, yolofy_path: Path):
        patient_image = patient_xml.with_suffix(".svs")

        if not patient_image.exists():
            return

        slide = OpenSlide(str(patient_image))
        filename = str((yolofy_path / patient_image.stem).with_suffix(".png"))
        slide.get_thumbnail(slide.level_dimensions[0]).save(filename)
        width, height = slide.level_dimensions[0]
        slide.close()

        nuclei = self.xml_parser.parse(patient_xml)

        normalised_nuclei = []
        for nucleus in nuclei:
            nucleus[:, 0] /= width
            nucleus[:, 1] /= height
            normalised_nuclei.append(nucleus)

        annotations = self.nuclei_to_annotations(normalised_nuclei)

        filename = (yolofy_path / patient_image.stem).with_suffix(".txt")
        with open(filename, "w") as f:
            f.write(annotations)

    def nuclei_to_annotations(self, nuclei: np.ndarray) -> str:
        annotations = ""

        for nucleus in nuclei:
            annotations += "0"
            for vertex in nucleus:
                annotations += f" {vertex[0]} {vertex[1]}"
            # close off the nucleus polygon (using first vertex of contour)
            annotations += f" {nucleus[0, 0]} {nucleus[0, 1]}\n"

        return annotations


class CryoNuSegYolofier(Yolofier):
    def __init__(self) -> None:
        super().__init__()

    def yolofy_files(
        self,
        original_paths: list[Path],
        yolofy_paths: list[Path],
        mask_paths: list[Path],
    ):
        for original_path, yolofy_path, mask_path in zip(
            original_paths, yolofy_paths, mask_paths
        ):
            pooldata = self.get_pooldata(original_path, yolofy_path, mask_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.istarmap(self.yolofy_file, pooldata), total=len(pooldata)
                ):
                    pass

    def get_pooldata(self, original_path: Path, yolofy_path: Path, mask_path: Path):
        patient_imgs = sorted(original_path.glob("*.tif"))
        patient_masks = sorted(mask_path.glob("*.png"))
        return [
            (patient_img, patient_mask, yolofy_path)
            for patient_img, patient_mask in zip(patient_imgs, patient_masks)
        ]

    def yolofy_file(self, patient_img: Path, patient_mask: Path, yolofy_path: Path):
        img = cv2.imread(str(patient_img))
        cv2.imwrite(str(yolofy_path / patient_img.with_suffix(".png").name), img)

        mask = cv2.imread(str(patient_mask), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]

        nuclei = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        normalised_nuclei = []
        for nucleus in nuclei:
            float_nucleus = nucleus.astype(float)
            float_nucleus[:, :, 0] /= width
            float_nucleus[:, :, 1] /= height
            normalised_nuclei.append(float_nucleus)

        annotations = self.nuclei_to_annotations(normalised_nuclei)

        filename = yolofy_path / patient_mask.with_suffix(".txt").name
        with open(filename, "w") as f:
            f.write(annotations)

    def nuclei_to_annotations(self, nuclei):
        annotations = ""
        for nucleus in nuclei:
            annotations += "0"
            for vertex in nucleus:
                annotations += f" {vertex[0, 0]} {vertex[0, 1]}"
            annotations += f" {nucleus[0, 0, 0]} {nucleus[0, 0, 1]}\n"
        return annotations


class TNBCYolofier(Yolofier):
    def __init__(self) -> None:
        super().__init__()

    def yolofy_files(
        self,
        original_paths: list[Path],
        yolofy_paths: list[Path],
        mask_paths: list[Path],
    ):
        for original_path, yolofy_path, mask_path in zip(
            original_paths, yolofy_paths, mask_paths
        ):
            pooldata = self.get_pooldata(original_path, yolofy_path, mask_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.istarmap(self.yolofy_file, pooldata), total=len(pooldata)
                ):
                    pass

    def get_pooldata(self, original_path: Path, yolofy_path: Path, mask_path: Path):
        patient_imgs = sorted(original_path.glob("*.png"))
        patient_masks = sorted(mask_path.glob("*.png"))
        return [
            (patient_img, patient_mask, yolofy_path)
            for patient_img, patient_mask in zip(patient_imgs, patient_masks)
        ]

    def yolofy_file(self, patient_img: Path, patient_mask: Path, yolofy_path: Path):
        # original image is already a PNG
        shutil.copy(patient_img, (yolofy_path / patient_img.name))

        mask = cv2.imread(str(patient_mask), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]

        nuclei = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        normalised_nuclei = []
        for nucleus in nuclei:
            float_nucleus = nucleus.astype(float)
            float_nucleus[:, :, 0] /= width
            float_nucleus[:, :, 1] /= height
            normalised_nuclei.append(float_nucleus)

        annotations = self.nuclei_to_annotations(normalised_nuclei)

        filename = yolofy_path / patient_mask.with_suffix(".txt").name
        with open(filename, "w") as f:
            f.write(annotations)

    def nuclei_to_annotations(self, nuclei):
        annotations = ""
        for nucleus in nuclei:
            annotations += "0"
            for vertex in nucleus:
                annotations += f" {vertex[0, 0]} {vertex[0, 1]}"
            annotations += f" {nucleus[0, 0, 0]} {nucleus[0, 0, 1]}\n"
        return annotations


class StainNormaliser:
    def __init__(self, choice: str, fit_image: Path) -> None:
        super().__init__()

        self.fit_image = fit_image
        self.T = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
        )

        if choice == "macenko":
            self.normaliser = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        elif choice == "reinhard":
            self.normaliser = torchstain.normalizers.ReinhardNormalizer(
                backend="torch", method="modified"
            )

        fit_img = cv2.cvtColor(cv2.imread(str(fit_image)), cv2.COLOR_BGR2RGB)
        self.normaliser.fit(self.T(fit_img))

    def normalise(self, yolo_paths: list[Path], yolosn_paths: list[Path]):
        for yolo_path, yolosn_path in zip(yolo_paths, yolosn_paths):
            pngs = sorted(yolo_path.glob("*.png"))
            txts = sorted(yolo_path.glob("*.txt"))
            for png, txt in zip(pngs, txts):
                if png.name != self.fit_image.name:
                    to_normalise = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
                    if "Reinhard" in self.normaliser.__class__.__name__:
                        norm = self.normaliser.normalize(I=self.T(to_normalise))
                    else:
                        norm, _, _ = self.normaliser.normalize(
                            I=self.T(to_normalise), stains=False
                        )

                    norm = norm.numpy().astype(np.uint8)
                    norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(yolosn_path / png.name), norm)
                else:
                    shutil.copy(png, yolosn_path)
                shutil.copy(txt, yolosn_path)


# class MoNuSACYolofier(Yolofier):
#     def __init__(self, dataset_root: str | Path):
#         super().__init__(dataset_root)
#         self.xml_parser = MoNuSACXMLParser()
#         self.label_map = {'Epithelial': 1,
#                           'Lymphocyte': 2,
#                           'Neutrophil': 3,
#                           'Macrophage': 4,
#                           'Ambiguous': 5,}
#         self.yolofy_1cls = self.dataset_root / "yolo_1cls"
#         for yolofy_path in [self.yolofy_1cls / "train", self.yolofy_1cls / "val", self.yolofy_1cls / "test"]:
#             if not yolofy_path.exists():
#                 yolofy_path.mkdir(parents=True)


#     def _get_pooldata(self, yolofy_path: Path, original_path: Path):
#         patient_images = sorted(original_path.glob("*.svs"))
#         patient_xmls = sorted(original_path.glob("*.xml"))
#         pooldata = [(yolofy_path, patient_image, patient_xml)
#                      for patient_image, patient_xml
#                      in zip(patient_images, patient_xmls)]
#         return pooldata

#     def _yolofy(self, yolofy_path: Path, patient_image: Path, patient_xml: Path):
#         if not patient_image.exists() or not patient_xml.exists():
#             return

#         # YOLOv8 requires .png images
#         slide = OpenSlide(str(patient_image))
#         filename = str((yolofy_path / patient_image.stem).with_suffix(".png"))
#         filename_1cls = str((self.yolofy_1cls / yolofy_path.stem / patient_image.stem).with_suffix(".png"))

#         slide.get_thumbnail(slide.level_dimensions[0]).save(filename)
#         slide.get_thumbnail(slide.level_dimensions[0]).save(filename_1cls)
#         width, height = slide.level_dimensions[0]
#         slide.close()

#         nuclei = self.xml_parser.parse(patient_xml)

#         normalised_contours = []
#         for nucleus, class_label in nuclei:
#             nucleus[:,0] /= width
#             nucleus[:,1] /= height
#             normalised_contours.append((nucleus, class_label))

#         annotations = self.nuclei_to_annotations(normalised_contours)
#         annotations_1cls = self.nuclei_to_annotations(normalised_contours, multiclass=False)

#         filename = (yolofy_path / patient_image.stem).with_suffix(".txt")
#         with open(filename, "w") as f:
#             f.write(annotations)

#         filename_1cls = (self.yolofy_1cls / yolofy_path.stem / patient_image.stem).with_suffix(".txt")
#         with open(filename_1cls, "w") as f:
#             f.write(annotations_1cls)

#     def nuclei_to_annotations(self, nuclei: list[tuple[np.ndarray, str]], multiclass=True) -> str:
#         annotations = ""

#         for nucleus, class_label in nuclei:
#             if multiclass:
#                 annotations += str(self.label_map[class_label])
#             else:
#                 annotations += "0"

#             for vertex in nucleus:
#                 annotations += f" {vertex[0]} {vertex[1]}"
#             # close off the nucleus polygon (using first vertex of contour)
#             annotations += f" {nucleus[0, 0]} {nucleus[0, 1]}\n"

#         return annotations


class Dataset(ABC):
    def __init__(
        self, dataset_root: Path | str, zip_folder: Path | str, unzip_folder: Path | str
    ) -> None:
        if isinstance(dataset_root, str):
            dataset_root = Path(dataset_root)
        if isinstance(zip_folder, str):
            zip_folder = Path(zip_folder)
        if isinstance(unzip_folder, str):
            unzip_folder = Path(unzip_folder)

        self.dataset_root = dataset_root
        self.zip_folder = zip_folder
        self.unzip_folder = unzip_folder

    @property
    def dataset_name(self) -> str:
        return self.__class__.__name__.split("Dataset")[0]

    @property
    def zip_sources(self) -> list[str]:
        return self.__class__.ZIP_SOURCES

    @property
    def expected_zip_names(self) -> list[Path]:
        expected_zip_names = self.__class__.ZIP_NAMES
        if any(isinstance(zip_name, str) for zip_name in expected_zip_names):
            expected_zip_names = [Path(zip_name) for zip_name in expected_zip_names]
        return expected_zip_names

    @property
    def original_paths(self) -> list[Path]:
        original_paths = [
            self.dataset_root / self.dataset_name / "original" / "train",
            self.dataset_root / self.dataset_name / "original" / "val",
            self.dataset_root / self.dataset_name / "original" / "test",
        ]
        return original_paths

    @property
    def mask_paths(self) -> list[Path]:
        mask_paths = [
            self.dataset_root / self.dataset_name / "masks" / "train",
            self.dataset_root / self.dataset_name / "masks" / "val",
            self.dataset_root / self.dataset_name / "masks" / "test",
        ]
        return mask_paths

    @property
    def yolofy_paths(self) -> list[Path]:
        yolofy_paths = [
            self.dataset_root / self.dataset_name / "yolo" / "train",
            self.dataset_root / self.dataset_name / "yolo" / "val",
            self.dataset_root / self.dataset_name / "yolo" / "test",
        ]
        return yolofy_paths

    @property
    def yolosn_paths(self) -> list[Path]:
        yolosn_paths = [
            self.dataset_root / self.dataset_name / "yolo_sn" / "train",
            self.dataset_root / self.dataset_name / "yolo_sn" / "val",
            self.dataset_root / self.dataset_name / "yolo_sn" / "test",
        ]
        return yolosn_paths

    def download(self, downloader: Downloader) -> list[Path]:
        zip_paths = []

        if not self.requires_download():
            print(f"{self.dataset_name} already downloaded, skipping...")
            zip_paths = [
                self.zip_folder / zip_name.name for zip_name in self.expected_zip_names
            ]
        else:
            for zip_source in self.zip_sources:
                downloaded_zip_path = downloader.download(zip_source)
                shutil.move(downloaded_zip_path, self.zip_folder)
                zip_paths.append(self.zip_folder / downloaded_zip_path.name)

        return zip_paths

    def requires_download(self) -> bool:
        for zip_name in self.expected_zip_names:
            if not (self.zip_folder / zip_name.name).exists():
                return True
        return False

    def unzip(self, unzipper: Unzipper, zip_paths: list[Path]) -> list[Path]:
        unzip_paths = []

        if not self.requires_unzip():
            print(f"{self.dataset_name} already unzipped, skipping...")
            unzip_paths = [
                self.unzip_folder / zip_name.stem
                for zip_name in self.expected_zip_names
            ]
        else:
            for zip_path in zip_paths:
                unzip_path = unzipper.unzip(
                    zip_path, (self.unzip_folder / zip_path.stem)
                )
                unzip_paths.append(unzip_path)

        return unzip_paths

    def requires_unzip(self) -> bool:
        for zip_name in self.expected_zip_names:
            if not (self.unzip_folder / zip_name.stem).exists():
                return True
        return False

    def organise(self, organiser: Organiser, unzip_paths: list[Path]):
        if not self.requires_organise():
            print(f"{self.dataset_name} already organised, skipping...")
        else:
            organiser.organise(unzip_paths, self.original_paths)

    def requires_organise(self):
        for path in self.original_paths:
            if not path.exists() or not any(path.iterdir()):
                return True
        return False

    def generate_masks(self, mask_generator: MaskGenerator):
        if not self.requires_generate_masks():
            print(f"{self.dataset_name} already has generated masks, skipping...")
        else:
            mask_generator.generate_masks(self.original_paths, self.mask_paths)

    def requires_generate_masks(self):
        for path in self.mask_paths:
            if not path.exists() or not any(path.iterdir()):
                return True
        return False

    def yolofy(self, yolofier: Yolofier):
        if not self.requires_yolofy():
            print(f"{self.dataset_name} already has yolo-trainable data, skipping...")
        else:
            yolofier.yolofy_files(self.original_paths, self.yolofy_paths)

    def requires_yolofy(self):
        for path in self.yolofy_paths:
            if not path.exists() or not any(path.iterdir()):
                return True
        return False

    def normalise(self, stain_normaliser: StainNormaliser, fit_image: Path):
        if not self.requires_stainnorm():
            print(f"{self.dataset_name} already has stain_normalised data, skipping...")
        else:
            stain_normaliser.normalise(fit_image, self.yolofy_paths, self.yolosn_paths)

    def requires_stainnorm(self):
        for path in self.yolosn_paths:
            if not path.exists() or not any(path.iterdir()):
                return True
            return False

    def make_paths(self, paths: list[Path]):
        for path in paths:
            if not path.exists():
                path.mkdir(parents=True)


class MoNuSegDataset(Dataset):
    ZIP_SOURCES = [
        "1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA",
        "1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw",
    ]
    ZIP_NAMES = ["MoNuSeg 2018 Training Data.zip", "MoNuSegTestData.zip"]

    def __init__(self, dataset_root: Path, **kwargs) -> None:
        super().__init__(dataset_root, **kwargs)

    def normalise(self, stain_normaliser: StainNormaliser):
        # Manually chosen
        fit_image = self.yolofy_paths[0] / "TCGA-A7-A13E-01Z-00-DX1.png"
        super().normalise(stain_normaliser, fit_image)


class MoNuSACDataset(Dataset):
    ZIP_SOURCES = [
        "1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq",
        "1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ",
    ]
    ZIP_NAMES = [
        "MoNuSAC_images_and_annotations.zip",
        "MoNuSAC Testing Data and Annotations.zip",
    ]

    def __init__(self, dataset_root: Path, **kwargs) -> None:
        super().__init__(dataset_root, **kwargs)

    def normalise(self, stain_normaliser: StainNormaliser):
        # Manually chosen
        fit_image = self.yolofy_paths[0] / "TCGA-A2-A0ES-01Z-00-DX1_2.png"
        super().normalise(stain_normaliser, fit_image)


class CryoNuSegDataset(Dataset):
    ZIP_SOURCES = ["ipateam/segmentation-of-nuclei-in-cryosectioned-he-images"]
    ZIP_NAMES = ["segmentation-of-nuclei-in-cryosectioned-he-images.zip"]

    def __init__(self, dataset_root: Path, **kwargs) -> None:
        super().__init__(dataset_root, **kwargs)

    def download(self, downloader: Downloader) -> list[Path]:
        zip_paths = []

        if not self.requires_download():
            print(f"{self.dataset_name} already downloaded, skipping...")
            zip_paths = [
                self.zip_folder / zip_name.name for zip_name in self.expected_zip_names
            ]
        else:
            for zip_source in self.zip_sources:
                downloaded_zip_path = downloader.download(
                    zip_source, self.expected_zip_names
                )
                shutil.move(downloaded_zip_path, self.zip_folder)
                zip_paths.append(self.zip_folder / downloaded_zip_path.name)

        return zip_paths

    def organise(self, organiser: Organiser, unzip_paths: list[Path]):
        if not self.requires_organise():
            print(f"{self.dataset_name} already organised, skipping...")
        else:
            organiser.organise(unzip_paths, self.original_paths, self.mask_paths)

    def requires_generate_masks(self):
        for path in self.mask_paths:
            if not path.exists() or not any(path.iterdir()):
                return True

            # Check one file each directory to see if it has been converted
            for file in path.iterdir():
                img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                if np.max(img) == 255:
                    return True
                break

        return False

    def yolofy(self, yolofier: Yolofier):
        if not self.requires_yolofy():
            print(f"{self.dataset_name} already has yolo-trainable data, skipping...")
        else:
            yolofier.yolofy_files(
                self.original_paths, self.yolofy_paths, self.mask_paths
            )

    def normalise(self, stain_normaliser: StainNormaliser):
        # Manually chosen
        fit_image = self.yolofy_paths[0] / "Human_Larynx_02.png"
        super().normalise(stain_normaliser, fit_image)


class TNBCDataset(Dataset):
    ZIP_SOURCES = ["10.5281/zenodo.1175282"]
    ZIP_NAMES = ["TNBC_NucleiSegmentation.zip"]

    def __init__(self, dataset_root: Path, **kwargs) -> None:
        super().__init__(dataset_root, **kwargs)

    def download(self, downloader: Downloader) -> list[Path]:
        zip_paths = []

        if not self.requires_download():
            print(f"{self.dataset_name} already downloaded, skipping...")
            zip_paths = [
                self.zip_folder / zip_name.name for zip_name in self.expected_zip_names
            ]
        else:
            for zip_source in self.zip_sources:
                downloaded_zip_path = downloader.download(
                    zip_source, self.expected_zip_names
                )
                shutil.move(downloaded_zip_path, self.zip_folder)
                zip_paths.append(self.zip_folder / downloaded_zip_path.name)

        return zip_paths

    def organise(self, organiser: Organiser, unzip_paths: list[Path]):
        if not self.requires_organise():
            print(f"{self.dataset_name} already organised, skipping...")
        else:
            organiser.organise(unzip_paths, self.original_paths, self.mask_paths)

    def requires_generate_masks(self):
        for path in self.mask_paths:
            if not path.exists() or not any(path.iterdir()):
                return True

            # Check one file each directory to see if it has been converted
            for file in path.iterdir():
                img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                if np.max(img) == 255:
                    return True
                break

        return False

    def yolofy(self, yolofier: Yolofier):
        if not self.requires_yolofy():
            print(f"{self.dataset_name} already has yolo-trainable data, skipping...")
        else:
            yolofier.yolofy_files(
                self.original_paths, self.yolofy_paths, self.mask_paths
            )

    def normalise(self, stain_normaliser: StainNormaliser):
        # Manually chosen
        fit_image = self.yolofy_paths[0] / "03_1.png"
        super().normalise(stain_normaliser, fit_image)


class DatasetManager:
    def __init__(self, dataset_root: Path | str) -> None:
        if isinstance(dataset_root, str):
            dataset_root = Path(dataset_root)
        self.dataset_root = dataset_root

    @property
    def zip_folder(self):
        return self.dataset_root / "zips"

    @property
    def unzip_folder(self):
        return self.dataset_root / "unzips"

    def create_directories(self, dataset: Dataset):
        if not self.zip_folder.exists():
            self.zip_folder.mkdir()
        if not self.unzip_folder.exists():
            self.unzip_folder.mkdir()

        dataset.make_paths(dataset.original_paths)
        dataset.make_paths(dataset.mask_paths)
        dataset.make_paths(dataset.yolofy_paths)
        dataset.make_paths(dataset.yolosn_paths)

    def prepare(self, dataset_name: str):
        if dataset_name.lower() == "monuseg":
            dataset = MoNuSegDataset(
                self.dataset_root,
                zip_folder=self.zip_folder,
                unzip_folder=self.unzip_folder,
            )
            downloader = GDownloader()
            organiser = MoNuSegOrganiser()
            xml_parser = MoNuSegXMLParser()
            mask_generator = MoNuSegMaskGenerator(xml_parser=xml_parser)
            yolofier = MoNuSegYolofier(xml_parser=xml_parser)
        elif dataset_name.lower() == "monusac":
            dataset = MoNuSACDataset(
                self.dataset_root,
                zip_folder=self.zip_folder,
                unzip_folder=self.unzip_folder,
            )
            downloader = GDownloader()
            organiser = MoNuSACOrganiser()
            xml_parser = MoNuSACXMLParser()
            mask_generator = MoNuSACMaskGenerator(xml_parser=xml_parser)
            yolofier = MoNuSACYolofier(xml_parser=xml_parser)
        elif dataset_name.lower() == "cryonuseg":
            dataset = CryoNuSegDataset(
                self.dataset_root,
                zip_folder=self.zip_folder,
                unzip_folder=self.unzip_folder,
            )
            downloader = KaggleDownloader()
            organiser = CryoNuSegOrganiser()
            mask_generator = CryoNuSegMaskGenerator()
            yolofier = CryoNuSegYolofier()
        elif dataset_name.lower() == "tnbc":
            dataset = TNBCDataset(
                self.dataset_root,
                zip_folder=self.zip_folder,
                unzip_folder=self.unzip_folder,
            )
            downloader = ZenodoDownloader()
            organiser = TNBCOrganiser()
            mask_generator = TNBCMaskGenerator()
            yolofier = TNBCYolofier()

        unzipper = Unzipper()
        stain_normaliser = StainNormaliser("reinhard")

        try:
            self.create_directories(dataset)

            print(f"Attempting to download {dataset.dataset_name}")
            zip_paths = dataset.download(downloader)

            print(f"Unzipping {dataset.dataset_name}")
            unzip_paths = dataset.unzip(unzipper, zip_paths)

            print(f"Organising {dataset.dataset_name}")
            dataset.organise(organiser, unzip_paths)

            print(f"Generate masks for {dataset.dataset_name}")
            dataset.generate_masks(mask_generator)

            print(f"Creating YOLO compatible training data for {dataset.dataset_name}")
            dataset.yolofy(yolofier)

            print(
                f"Creating stain normalised variation of YOLO images for {dataset.dataset_name}"
            )
            dataset.normalise(stain_normaliser)
        except DownloaderError as e:
            print(e)


def copy_folder_if_new(folder: Path, dest: Path):
    for image in folder.iterdir():
        copy_file_if_new(image, dest)


def copy_file_if_new(src: Path, dest: Path):
    if not (dest / src.name).exists():
        shutil.copy(src, dest)


def make_folder_if_new(path: Path):
    if not path.exists():
        path.mkdir(parents=True)


def make_directories(dataset_root: Path, dataset_name: str):
    base_directory = dataset_root / dataset_name / "original"
    masks_directory = dataset_root / dataset_name / "masks"
    yolo_directory = dataset_root / dataset_name / "yolo"
    yolo_sn_directory = dataset_root / dataset_name / "yolo_sn"

    for path in {"train", "val", "test"}:
        make_folder_if_new((base_directory / path))
        make_folder_if_new((masks_directory / path))
        make_folder_if_new((yolo_directory / path))
        make_folder_if_new((yolo_sn_directory / path))

    make_folder_if_new((dataset_root / "zips"))
    make_folder_if_new((dataset_root / "unzips"))

def generate_cryonuseg_mask(mask: Path):
    img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    img[img == 255] = 1
    cv2.imwrite(str(mask), img)

def yolofy_cryonuseg_file(patient_img: Path, patient_mask: Path, yolofy_path: Path):
    img = cv2.imread(str(patient_img))
    cv2.imwrite(str(yolofy_path / patient_img.with_suffix(".png").name), img)

    mask = cv2.imread(str(patient_mask), cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape[:2]

    nuclei = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    normalised_nuclei = []
    for nucleus in nuclei:
        float_nucleus = nucleus.astype(float)
        float_nucleus[:, :, 0] /= width
        float_nucleus[:, :, 1] /= height
        normalised_nuclei.append(float_nucleus)

    annotations = ""
    for normalised_nucleus in normalised_nuclei:
        annotations += "0"
        for vertex in normalised_nucleus:
            annotations += f" {vertex[0, 0]} {vertex[0, 1]}"
        annotations += f" {normalised_nucleus[0, 0, 0]} {normalised_nucleus[0, 0, 1]}\n"

    filename = yolofy_path / patient_mask.with_suffix(".txt").name
    with open(filename, "w") as f:
        f.write(annotations)

def generate_tnbc_mask(mask: Path):
    img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    img[img == 255] = 1
    cv2.imwrite(str(mask), img)


def yolofy_tnbc_file(patient_img: Path, patient_mask: Path, yolofy_path: Path):
    # original image is already a PNG
    shutil.copy(patient_img, (yolofy_path / patient_img.name))

    mask = cv2.imread(str(patient_mask), cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape[:2]

    nuclei = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    normalised_nuclei = []
    for nucleus in nuclei:
        float_nucleus = nucleus.astype(float)
        float_nucleus[:, :, 0] /= width
        float_nucleus[:, :, 1] /= height
        normalised_nuclei.append(float_nucleus)

    annotations = ""
    for normalised_nucleus in normalised_nuclei:
        annotations += "0"
        for vertex in normalised_nucleus:
            annotations += f" {vertex[0, 0]} {vertex[0, 1]}"
        annotations += f" {normalised_nucleus[0, 0, 0]} {normalised_nucleus[0, 0, 1]}\n"

    filename = yolofy_path / patient_mask.with_suffix(".txt").name
    with open(filename, "w") as f:
        f.write(annotations)


def generate_monuseg_mask(xml_parser: XMLParser, patient: Path, mask_path: Path):
    patient_image = patient.with_suffix(".tif")
    if not patient_image.exists():
        return

    height, width = cv2.imread(str(patient_image)).shape[:2]

    nuclei = xml_parser.parse(patient)

    # skimage.draw.polygon uses the point-in-polygon test for accuracy!
    # DO NOT use opencv to draw polygons that are defined with real coordinates
    mask = np.zeros((height, width))
    for nucleus in nuclei:
        rr, cc = polygon(nucleus[:, 1], nucleus[:, 0], mask.shape)
        mask[rr, cc] = 1

    # Perform morphological opening to remove erroneous pixels.
    # These exist as the original annotations contain loops in the vertices.
    # These loops result in non-convex shapes that create "holes"
    # in the corners of the nuclei in masks which should be removed.
    # NOTE: I cannot do this, as it would damage the annotations.
    # This must simply be left be, as fixing the problem causes more problems,
    # due to the fact that nuclei are so close to one another and so small.
    # An example of the phenomenon: masks/test/TCGA-2Z-A9J9, coord (x,y): (335,73)

    filename = str((mask_path / patient.stem).with_suffix(".png"))
    cv2.imwrite(filename, mask)


def yolofy_monuseg_file(xml_parser: XMLParser, patient_xml: Path, yolofy_path: Path):
    patient_image = patient_xml.with_suffix(".tif")

    if not patient_image.exists():
        return

    # YOLOv8 requires .png images
    patient_img = cv2.imread(str(patient_image))
    filename = str((yolofy_path / patient_image.stem).with_suffix(".png"))
    cv2.imwrite(filename, patient_img)

    height, width = patient_img.shape[:2]
    nuclei = xml_parser.parse(patient_xml)

    normalised_nuclei = []
    for nucleus in nuclei:
        nucleus[:, 0] /= width
        nucleus[:, 1] /= height
        normalised_nuclei.append(nucleus)

    # YOLO annotations require <label_id> (<x> <y>)*
    annotations = ""
    for normalised_nucleus in normalised_nuclei:
        annotations += "0"
        for vertex in normalised_nucleus:
            annotations += f" {vertex[0]} {vertex[1]}"
        # close off the normalised_nucleus polygon (using first vertex of contour)
        annotations += f" {normalised_nucleus[0, 0]} {normalised_nucleus[0, 1]}\n"

    filename = (yolofy_path / patient_image.stem).with_suffix(".txt")
    with open(filename, "w") as f:
        f.write(annotations)


def generate_monusac_mask(xml_parser: XMLParser, patient: Path, mask_path: Path):
    patient_image = patient.with_suffix(".tif")
    if not patient_image.exists():
        return

    height, width = cv2.imread(str(patient_image)).shape[:2]

    nuclei = xml_parser.parse(patient)

    # skimage.draw.polygon uses the point-in-polygon test for accuracy!
    # DO NOT use opencv to draw polygons that are defined with real coordinates
    mask = np.zeros((height, width))
    for nucleus in nuclei:
        rr, cc = polygon(nucleus[:, 1], nucleus[:, 0], mask.shape)
        mask[rr, cc] = 1

    # Perform morphological opening to remove erroneous pixels.
    # These exist as the original annotations contain loops in the vertices.
    # These loops result in non-convex shapes that create "holes"
    # in the corners of the nuclei in masks which should be removed.
    # NOTE: I cannot do this, as it would damage the annotations.
    # This must simply be left be, as fixing the problem causes more problems,
    # due to the fact that nuclei are so close to one another and so small.
    # An example of the phenomenon: masks/test/TCGA-2Z-A9J9, coord (x,y): (335,73)

    filename = str((mask_path / patient.stem).with_suffix(".png"))
    cv2.imwrite(filename, mask)


def yolofy_monusac_file(xml_parser: XMLParser, patient_xml: Path, yolofy_path: Path):
    patient_image = patient_xml.with_suffix(".svs")

    if not patient_image.exists():
        return

    slide = OpenSlide(str(patient_image))
    filename = str((yolofy_path / patient_image.stem).with_suffix(".png"))
    slide.get_thumbnail(slide.level_dimensions[0]).save(filename)
    width, height = slide.level_dimensions[0]
    slide.close()

    nuclei = xml_parser.parse(patient_xml)

    normalised_nuclei = []
    for nucleus in nuclei:
        nucleus[:, 0] /= width
        nucleus[:, 1] /= height
        normalised_nuclei.append(nucleus)

    annotations = ""

    for normalised_nucleus in normalised_nuclei:
        annotations += "0"
        for vertex in normalised_nucleus:
            annotations += f" {vertex[0]} {vertex[1]}"
        # close off the normalised_nucleus polygon (using first vertex of contour)
        annotations += f" {normalised_nucleus[0, 0]} {normalised_nucleus[0, 1]}\n"

    filename = (yolofy_path / patient_image.stem).with_suffix(".txt")
    with open(filename, "w") as f:
        f.write(annotations)


def requires_download(zip_folder: Path, zip_names: list[str]) -> bool:
    for zip_name in zip_names:
        if not (zip_folder / zip_name).exists():
            return True
    return False


def requires_unzip(unzip_folder: Path, zip_names: list[str]) -> bool:
    for zip_name in zip_names:
        if not ((unzip_folder / zip_name).with_suffix("")).exists():
            return True
    return False


def files_exist(folder: Path, file_names: set[str]) -> bool:
    for file in file_names:
        if not (folder / file).exists():
            return False
    return True


def get_monuseg(dataset_root: Path):
    ZIP_SOURCES = [
        "1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA",
        "1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw",
    ]
    # This can be a set, but make sure it is sorted when retrieved via an
    # iterator, as sets are UNORDERED. Yes, this caused a bug.
    ZIP_NAMES = ["MoNuSeg 2018 Training Data.zip", "MoNuSegTestData.zip"]

    # download
    zip_folder = dataset_root / "zips"

    if files_exist(zip_folder, ZIP_NAMES):
        print("No need to download...")
        zip_paths = [zip_folder / zip_name for zip_name in ZIP_NAMES]
    else:
        print("Downloading...")
        zip_paths = []
        downloader = GDownloader()
        for zip_src in ZIP_SOURCES:
            downloaded_zip_path = downloader.download(zip_src)
            shutil.move(downloaded_zip_path, zip_folder)
            zip_paths.append(zip_folder / downloaded_zip_path.name)

    # Unzip
    unzip_folder = dataset_root / "unzips"
    unzip_paths = [(unzip_folder / zip_path.stem) for zip_path in zip_paths]
    unzip_names = [zip_name[:-4] for zip_name in ZIP_NAMES]

    unzipper = Unzipper()
    if not files_exist(unzip_folder, unzip_names):
        print("Unzipping...")
        for zip_path in zip_paths:
            unzipper.unzip(zip_path, (unzip_folder / zip_path.stem))
    else:
        print("Unzipped datasets already present...")

    # move
    xmls = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/*.xml"))
    tifs = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/*.tif"))

    split_idx = -7
    train_xmls, train_tifs = xmls[:split_idx], tifs[:split_idx]
    val_xmls, val_tifs = xmls[split_idx:], tifs[split_idx:]

    original_paths = [
        (dataset_root / "MoNuSeg" / "original" / split)
        for split in ["train", "val", "test"]
    ]

    train_path = original_paths[0]
    for train_xml, train_tif in zip(train_xmls, train_tifs):
        copy_file_if_new(train_xml, train_path)
        copy_file_if_new(train_tif, train_path)

    val_path = original_paths[1]
    for val_xml, val_tif in zip(val_xmls, val_tifs):
        copy_file_if_new(val_xml, val_path)
        copy_file_if_new(val_tif, val_path)

    test_files = (unzip_paths[1] / unzip_paths[1].stem).glob("*")
    test_path = original_paths[2]
    for file in test_files:
        copy_file_if_new(file, test_path)

    # masks
    mask_paths = [
        (dataset_root / "MoNuSeg" / "masks" / split)
        for split in ["train", "val", "test"]
    ]

    xml_parser = MoNuSegXMLParser()

    for original_path, mask_path in zip(original_paths, mask_paths):
        patients = sorted(original_path.glob("*.xml"))
        pooldata = [(xml_parser, patient, mask_path) for patient in patients]

        with Pool() as pool:
            pool.starmap(generate_monuseg_mask, pooldata)

    # yolo
    yolo_paths = [
        (dataset_root / "MoNuSeg" / "yolo" / split)
        for split in ["train", "val", "test"]
    ]

    for original_path, yolo_path in zip(original_paths, yolo_paths):
        patients = sorted(original_path.glob("*.xml"))
        pooldata = [(xml_parser, patient, yolo_path) for patient in patients]

        with Pool() as pool:
            pool.starmap(yolofy_monuseg_file, pooldata)

    # yolo stain-normalised
    yolosn_paths = [
        (dataset_root / "MoNuSeg" / "yolo_sn" / split)
        for split in ["train", "val", "test"]
    ]
    fit_image = yolo_paths[0] / "TCGA-A7-A13E-01Z-00-DX1.png"
    stain_normaliser = StainNormaliser("reinhard", fit_image)

    # Manually chosen
    stain_normaliser.normalise(yolo_paths, yolosn_paths)


def get_monusac(dataset_root: Path):
    ZIP_SOURCES = [
        "1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq",
        "1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ",
    ]
    ZIP_NAMES = [
        "MoNuSAC_images_and_annotations.zip",
        "MoNuSAC Testing Data and Annotations.zip",
    ]

    # download
    zip_folder = dataset_root / "zips"

    if files_exist(zip_folder, ZIP_NAMES):
        print("No need to download...")
        zip_paths = [zip_folder / zip_name for zip_name in ZIP_NAMES]
    else:
        print("Downloading...")
        zip_paths = []
        downloader = GDownloader()
        for zip_src in ZIP_SOURCES:
            downloaded_zip_path = downloader.download(zip_src)
            shutil.move(downloaded_zip_path, zip_folder)
            zip_paths.append(zip_folder / downloaded_zip_path.name)

    # Unzip
    unzip_folder = dataset_root / "unzips"
    unzip_paths = [(unzip_folder / zip_path.stem) for zip_path in zip_paths]
    unzip_names = [zip_name[:-4] for zip_name in ZIP_NAMES]

    unzipper = Unzipper()
    if not files_exist(unzip_folder, unzip_names):
        print("Unzipping...")
        for zip_path in zip_paths:
            unzipper.unzip(zip_path, (unzip_folder / zip_path.stem))
    else:
        print("Unzipped datasets already present...")

    # move
    original_paths = [
        (dataset_root / "MoNuSAC" / "original" / split)
        for split in ["train", "val", "test"]
    ]

    patients = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("*"))

    split_idx = -11
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]
    test_patients = sorted((unzip_paths[1] / unzip_paths[1].stem).glob("*"))

    train_pooldata = [(patient, original_paths[0]) for patient in train_patients]
    val_pooldata = [(patient, original_paths[1]) for patient in val_patients]
    test_pooldata = [(patient, original_paths[2]) for patient in test_patients]
    with Pool() as pool:
        pool.starmap(copy_folder_if_new, train_pooldata)
        pool.starmap(copy_folder_if_new, val_pooldata)
        pool.starmap(copy_folder_if_new, test_pooldata)

    # masks
    mask_paths = [
        (dataset_root / "MoNuSAC" / "masks" / split)
        for split in ["train", "val", "test"]
    ]

    xml_parser = MoNuSACXMLParser()

    for original_path, mask_path in zip(original_paths, mask_paths):
        patients = sorted(original_path.glob("*.xml"))
        pooldata = [(xml_parser, patient, mask_path) for patient in patients]

        with Pool() as pool:
            pool.starmap(generate_monusac_mask, pooldata)

    # yolo
    yolo_paths = [
        (dataset_root / "MoNuSAC" / "yolo" / split)
        for split in ["train", "val", "test"]
    ]

    for original_path, yolo_path in zip(original_paths, yolo_paths):
        patients = sorted(original_path.glob("*.xml"))
        pooldata = [(xml_parser, patient, yolo_path) for patient in patients]

        with Pool() as pool:
            pool.starmap(yolofy_monusac_file, pooldata)

    # yolo stain-normalised
    yolosn_paths = [
        (dataset_root / "MoNuSAC" / "yolo_sn" / split)
        for split in ["train", "val", "test"]
    ]
    fit_image = yolo_paths[0] / "TCGA-A2-A0ES-01Z-00-DX1_2.png"
    stain_normaliser = StainNormaliser("reinhard", fit_image)

    # Manually chosen
    stain_normaliser.normalise(yolo_paths, yolosn_paths)


def get_tnbc(dataset_root: Path):
    ZIP_SOURCES = ["10.5281/zenodo.1175282"]
    ZIP_NAMES = ["TNBC_NucleiSegmentation.zip"]

    # download
    zip_folder = dataset_root / "zips"
    if files_exist(zip_folder, ZIP_NAMES):
        print("No need to download...")
        zip_paths = [zip_folder / zip_name for zip_name in ZIP_NAMES]
    else:
        print("Downloading...")
        zip_paths = []
        downloader = ZenodoDownloader()
        for zip_source, zip_name in zip(ZIP_SOURCES, ZIP_NAMES):
            downloader.download(zip_source)
            shutil.move(zip_name, zip_folder)
            zip_paths.append(zip_folder / zip_name)

    # Unzip
    unzip_folder = dataset_root / "unzips"
    unzip_paths = [(unzip_folder / zip_path.stem) for zip_path in zip_paths]
    unzip_names = [zip_name[:-4] for zip_name in ZIP_NAMES]

    unzipper = Unzipper()
    if not files_exist(unzip_folder, unzip_names):
        print("Unzipping...")
        for zip_path in zip_paths:
            unzipper.unzip(zip_path, (unzip_folder / zip_path.stem))
    else:
        print("Unzipped datasets already present...")

    # move
    original_paths = [
        (dataset_root / "TNBC" / "original" / split)
        for split in ["train", "val", "test"]
    ]

    mask_paths = [
        (dataset_root / "TNBC" / "masks" / split) for split in ["train", "val", "test"]
    ]

    folders = sorted(unzip_paths[0].glob("**/Slide_*"))
    train_folders = folders[:6]
    val_folders = [folders[i] for i in [6, 9, 10]]
    test_folders = [folders[i] for i in [7, 8]]

    for train_folder in train_folders:
        copy_folder_if_new(train_folder, original_paths[0])
    for val_folder in val_folders:
        copy_folder_if_new(val_folder, original_paths[1])
    for test_folder in test_folders:
        copy_folder_if_new(test_folder, original_paths[2])

    mask_folders = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/GT_*"))
    train_mask_folders = mask_folders[:6]
    val_mask_folders = [mask_folders[i] for i in [6, 9, 10]]
    test_mask_folders = [mask_folders[i] for i in [7, 8]]

    for train_mask_folder in train_mask_folders:
        copy_folder_if_new(train_mask_folder, mask_paths[0])
    for val_mask_folder in val_mask_folders:
        copy_folder_if_new(val_mask_folder, mask_paths[1])
    for test_mask_folder in test_mask_folders:
        copy_folder_if_new(test_mask_folder, mask_paths[2])

    # masks
    for original_path, mask_path in zip(original_paths, mask_paths):
        masks = sorted(mask_path.glob("*.png"))
        pooldata = [mask for mask in masks]

        with Pool() as pool:
            pool.map(generate_tnbc_mask, pooldata)

    # yolo
    yolo_paths = [
        (dataset_root / "TNBC" / "yolo" / split) for split in ["train", "val", "test"]
    ]

    for original_path, yolo_path, mask_path in zip(
        original_paths, yolo_paths, mask_paths
    ):
        patient_imgs = sorted(original_path.glob("*.png"))
        patient_masks = sorted(mask_path.glob("*.png"))
        pooldata = [
            (patient_img, patient_mask, yolo_path)
            for patient_img, patient_mask in zip(patient_imgs, patient_masks)
        ]

        with Pool() as pool:
            pool.starmap(yolofy_tnbc_file, pooldata)

    # yolo stain-normalised
    yolosn_paths = [
        (dataset_root / "TNBC" / "yolo_sn" / split)
        for split in ["train", "val", "test"]
    ]
    fit_image = yolo_paths[0] / "03_1.png"
    stain_normaliser = StainNormaliser("reinhard", fit_image)

    # Manually chosen
    stain_normaliser.normalise(yolo_paths, yolosn_paths)


def get_cryonuseg(dataset_root: Path):
    ZIP_SOURCES = ["ipateam/segmentation-of-nuclei-in-cryosectioned-he-images"]
    ZIP_NAMES = ["segmentation-of-nuclei-in-cryosectioned-he-images.zip"]

    # download
    zip_folder = dataset_root / "zips"
    if files_exist(zip_folder, ZIP_NAMES):
        print("No need to download...")
        zip_paths = [zip_folder / zip_name for zip_name in ZIP_NAMES]
    else:
        print("Downloading...")
        zip_paths = []
        downloader = KaggleDownloader()
        for zip_source, zip_name in zip(ZIP_SOURCES, ZIP_NAMES):
            downloader.download(zip_source)
            shutil.move(zip_name, zip_folder)
            zip_paths.append(zip_folder / zip_name)

    # Unzip
    unzip_folder = dataset_root / "unzips"
    unzip_paths = [(unzip_folder / zip_path.stem) for zip_path in zip_paths]
    unzip_names = [zip_name[:-4] for zip_name in ZIP_NAMES]

    unzipper = Unzipper()
    if not files_exist(unzip_folder, unzip_names):
        print("Unzipping...")
        for zip_path in zip_paths:
            unzipper.unzip(zip_path, (unzip_folder / zip_path.stem))
    else:
        print("Unzipped datasets already present...")




    # move
    original_paths = [
        (dataset_root / "CryoNuSeg" / "original" / split)
        for split in ["train", "val", "test"]
    ]

    mask_paths = [
        (dataset_root / "CryoNuSeg" / "masks" / split) for split in ["train", "val", "test"]
    ]

    train_tifs = sorted(unzip_paths[0].glob("tissue images/*[12].tif"))
    val_test_tifs = sorted(unzip_paths[0].glob("tissue images/*3.tif"))
    val_tifs = val_test_tifs[:5]
    test_tifs = val_test_tifs[5:]

    for train_tif in train_tifs:
        copy_file_if_new(train_tif, original_paths[0])
    for val_tif in val_tifs:
        copy_file_if_new(val_tif, original_paths[1])
    for test_tif in test_tifs:
        copy_file_if_new(test_tif, original_paths[2])

    train_mask_tifs = sorted(
        unzip_paths[0].glob("Annotator 2*/**/mask binary without border/*[12].png")
    )
    val_test_mask_tifs = sorted(
        unzip_paths[0].glob("Annotator 2*/**/mask binary without border/*3.png")
    )
    val_mask_tifs = val_test_mask_tifs[:5]
    test_mask_tifs = val_test_mask_tifs[5:]

    for train_mask_tif in train_mask_tifs:
        copy_file_if_new(train_mask_tif, mask_paths[0])
    for val_mask_tif in val_mask_tifs:
        copy_file_if_new(val_mask_tif, mask_paths[1])
    for test_mask_tif in test_mask_tifs:
        copy_file_if_new(test_mask_tif, mask_paths[2])

    # masks
    for original_path, mask_path in zip(original_paths, mask_paths):
        masks = sorted(mask_path.glob("*.png"))
        pooldata = [mask for mask in masks]

        with Pool() as pool:
            pool.map(generate_cryonuseg_mask, pooldata)

    def yolofy_files(
        self,
        original_paths: list[Path],
        yolofy_paths: list[Path],
        mask_paths: list[Path],
    ):
        for original_path, yolofy_path, mask_path in zip(
            original_paths, yolofy_paths, mask_paths
        ):
            pooldata = self.get_pooldata(original_path, yolofy_path, mask_path)

            with Pool() as pool:
                for _ in tqdm(
                    pool.istarmap(self.yolofy_file, pooldata), total=len(pooldata)
                ):
                    pass


    # yolo
    yolo_paths = [
        (dataset_root / "CryoNuSeg" / "yolo" / split) for split in ["train", "val", "test"]
    ]

    for original_path, yolo_path, mask_path in zip(
        original_paths, yolo_paths, mask_paths
    ):
        patient_imgs = sorted(original_path.glob("*.tif"))
        patient_masks = sorted(mask_path.glob("*.png"))
        pooldata = [
            (patient_img, patient_mask, yolo_path)
            for patient_img, patient_mask in zip(patient_imgs, patient_masks)
        ]

        with Pool() as pool:
            pool.starmap(yolofy_cryonuseg_file, pooldata)

    # yolo stain-normalised
    yolosn_paths = [
        (dataset_root / "CryoNuSeg" / "yolo_sn" / split)
        for split in ["train", "val", "test"]
    ]
    fit_image = yolo_paths[0] / "Human_Larynx_02.png"
    stain_normaliser = StainNormaliser("reinhard", fit_image)

    # Manually chosen
    stain_normaliser.normalise(yolo_paths, yolosn_paths)
