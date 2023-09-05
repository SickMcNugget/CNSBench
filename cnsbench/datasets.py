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


class Downloader(ABC):
    @abstractmethod
    def download(self, zip_source: str, *args) -> bool:
        """Downloads a zip file from it's online id and returns the downloaded path"""


class GDownloader(Downloader):
    def __init__(self) -> None:
        super().__init__()

    def download(self, zip_source: str) -> bool:
        zip_path = gdown.download(id=zip_source)

        if zip_path is None:
            return False

        return True


class ZenodoDownloader(Downloader):
    def __init__(self) -> None:
        super().__init__()

    def download(self, zip_source: str):
        try:
            ret = subprocess.run(["zenodo_get", zip_source], stderr=subprocess.DEVNULL)
            ret.check_returncode()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

        return True


class KaggleDownloader(Downloader):
    def __init__(self) -> None:
        super().__init__()

    def download(self, zip_source: str) -> bool:
        try:
            ret = subprocess.run(
                ["kaggle", "datasets", "download", "-d", zip_source],
                stderr=subprocess.DEVNULL,
            )
            ret.check_returncode()
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(
                "Please configure ~/.kaggle/kaggle.json. Follow the instructions"
                + " on their website."
            )
            return False

        return True


class Unzipper:
    def __init__(self) -> None:
        pass

    def unzip(self, zip_path: Path, unzip_folder: Path) -> Path:
        with ZipFile(zip_path, "r") as zfile:
            zfile.extractall(unzip_folder)

        return unzip_folder / zip_path.stem


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


class StainExtractor:
    def __init__(self, fit_image: Path) -> None:
        super().__init__()

        self.fit_image = fit_image
        self.T = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
        )

        self.normaliser = torchstain.normalizers.MacenkoNormalizer(backend="torch")

        fit_img = cv2.cvtColor(cv2.imread(str(fit_image)), cv2.COLOR_BGR2RGB)
        self.normaliser.fit(self.T(fit_img))

    def extract(self, yolo_paths: list[Path], stain_paths: list[Path]):
        for yolo_path, stain_path in zip(yolo_paths, stain_paths):
            pngs = sorted(yolo_path.glob("*.png"))
            for png in pngs:
                to_normalise = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
                _, H, E = self.normaliser.normalize(I=self.T(to_normalise), stains=True)

                H = H.numpy().astype(np.uint8)
                E = E.numpy().astype(np.uint8)
                H = cv2.cvtColor(H, cv2.COLOR_RGB2BGR)
                E = cv2.cvtColor(E, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(stain_path / f"{png.stem}_H.png"), H)
                cv2.imwrite(str(stain_path / f"{png.stem}_E.png"), E)


class Dataset(ABC):
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root

    @property
    @abstractmethod
    def name(self) -> str:
        """This should simply be the dataset name"""

    @property
    @abstractmethod
    def zip_sources(self) -> list[str]:
        """These should be the identifiers for downloading zip files"""

    @property
    @abstractmethod
    def zip_names(self) -> list[str]:
        """These should be the expected names for downloaded zip files"""

    @property
    @abstractmethod
    def fit_image(self) -> Path:
        """This should be the path to the image which the stain normaliser is fit on"""

    @property
    def unzip_names(self) -> list[str]:
        return [Path(zip_name).stem for zip_name in self.zip_names]

    @property
    def original_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.name / "original" / "train",
            self.dataset_root / self.name / "original" / "val",
            self.dataset_root / self.name / "original" / "test",
        ]

    @property
    def mask_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.name / "masks" / "train",
            self.dataset_root / self.name / "masks" / "val",
            self.dataset_root / self.name / "masks" / "test",
        ]

    @property
    def yolo_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.name / "yolo" / "train",
            self.dataset_root / self.name / "yolo" / "val",
            self.dataset_root / self.name / "yolo" / "test",
        ]

    @property
    def yolosn_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.name / "yolo_sn" / "train",
            self.dataset_root / self.name / "yolo_sn" / "val",
            self.dataset_root / self.name / "yolo_sn" / "test",
        ]

    @property
    def stain_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.name / "stains" / "train",
            self.dataset_root / self.name / "stains" / "val",
            self.dataset_root / self.name / "stains" / "test",
        ]

    @abstractmethod
    def download(self, zip_source) -> bool:
        """Downloads the dataset and returns a boolean status."""

    @abstractmethod
    def organise(self, unzip_paths: list[Path]):
        """Collects files from the unzipped dataset and organises them for use"""

    @abstractmethod
    def yolofy(self):
        """Creates a directory containing .txt and .png files for YOLOv8 training"""

    @abstractmethod
    def generate_masks(self):
        """Creates binary ([0,1]) masks for every annotation in the dataset"""


class MoNuSegDataset(Dataset):
    def __init__(self, dataset_root: Path):
        super().__init__(dataset_root)

    @property
    def name(self) -> str:
        return "MoNuSeg"

    @property
    def zip_sources(self) -> list[str]:
        return [
            "1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA",
            "1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw",
        ]

    @property
    def zip_names(self) -> list[str]:
        return ["MoNuSeg 2018 Training Data.zip", "MoNuSegTestData.zip"]

    @property
    def fit_image(self) -> Path:
        return self.yolo_paths[0] / "TCGA-A7-A13E-01Z-00-DX1.png"

    def download(self, zip_source):
        downloader = GDownloader()
        return downloader.download(zip_source)

    def organise(self, unzip_paths: list[Path]):
        # move
        xmls = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/*.xml"))
        tifs = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/*.tif"))

        split_idx = -7
        train_xmls, train_tifs = xmls[:split_idx], tifs[:split_idx]
        val_xmls, val_tifs = xmls[split_idx:], tifs[split_idx:]
        test_files = (unzip_paths[1] / unzip_paths[1].stem).glob("*")

        train_path = self.original_paths[0]
        for train_xml, train_tif in zip(train_xmls, train_tifs):
            copy_file_if_new(train_xml, train_path)
            copy_file_if_new(train_tif, train_path)

        val_path = self.original_paths[1]
        for val_xml, val_tif in zip(val_xmls, val_tifs):
            copy_file_if_new(val_xml, val_path)
            copy_file_if_new(val_tif, val_path)

        test_path = self.original_paths[2]
        for file in test_files:
            copy_file_if_new(file, test_path)

    def yolofy(self):
        xml_parser = MoNuSegXMLParser()
        for original_path, yolo_path in zip(self.original_paths, self.yolo_paths):
            patients = sorted(original_path.glob("*.xml"))
            pooldata = [(xml_parser, patient, yolo_path) for patient in patients]

            with Pool() as pool:
                pool.starmap(self._yolofy_one_file, pooldata)

    def _yolofy_one_file(
        self, xml_parser: XMLParser, patient_xml: Path, yolo_path: Path
    ):
        patient_image = patient_xml.with_suffix(".tif")

        if not patient_image.exists():
            return

        # YOLOv8 requires .png images
        patient_img = cv2.imread(str(patient_image))
        filename = str((yolo_path / patient_image.stem).with_suffix(".png"))
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

        filename = (yolo_path / patient_image.stem).with_suffix(".txt")
        with open(filename, "w") as f:
            f.write(annotations)

    def generate_masks(self):
        xml_parser = MoNuSegXMLParser()

        for original_path, mask_path in zip(self.original_paths, self.mask_paths):
            patients = sorted(original_path.glob("*.xml"))
            pooldata = [(xml_parser, patient, mask_path) for patient in patients]

            with Pool() as pool:
                pool.starmap(self._generate_one_mask, pooldata)

    def _generate_one_mask(self, xml_parser: XMLParser, patient: Path, mask_path: Path):
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


class MoNuSACDataset(Dataset):
    def __init__(self, dataset_root: Path):
        super().__init__(dataset_root)

    @property
    def name(self) -> str:
        return "MoNuSAC"

    @property
    def zip_sources(self) -> list[str]:
        return [
            "1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq",
            "1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ",
        ]

    @property
    def zip_names(self) -> list[str]:
        return [
            "MoNuSAC_images_and_annotations.zip",
            "MoNuSAC Testing Data and Annotations.zip",
        ]

    @property
    def fit_image(self) -> Path:
        return self.yolo_paths[0] / "TCGA-A2-A0ES-01Z-00-DX1_2.png"

    def download(self, zip_source):
        downloader = GDownloader()
        return downloader.download(zip_source)

    def organise(self, unzip_paths: list[Path]):
        patients = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("*"))

        split_idx = -11
        train_patients = patients[:split_idx]
        val_patients = patients[split_idx:]
        test_patients = sorted((unzip_paths[1] / unzip_paths[1].stem).glob("*"))

        train_pooldata = [
            (patient, self.original_paths[0]) for patient in train_patients
        ]
        val_pooldata = [(patient, self.original_paths[1]) for patient in val_patients]
        test_pooldata = [(patient, self.original_paths[2]) for patient in test_patients]
        with Pool() as pool:
            pool.starmap(copy_folder_if_new, train_pooldata)
            pool.starmap(copy_folder_if_new, val_pooldata)
            pool.starmap(copy_folder_if_new, test_pooldata)

    def yolofy(self):
        xml_parser = MoNuSACXMLParser()

        for original_path, yolo_path in zip(self.original_paths, self.yolo_paths):
            patients = sorted(original_path.glob("*.xml"))
            pooldata = [(xml_parser, patient, yolo_path) for patient in patients]

            with Pool() as pool:
                pool.starmap(self._yolofy_one_file, pooldata)

    def _yolofy_one_file(
        self, xml_parser: XMLParser, patient_xml: Path, yolo_path: Path
    ):
        patient_image = patient_xml.with_suffix(".svs")

        if not patient_image.exists():
            return

        slide = OpenSlide(str(patient_image))
        filename = str((yolo_path / patient_image.stem).with_suffix(".png"))
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

        filename = (yolo_path / patient_image.stem).with_suffix(".txt")
        with open(filename, "w") as f:
            f.write(annotations)

    def generate_masks(self):
        xml_parser = MoNuSACXMLParser()

        for original_path, mask_path in zip(self.original_paths, self.mask_paths):
            patients = sorted(original_path.glob("*.xml"))
            pooldata = [(xml_parser, patient, mask_path) for patient in patients]

            with Pool() as pool:
                pool.starmap(self._generate_one_mask, pooldata)

    def _generate_one_mask(self, xml_parser: XMLParser, patient: Path, mask_path: Path):
        patient_image = patient.with_suffix(".svs")
        if not patient_image.exists():
            return

        slide = OpenSlide(str(patient_image))
        filename = str((mask_path / patient_image.stem).with_suffix(".png"))
        slide.get_thumbnail(slide.level_dimensions[0]).save(filename)
        width, height = slide.level_dimensions[0]
        slide.close()

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


class CryoNuSegDataset(Dataset):
    def __init__(self, dataset_root: Path):
        super().__init__(dataset_root)

    @property
    def name(self) -> str:
        return "CryoNuSeg"

    @property
    def zip_sources(self) -> list[str]:
        return ["ipateam/segmentation-of-nuclei-in-cryosectioned-he-images"]

    @property
    def zip_names(self) -> list[str]:
        return ["segmentation-of-nuclei-in-cryosectioned-he-images.zip"]

    @property
    def fit_image(self) -> Path:
        return self.yolo_paths[0] / "Human_Larynx_02.png"

    def download(self, zip_source):
        downloader = KaggleDownloader()
        return downloader.download(zip_source)

    def organise(self, unzip_paths: list[Path]):
        train_tifs = sorted(unzip_paths[0].glob("tissue images/*[12].tif"))
        val_test_tifs = sorted(unzip_paths[0].glob("tissue images/*3.tif"))
        val_tifs = val_test_tifs[:5]
        test_tifs = val_test_tifs[5:]

        for train_tif in train_tifs:
            copy_file_if_new(train_tif, self.original_paths[0])
        for val_tif in val_tifs:
            copy_file_if_new(val_tif, self.original_paths[1])
        for test_tif in test_tifs:
            copy_file_if_new(test_tif, self.original_paths[2])

        train_mask_tifs = sorted(
            unzip_paths[0].glob("Annotator 2*/**/mask binary without border/*[12].png")
        )
        val_test_mask_tifs = sorted(
            unzip_paths[0].glob("Annotator 2*/**/mask binary without border/*3.png")
        )
        val_mask_tifs = val_test_mask_tifs[:5]
        test_mask_tifs = val_test_mask_tifs[5:]

        for train_mask_tif in train_mask_tifs:
            copy_file_if_new(train_mask_tif, self.mask_paths[0])
        for val_mask_tif in val_mask_tifs:
            copy_file_if_new(val_mask_tif, self.mask_paths[1])
        for test_mask_tif in test_mask_tifs:
            copy_file_if_new(test_mask_tif, self.mask_paths[2])

    def yolofy(self):
        for original_path, yolo_path, mask_path in zip(
            self.original_paths, self.yolo_paths, self.mask_paths
        ):
            patient_imgs = sorted(original_path.glob("*.tif"))
            patient_masks = sorted(mask_path.glob("*.png"))
            pooldata = [
                (patient_img, patient_mask, yolo_path)
                for patient_img, patient_mask in zip(patient_imgs, patient_masks)
            ]

            with Pool() as pool:
                pool.starmap(self._yolofy_one_file, pooldata)

    def _yolofy_one_file(
        self, patient_img: Path, patient_mask: Path, yolofy_path: Path
    ):
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
            annotations += (
                f" {normalised_nucleus[0, 0, 0]} {normalised_nucleus[0, 0, 1]}\n"
            )

        filename = yolofy_path / patient_mask.with_suffix(".txt").name
        with open(filename, "w") as f:
            f.write(annotations)

    def generate_masks(self):
        for original_path, mask_path in zip(self.original_paths, self.mask_paths):
            masks = sorted(mask_path.glob("*.png"))
            pooldata = [mask for mask in masks]

            with Pool() as pool:
                pool.map(convert_mask_value, pooldata)


class TNBCDataset(Dataset):
    def __init__(self, dataset_root: Path):
        super().__init__(dataset_root)

    @property
    def name(self) -> str:
        return "TNBC"

    @property
    def zip_sources(self) -> list[str]:
        return ["10.5281/zenodo.1175282"]

    @property
    def zip_names(self) -> list[str]:
        return ["TNBC_NucleiSegmentation.zip"]

    @property
    def fit_image(self) -> Path:
        return self.yolo_paths[0] / "03_1.png"

    def download(self, zip_source):
        downloader = ZenodoDownloader()
        return downloader.download(zip_source)

    def organise(self, unzip_paths: list[Path]):
        folders = sorted(unzip_paths[0].glob("**/Slide_*"))
        train_folders = folders[:6]
        val_folders = [folders[i] for i in [6, 9, 10]]
        test_folders = [folders[i] for i in [7, 8]]

        for train_folder in train_folders:
            copy_folder_if_new(train_folder, self.original_paths[0])
        for val_folder in val_folders:
            copy_folder_if_new(val_folder, self.original_paths[1])
        for test_folder in test_folders:
            copy_folder_if_new(test_folder, self.original_paths[2])

        mask_folders = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/GT_*"))
        train_mask_folders = mask_folders[:6]
        val_mask_folders = [mask_folders[i] for i in [6, 9, 10]]
        test_mask_folders = [mask_folders[i] for i in [7, 8]]

        for train_mask_folder in train_mask_folders:
            copy_folder_if_new(train_mask_folder, self.mask_paths[0])
        for val_mask_folder in val_mask_folders:
            copy_folder_if_new(val_mask_folder, self.mask_paths[1])
        for test_mask_folder in test_mask_folders:
            copy_folder_if_new(test_mask_folder, self.mask_paths[2])

    def yolofy(self):
        for original_path, yolo_path, mask_path in zip(
            self.original_paths, self.yolo_paths, self.mask_paths
        ):
            patient_imgs = sorted(original_path.glob("*.png"))
            patient_masks = sorted(mask_path.glob("*.png"))
            pooldata = [
                (patient_img, patient_mask, yolo_path)
                for patient_img, patient_mask in zip(patient_imgs, patient_masks)
            ]

            with Pool() as pool:
                pool.starmap(self._yolofy_one_file, pooldata)

    def _yolofy_one_file(self, patient_img: Path, patient_mask: Path, yolo_path: Path):
        # original image is already a PNG
        shutil.copy(patient_img, (yolo_path / patient_img.name))

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
            annotations += (
                f" {normalised_nucleus[0, 0, 0]} {normalised_nucleus[0, 0, 1]}\n"
            )

        filename = yolo_path / patient_mask.with_suffix(".txt").name
        with open(filename, "w") as f:
            f.write(annotations)

    def generate_masks(self):
        for original_path, mask_path in zip(self.original_paths, self.mask_paths):
            masks = sorted(mask_path.glob("*.png"))
            pooldata = [mask for mask in masks]

            with Pool() as pool:
                pool.map(convert_mask_value, pooldata)

    def _generate_one_mask(self, mask: Path):
        img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        img[img == 255] = 1
        cv2.imwrite(str(mask), img)


def copy_folder_if_new(folder: Path, dest: Path):
    for image in folder.iterdir():
        copy_file_if_new(image, dest)


def copy_file_if_new(src: Path, dest: Path):
    if not (dest / src.name).exists():
        shutil.copy(src, dest)


def make_folder_if_new(path: Path):
    if not path.exists():
        path.mkdir(parents=True)


def make_directories(dataset_root: Path, name: str):
    base_directory = dataset_root / name / "original"
    masks_directory = dataset_root / name / "masks"
    yolo_directory = dataset_root / name / "yolo"
    yolo_sn_directory = dataset_root / name / "yolo_sn"
    stains_directory = dataset_root / name / "stains"

    for path in {"train", "val", "test"}:
        make_folder_if_new((base_directory / path))
        make_folder_if_new((masks_directory / path))
        make_folder_if_new((yolo_directory / path))
        make_folder_if_new((yolo_sn_directory / path))
        make_folder_if_new((stains_directory / path))

    make_folder_if_new((dataset_root / "zips"))
    make_folder_if_new((dataset_root / "unzips"))


def convert_mask_value(mask: Path, original=255, new=1):
    img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    img[img == original] = new
    cv2.imwrite(str(mask), img)


def files_exist(folder: Path, file_names: set[str]) -> bool:
    for file in file_names:
        if not (folder / file).exists():
            return False
    return True


def directory_empty(folder: Path) -> bool:
    if not any(folder.iterdir()):
        return True
    return False


def missing_files(folders: list[Path]) -> bool:
    for folder in folders:
        if directory_empty(folder):
            return True
    return False


def get_dataset_type(dataset_name: str):
    DATASETS = {
        "MoNuSeg": MoNuSegDataset,
        "MoNuSAC": MoNuSACDataset,
        "TNBC": TNBCDataset,
        "CryoNuSeg": CryoNuSegDataset,
    }

    return DATASETS[dataset_name]
