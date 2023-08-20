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


class Dataset(ABC):
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root

    @property
    @abstractmethod
    def dataset_name(self) -> str:
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
    def original_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.dataset_name / "original" / "train",
            self.dataset_root / self.dataset_name / "original" / "val",
            self.dataset_root / self.dataset_name / "original" / "test",
        ]

    @property
    def mask_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.dataset_name / "masks" / "train",
            self.dataset_root / self.dataset_name / "masks" / "val",
            self.dataset_root / self.dataset_name / "masks" / "test",
        ]

    @property
    def yolo_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.dataset_name / "yolo" / "train",
            self.dataset_root / self.dataset_name / "yolo" / "val",
            self.dataset_root / self.dataset_name / "yolo" / "test",
        ]

    @property
    def yolosn_paths(self) -> list[Path]:
        return [
            self.dataset_root / self.dataset_name / "yolo_sn" / "train",
            self.dataset_root / self.dataset_name / "yolo_sn" / "val",
            self.dataset_root / self.dataset_name / "yolo_sn" / "test",
        ]


class MoNuSegDataset(Dataset):
    def __init__(self, dataset_root: Path):
        super().__init__(dataset_root)

    @property
    def dataset_name(self) -> str:
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

    def organise(self, unzip_paths: list[Path]):
        # move
        xmls = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/*.xml"))
        tifs = sorted((unzip_paths[0] / unzip_paths[0].stem).glob("**/*.tif"))

        split_idx = -7
        train_xmls, train_tifs = xmls[:split_idx], tifs[:split_idx]
        val_xmls, val_tifs = xmls[split_idx:], tifs[split_idx:]

        train_path = self.original_paths[0]
        for train_xml, train_tif in zip(train_xmls, train_tifs):
            copy_file_if_new(train_xml, train_path)
            copy_file_if_new(train_tif, train_path)

        val_path = self.original_paths[1]
        for val_xml, val_tif in zip(val_xmls, val_tifs):
            copy_file_if_new(val_xml, val_path)
            copy_file_if_new(val_tif, val_path)

        test_files = (unzip_paths[1] / unzip_paths[1].stem).glob("*")
        test_path = self.original_paths[2]
        for file in test_files:
            copy_file_if_new(file, test_path)

    def yolofy_files(self):
        xml_parser = MoNuSegXMLParser()
        for original_path, yolo_path in zip(self.original_paths, self.yolo_paths):
            patients = sorted(original_path.glob("*.xml"))
            pooldata = [(xml_parser, patient, yolo_path) for patient in patients]

            with Pool() as pool:
                pool.starmap(self._yolofy_one_file, pooldata)

    def _yolofy_one_file(
        self, xml_parser: XMLParser, patient_xml: Path, yolofy_path: Path
    ):
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

    def generate_masks(self):
        xml_parser = MoNuSegXMLParser()

        for original_path, mask_path in zip(self.original_paths, self.mask_paths):
            patients = sorted(original_path.glob("*.xml"))
            pooldata = [(xml_parser, patient, mask_path) for patient in patients]

            with Pool() as pool:
                pool.starmap(self._generate_monuseg_mask, pooldata)

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
    def dataset_name(self) -> str:
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

    def organise(self, unzip_paths: Path):
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

    def yolofy_files(self):
        xml_parser = MoNuSACXMLParser()

        for original_path, yolo_path in zip(self.original_paths, self.yolo_paths):
            patients = sorted(original_path.glob("*.xml"))
            pooldata = [(xml_parser, patient, yolo_path) for patient in patients]

            with Pool() as pool:
                pool.starmap(self.yolofy_one_file, pooldata)

    def _yolofy_one_file(
        self, xml_parser: XMLParser, patient_xml: Path, yolofy_path: Path
    ):
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

    def generate_masks(self):
        xml_parser = MoNuSACXMLParser()

        for original_path, mask_path in zip(self.original_paths, self.mask_paths):
            patients = sorted(original_path.glob("*.xml"))
            pooldata = [(xml_parser, patient, mask_path) for patient in patients]

            with Pool() as pool:
                pool.starmap(self._generate_monuseg_mask, pooldata)

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


class CryoNuSegDataset(Dataset):
    def __init__(self, dataset_root: Path):
        super().__init__(dataset_root)

    @property
    def dataset_name(self) -> str:
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


class TNBCDataset(Dataset):
    def __init__(self, dataset_root: Path):
        super().__init__(dataset_root)

    @property
    def dataset_name(self) -> str:
        return "TNBC"

    @property
    def zip_sources(self) -> list[str]:
        return ["10.5281/zenodo.1175282"]

    @property
    def zip_names(self) -> list[str]:
        return ["TNBC_NucleiSegmentation.zip"]

    @property
    def fit_image(self) -> Path:
        return self.yolofy_paths[0] / "03_1.png"


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


def convert_mask_value(mask: Path, original=255, new=1):
    img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    img[img == original] = new
    cv2.imwrite(str(mask), img)


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
        if directory_empty:
            return True
    return False


def get_dataset(dataset_root: Path, dataset_name: str):
    DATASETS = {
        "MoNuSeg": MoNuSegDataset,
        "MoNuSAC": MoNuSACDataset,
        "TNBC": TNBCDataset,
        "CryoNuSeg": CryoNuSegDataset,
    }

    dataset = DATASETS[dataset_name](dataset_root)

    zip_folder = dataset_root / "zips"
    unzip_folder = dataset_root / "unzips"

    if files_exist(zip_folder, dataset.zip_names):
        print(f"No need to download {dataset.name}...")
    else:
        print(f"Downloading {dataset.name}...")
        for zip_name, zip_source in zip(dataset.zip_sources, dataset.zip_names):
            if dataset.download():
                shutil.move(zip_name, zip_folder)

    zip_paths = [zip_folder / zip_name for zip_name in dataset.zip_names]

    unzipper = Unzipper()
    if files_exist(unzip_folder, dataset.unzip_names):
        print(f"No need to unzip {dataset.name}...")
    else:
        print(f"Unzipping {dataset.name}...")
        for zip_path in zip_paths:
            unzipper.unzip(zip_path, (unzip_folder / zip_path.stem))

    unzip_paths = [(unzip_folder / zip_path.stem) for zip_path in zip_paths]

    if not missing_files(dataset.original_paths):
        print(f"No need to organise {dataset.name}...")
    else:
        print(f"Organising {dataset.name}...")
        dataset.organise(unzip_paths)

    if not missing_files(dataset.mask_paths):
        print(f"No need to generate masks for {dataset.name}...")
    else:
        print(f"Generating masks for {dataset.name}...")
        dataset.generate_masks()

    if not missing_files(dataset.yolo_paths):
        print(f"No need to yolofy {dataset.name}...")
    else:
        print(f"Yolofying {dataset.name}...")
        dataset.yolofy()

    if not missing_files(dataset.yolosn_paths):
        print(f"No need to stain normalise {dataset.name}...")
    else:
        # fit_image = yolo_paths[0] / "TCGA-A7-A13E-01Z-00-DX1.png"
        print(f"Stain normalising {dataset.name}...")
        stain_normaliser = StainNormaliser("reinhard", dataset.fit_image)
        stain_normaliser.normalise(dataset.yolo_paths, dataset.yolosn_paths)


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

    # masks
    mask_paths = [
        (dataset_root / "MoNuSAC" / "masks" / split)
        for split in ["train", "val", "test"]
    ]

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
        (dataset_root / "CryoNuSeg" / "masks" / split)
        for split in ["train", "val", "test"]
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
        (dataset_root / "CryoNuSeg" / "yolo" / split)
        for split in ["train", "val", "test"]
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
