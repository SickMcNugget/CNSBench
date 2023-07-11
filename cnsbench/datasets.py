from __future__ import annotations

#builtins
from abc import ABC, abstractmethod
from pathlib import Path
from zipfile import ZipFile
import shutil
import multiprocessing.pool as mpp
from multiprocessing import Pool

# Linalg, parsing, image manip
import xml.etree.ElementTree as ET
from shapely import Polygon
from skimage.draw import polygon
import numpy as np
import cv2
import imagesize

# Progress
from tqdm import tqdm

# Dataset downloading
import gdown

# This allows for a loading bar whilst using multiprocessing 
# (Not my code, https://stackoverflow.com/a/57364423)
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap


class Downloader(ABC):
    def __init__(self, zip_ids: list[str]):
        self.zip_ids: list[str] = zip_ids
        self.zip_paths: list[Path] = []
        self.prefix: Path = Path("zips")
        self.expected_zip_names = []

    @property
    def prefix(self):
        return self._prefix
    
    @prefix.setter
    def prefix(self, value):
        self._prefix = value
        if not self._prefix.exists():
            self._prefix.mkdir(parents=True)

    @property
    def zip_ids(self):
        return self._zip_ids

    @zip_ids.setter
    def zip_ids(self, value):
        self._zip_ids = value

    @property
    def zip_paths(self):
        return self._zip_paths
    
    @zip_paths.setter
    def zip_paths(self, value):
        self._zip_paths = value    

    def download(self) -> list[Path]:
        if not self._should_download():
            print("Dataset already downloaded, skipping...")
            self.zip_paths = [self.prefix / zip_name for zip_name in self.expected_zip_names]
            return self.zip_paths

        zip_paths = []
        for zip_id in self.zip_ids:
            zip_paths.append(self._download(zip_id))

        self.zip_paths = [self.prefix / zip_path.name for zip_path in zip_paths]

        # All zips need to be moved to an appropriate location, so this is relevant
        for src, dest in zip(zip_paths, self.zip_paths):
            shutil.move(src, dest)

        return self.zip_paths

    @abstractmethod
    def _should_download(self) -> bool:
        """Check if zips already exist or need to be downloaded"""

    @abstractmethod
    def _download(self, zip_id: str) -> Path:
        """Downloads a zip file from it's online id and returns the downloaded path"""

class MoNuSegDownloader(Downloader):
    def __init__(self, zip_ids: list[str] = ["1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA", 
                                             "1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw"]):
        super().__init__(zip_ids)
        self.expected_zip_names = ["MoNuSeg 2018 Training Data.zip",
                                   "MoNuSegTestData.zip"]

    def _download(self, zip_id: str) -> Path:
        zip_path = gdown.download(id=zip_id)
        return Path(zip_path)
    
    def _should_download(self) -> bool:
        for zip_name in self.expected_zip_names:
            if not (self.prefix / zip_name).exists():
                return True
        return False

class MoNuSACDownloader(Downloader):
    def __init__(self, zip_ids: list[str] = ["1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq", 
                                             "1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ"]):
        super().__init__(zip_ids)
        self.expected_zip_names = ["MoNuSAC_images_and_annotations.zip",
                                   "MoNuSAC Testing Data and Annotations.zip"]

    def _download(self, zip_id: str) -> Path:
        zip_path = gdown.download(id=zip_id)
        return Path(zip_path)
    
    def _should_download(self) -> bool:
        for zip_name in self.expected_zip_names:
            if not (self.prefix / zip_name).exists():
                return True
        return False

class Unzipper:
    def __init__(self, zip_paths: list[Path]):
        self.zip_paths: list[Path] = zip_paths
        self.prefix: Path = Path("unzips")

    @property
    def prefix(self):
        return self._prefix
    
    @prefix.setter
    def prefix(self, value: Path):
        self._prefix: Path = value
        if not self._prefix.exists():
            self._prefix.mkdir(parents=True)

    @property
    def zip_paths(self) -> list[Path]:
        return self._zip_paths

    @zip_paths.setter
    def zip_paths(self, value):
        self._zip_paths = value

    @property
    def unzip_paths(self) -> list[Path]:
        unzip_paths = [self.prefix / zip_path.stem for zip_path in self.zip_paths]
        return unzip_paths
    
    def unzip(self) -> list[Path]:
        for zip_path, unzip_path in zip(self.zip_paths, self.unzip_paths):
            with ZipFile(zip_path, "r") as zfile:
                zfile.extractall(unzip_path)

        return self.unzip_paths
    
class Mover(ABC):
    def __init__(self, dataset_root: str | Path, unzip_paths: list[Path]):
        if isinstance(dataset_root, str):
            dataset_root = Path(dataset_root)
        
        dataset_name = self.__class__.__name__.split("Mover")[0]
        self.dataset_root = dataset_root / dataset_name
        self.unzip_paths = unzip_paths

    @property
    def dataset_root(self):
        return self._dataset_root
    
    @dataset_root.setter
    def dataset_root(self, value: Path):
        self._dataset_root: Path = value
        if not self._dataset_root.exists():
            self._dataset_root.mkdir(parents=True)

    def move_if_new(self, src: Path, dest: Path):
        if not (dest / src.name).exists():
            shutil.move(src, dest)

    def should_move(self, original_paths: list[Path]):
        for path in original_paths:
            if not path.exists() or not any(path.iterdir()):
                return True
        return False

    def move_all(self):
        original_paths = [self.dataset_root / "original" / "train",
                          self.dataset_root / "original" / "val",
                          self.dataset_root / "original" / "test"]

        if not self.should_move(original_paths):
            print("Original data already present, skipping...")
            return
        
        for path in original_paths:
            if not path.exists():
                path.mkdir(parents=True)

        self.move_train()
        self.move_valid()
        self.move_test()

    @abstractmethod
    def move_train(self):
        """Move the dataset's training data to its new directory"""

    @abstractmethod
    def move_valid(self):
        """Move the dataset's validation data to its new directory"""

    @abstractmethod
    def move_test(self):
        """Move the dataset's testing data to its new directory"""

class MoNuSegMover(Mover):
    def __init__(self, dataset_root: str | Path, unzip_paths: list[Path]):
        super().__init__(dataset_root, unzip_paths)
        self.expected_zips = {
            "train": "MoNuSeg 2018 Training Data",
            "val": "MoNuSeg 2018 Training Data",
            "test": "MoNuSegTestData"
        }

    def move_train(self):
        train_path = self.dataset_root / "original" / "train"

        for unzip_path in self.unzip_paths:
            if self.expected_zips["train"] in str(unzip_path):
                xmls = sorted((unzip_path / unzip_path.stem).glob("**/*.xml"))
                tifs = sorted((unzip_path / unzip_path.stem).glob("**/*.tif"))
                train_xmls = xmls[:-7]
                train_tifs = tifs[:-7]

                for xml, tif in zip(train_xmls, train_tifs):
                    self.move_if_new(xml, train_path)
                    self.move_if_new(tif, train_path)

    def move_valid(self):
        val_path = self.dataset_root / "original" / "val"
        
        for unzip_path in self.unzip_paths:
            if self.expected_zips["val"] in str(unzip_path):
                xmls = sorted((unzip_path / unzip_path.stem).glob("**/*.xml"))
                tifs = sorted((unzip_path / unzip_path.stem).glob("**/*.tif"))
                val_xmls = xmls[-7:]
                val_tifs = tifs[-7:]

                for xml, tif in zip(val_xmls, val_tifs):
                    self.move_if_new(xml, val_path)
                    self.move_if_new(tif, val_path)


    def move_test(self):
        test_path = self.dataset_root / "original" / "test"
        
        for unzip_path in self.unzip_paths:
            if self.expected_zips["test"] in str(unzip_path):
                test_files = (unzip_path / unzip_path.stem).glob("*")
                for test_file in test_files:
                    self.move_if_new(test_file, test_path)

class MoNuSACMover(Mover):
    def __init__(self, dataset_root: str | Path, unzip_paths: list[Path]):
        super().__init__(dataset_root, unzip_paths)
        self.expected_zips = {
            "train": "MoNuSAC_images_and_annotations",
            "val": "MoNuSAC_images_and_annotations",
            "test": "MoNuSAC Testing Data and Annotations"

        }

    def move_train(self):
        train_path = self.dataset_root / "original" / "train"

        for unzip_path in self.unzip_paths:
            if self.expected_zips["train"] in str(unzip_path):

                patients = list((unzip_path / unzip_path.stem).glob("**/*.xml"))
                train_patients = patients[:-11]

                pooldata = [(patient, train_path) for patient in train_patients]
                with Pool() as pool:
                    for _ in tqdm(pool.istarmap(self.move_patient, pooldata), 
                                  total=len(pooldata)):
                        pass

    def move_patient(self, patient: Path, dest: Path):
        for file in patient.iterdir():
            if not (dest / file.name).exists():
                shutil.move(file, dest)

    def move_valid(self):
        val_path = self.dataset_root / "original" / "val"

        for unzip_path in self.unzip_paths:
            if self.expected_zips["val"] in str(unzip_path):

                patients = list((unzip_path / unzip_path.stem).glob("**/*.xml"))
                val_patients = patients[-11:]

                pooldata = [(patient, val_path) for patient in val_patients]
                with Pool() as pool:
                    for _ in tqdm(pool.istarmap(self.move_patient, pooldata), 
                                  total=len(pooldata)):
                        pass

    def move_test(self):
        test_path = self.dataset_root / "original" / "test"
        
        for unzip_path in self.unzip_paths:
            if self.expected_zips["test"] in str(unzip_path):
                patients = (unzip_path / unzip_path.stem).glob("*")
                pooldata = [(patient, test_path) for patient in patients]
                with Pool() as pool:
                    for _ in tqdm(pool.istarmap(self.move_patient, pooldata), 
                                  total=len(pooldata)):
                        pass

                for test_file in test_files:
                    self.move_if_new(test_file, test_path)


class MaskGenerator(ABC):
    def __init__(self, dataset_root: str | Path):
        if isinstance(dataset_root, str):
            dataset_root = Path(dataset_root)
        dataset_name = self.__class__.__name__.split("MaskGenerator")[0]
        self.dataset_root = dataset_root / dataset_name
    
    @property
    def dataset_root(self):
        return self._dataset_root
    
    @dataset_root.setter
    def dataset_root(self, value: Path):
        self._dataset_root: Path = value
        if not self._dataset_root.exists():
            self._dataset_root.mkdir(parents=True)

    def generate_masks(self):
        mask_paths = [self.dataset_root / "masks" / "train",
                      self.dataset_root / "masks" / "val",
                      self.dataset_root / "masks" / "test"]
        
        original_paths = [self.dataset_root / "original" / "train",
                          self.dataset_root / "original" / "val",
                          self.dataset_root / "original" / "test"]
        # Early stop if masks are already generated
        if not self.should_generate(mask_paths):
            print("Masks already exist, skipping generation...")
            return

        for mask_path, original_path in zip(mask_paths, original_paths):
            if not mask_path.exists():
                mask_path.mkdir(parents=True)
            
            pooldata = self._get_pooldata(mask_path, original_path)
            with Pool() as pool:
                for _ in tqdm(pool.istarmap(self._generate_mask, pooldata), total=len(pooldata)):
                    pass

    def should_generate(self, mask_paths: list[Path]):
        for path in mask_paths:
            if not path.exists() or not any(path.iterdir()):
                return True
        return False

    @abstractmethod
    def _get_pooldata(self, mask_path: Path, original_path: Path):
        """Return data needed for mask generation as a list of tuples"""

    @abstractmethod
    def _generate_mask(self, mask_path: Path, *args):
        """Generate a single mask for the dataset"""

class MoNuSegMaskGenerator(MaskGenerator):
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)
        self.xml_parser = MoNuSegXMLParser()

    def _get_pooldata(self, mask_path: Path, original_path: Path):
        patients = sorted(original_path.glob("*.xml"))
        return [(mask_path, patient) for patient in patients]

    def _generate_mask(self, mask_path: Path, patient: Path):
        patient_image = patient.with_suffix(".tif")
        if not patient_image.exists():
            return

        width, height = imagesize.get(patient_image)
        nuclei = self.xml_parser.parse(patient)

        # skimage.draw.polygon uses the point-in-polygon test for accuracy!
        # DO NOT use opencv to draw polygons that are defined with real coordinates
        mask = np.zeros((height, width))
        for nucleus in nuclei:
            rr, cc = polygon(nucleus[:,1], nucleus[:,0], mask.shape)
            mask[rr,cc] = 1
        
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

class XMLParser(ABC):
    def __init__(self):
        self._xml = None
        pass

    @property
    def xml(self) -> Path:
        return self._xml
    
    @xml.setter
    def xml(self, value: Path):
        if self._xml is None or self._xml != value:
            self._xml = value
            self._xml_tree = ET.parse(self._xml)

    def parse(self, xml: Path) -> list[np.ndarray]:
        self.xml = xml
        return self._parse()

    @abstractmethod
    def _parse(self) -> list[np.ndarray]:
        """pass the xml and return a list of nuclei outlines"""

class MoNuSegXMLParser(XMLParser):
    def __init__(self):
        super().__init__()

    def _parse(self) -> list[np.ndarray]:
        root = self._xml_tree.getroot()
    
        nuclei = []
        regions = root.findall("Annotation/Regions/Region")
        for region in regions:
            vertices = region.findall("Vertices/Vertex")

            # Invalid region by definition
            if len(vertices) < 3:
                continue

            nucleus = np.zeros((len(vertices), 2))
            for i, vertex in enumerate(vertices):
                nucleus[i,0] = float(vertex.get("X"))
                nucleus[i,1] = float(vertex.get("Y"))
            nuclei.append(nucleus)
        
        return nuclei

class Yolofier(ABC):
    def __init__(self, dataset_root: str | Path):
        if isinstance(dataset_root, str):
            dataset_root = Path(dataset_root)
        dataset_name = self.__class__.__name__.split("Yolofier")[0]
        self.dataset_root = dataset_root / dataset_name

    def yolofy(self):
        yolofy_paths = [self.dataset_root / "yolo" / "train",
                      self.dataset_root / "yolo" / "val",
                      self.dataset_root / "yolo" / "test"]
        
        mask_paths = [self.dataset_root / "masks" / "train",
                      self.dataset_root / "masks" / "val",
                      self.dataset_root / "masks" / "test"]

        original_paths = [self.dataset_root / "original" / "train",
                          self.dataset_root / "original" / "val",
                          self.dataset_root / "original" / "test"]
        
        if not self.should_yolofy(yolofy_paths):
            print("Yolofied data already exists, skipping...")
            return

        for yolofy_path, mask_path, original_path in zip(yolofy_paths, mask_paths, original_paths):
            if not yolofy_path.exists():
                yolofy_path.mkdir(parents=True)
            
            pooldata = self._get_pooldata(yolofy_path, mask_path, original_path)
            with Pool() as pool:
                for _ in tqdm(pool.istarmap(self._yolofy, pooldata), total=len(pooldata)):
                    pass

    def should_yolofy(self, yolofy_paths: list[Path]):
        for path in yolofy_paths:
            if not path.exists() or not any(path.iterdir()):
                return True
        return False

    @abstractmethod
    def _get_pooldata(self, yolofy_path: Path, mask_path: Path, original_path: Path):
        """Return data needed for mask generation as a list of tuples"""

    @abstractmethod
    def _yolofy(self, yolofy_path: Path, *args):
        """Generate a single mask for the dataset"""

class MoNuSegYolofier(Yolofier):
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)
        self.xml_parser = MoNuSegXMLParser()

    def _get_pooldata(self, yolofy_path: Path, mask_path: Path, original_path: Path):
        patient_images = sorted(original_path.glob("*.tif"))
        patient_masks = sorted(mask_path.glob("*.png"))
        pooldata = [(yolofy_path, patient_image, patient_mask) 
                     for patient_image, patient_mask 
                     in zip(patient_images, patient_masks)]
        return pooldata

    def _yolofy(self, yolofy_path: Path, patient_image: Path, patient_mask: Path):
        if not patient_image.exists() or not patient_mask.exists():
            return

        # YOLOv8 requires .png images
        patient_img = cv2.imread(str(patient_image))
        filename = str((yolofy_path / patient_image.stem).with_suffix(".png"))
        cv2.imwrite(filename, patient_img)

        height, width = patient_img.shape[:2]

        mask = cv2.imread(str(patient_mask), cv2.IMREAD_GRAYSCALE)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        normalised_contours = []
        for contour in contours:
            float_contour = contour.astype(float)
            float_contour[:,0,0] = float_contour[:,0,0] / width
            float_contour[:,0,1] = float_contour[:,0,1] / height
            normalised_contours.append(float_contour)

        annotations = self.contours_to_annotations(normalised_contours)

        filename = (yolofy_path / patient_image.stem).with_suffix(".txt")
        with open(filename, "w") as f:
            f.write(annotations)

    def contours_to_annotations(self, contours: np.ndarray) -> str:
        annotations = ""
        
        for contour in contours:
            annotations += "0"
            for vertex in contour:
                annotations += f" {vertex[0,0]} {vertex[0,1]}"
            # close off the nucleus polygon (using first vertex of contour)
            annotations += f" {contour[0, 0, 0]} {contour[0, 0, 1]}"
            annotations += "\n"

        return annotations