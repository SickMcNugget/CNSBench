from __future__ import annotations

#builtins
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from zipfile import ZipFile
import shutil
import multiprocessing.pool as mpp
from multiprocessing import Pool

# Linalg, parsing, image manip
import xml.etree.ElementTree as ET
from skimage.draw import polygon
import numpy as np
import cv2
import imagesize
from openslide import OpenSlide

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

class DownloaderError(Exception):
    """Raised when the downloader cannot download a file"""
    pass

class Downloader(ABC):
    def __init__(self, zip_sources: list[str]):
        self.zip_sources: list[str] = zip_sources
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
    def zip_sources(self):
        return self._zip_sources

    @zip_sources.setter
    def zip_sources(self, value):
        self._zip_sources = value

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
        for zip_source in self.zip_sources:
            zip_path = self._download(zip_source)
            zip_paths.append(zip_path)

        self.zip_paths = [self.prefix / zip_path.name for zip_path in zip_paths]

        # All zips need to be moved to an appropriate location, so this is relevant
        for src, dest in zip(zip_paths, self.zip_paths):
            shutil.move(src, dest)

        return self.zip_paths

    def _should_download(self) -> bool:
        for zip_name in self.expected_zip_names:
            if not (self.prefix / zip_name).exists():
                return True
        return False

    @abstractmethod
    def _download(self, zip_source: str) -> Path:
        """Downloads a zip file from it's online id and returns the downloaded path"""

class MoNuSegDownloader(Downloader):
    def __init__(self, zip_sources: list[str] = ["1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA", 
                                             "1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw"]):
        super().__init__(zip_sources)
        self.expected_zip_names = ["MoNuSeg 2018 Training Data.zip",
                                   "MoNuSegTestData.zip"]

    def _download(self, zip_source: str) -> Path:
        zip_path = gdown.download(id=zip_source)
        return Path(zip_path)
    
    def _should_download(self) -> bool:
        for zip_name in self.expected_zip_names:
            if not (self.prefix / zip_name).exists():
                return True
        return False

class MoNuSACDownloader(Downloader):
    def __init__(self, zip_sources: list[str] = ["1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq", 
                                             "1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ"]):
        super().__init__(zip_sources)
        self.expected_zip_names = ["MoNuSAC_images_and_annotations.zip",
                                   "MoNuSAC Testing Data and Annotations.zip"]

    def _download(self, zip_source: str) -> Path:
        zip_path = gdown.download(id=zip_source)
        return Path(zip_path)
    
class TNBCDownloader(Downloader):
    def __init__(self, zip_sources: list[str] = ["10.5281/zenodo.1175282"]):
        super().__init__(zip_sources)
        self.expected_zip_names = ["TNBC_NucleiSegmentation.zip"]

    def _download(self, zip_source: str) -> Path:
        try:
            subprocess.run(["zenodo_get", "-o", self.prefix, zip_source])
        except FileNotFoundError:
            pass
        return self.prefix / self.expected_zip_names[0]

class CryoNuSegDownloader(Downloader):
    def __init__(self, zip_sources: list[str] = ["ipateam/segmentation-of-nuclei-in-cryosectioned-he-images"]):
        super().__init__(zip_sources)
        self.expected_zip_names = ["segmentation-of-nuclei-in-cryosectioned-he-images.zip"]

    def _download(self, zip_source: str) -> Path:
        try:
            ret = subprocess.run(["kaggle", 
                                     "datasets", 
                                     "download", 
                                     "-p",
                                     str(self.prefix),
                                     "-d", 
                                     zip_source],
                                     stderr=subprocess.DEVNULL)
            ret.check_returncode()
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise DownloaderError("Please configure ~/.kaggle/kaggle.json. Follow the instructions" \
                + " on their website.")
        
        return self.prefix / self.expected_zip_names[0]

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
        need_unzip = False
        for unzip_path in self.unzip_paths:
            if not unzip_path.exists():
                need_unzip = True
        
        if need_unzip:
            for zip_path, unzip_path in zip(self.zip_paths, self.unzip_paths):
                with ZipFile(zip_path, "r") as zfile:
                    zfile.extractall(unzip_path)
        else:
            print("Unzipped data already present, skipping...")
        
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

    def copy_folder(self, folder: Path, dest: Path):
        for image in folder.iterdir():
            self.copy_if_new(image, dest)

    def copy_if_new(self, src: Path, dest: Path):
        if not (dest / src.name).exists():
            shutil.copy(src, dest)

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

        for original_path in original_paths:
            split = original_path.stem
            for unzip_path in self.unzip_paths:
                if self.expected_zips[split] in str(unzip_path):
                    move_fn = getattr(self, f"move_{split}")
                    move_fn(original_path, unzip_path)

    @abstractmethod
    def move_train(self, train_path: Path, unzip_path: Path):
        """Move the dataset's training data to its new directory"""

    @abstractmethod
    def move_val(self, val_path: Path, unzip_path: Path):
        """Move the dataset's validation data to its new directory"""

    @abstractmethod
    def move_test(self, test_path: Path, unzip_path: Path):
        """Move the dataset's testing data to its new directory"""

class MoNuSegMover(Mover):
    def __init__(self, dataset_root: str | Path, unzip_paths: list[Path]):
        super().__init__(dataset_root, unzip_paths)
        self.expected_zips = {
            "train": "MoNuSeg 2018 Training Data",
            "val": "MoNuSeg 2018 Training Data",
            "test": "MoNuSegTestData"
        }

    def move_train(self, train_path: Path, unzip_path: Path):
        xmls = sorted((unzip_path / unzip_path.stem).glob("**/*.xml"))
        tifs = sorted((unzip_path / unzip_path.stem).glob("**/*.tif"))
        train_xmls = xmls[:-7]
        train_tifs = tifs[:-7]

        for xml, tif in zip(train_xmls, train_tifs):
            self.copy_if_new(xml, train_path)
            self.copy_if_new(tif, train_path)

    def move_val(self, val_path: Path, unzip_path: Path):
        xmls = sorted((unzip_path / unzip_path.stem).glob("**/*.xml"))
        tifs = sorted((unzip_path / unzip_path.stem).glob("**/*.tif"))
        val_xmls = xmls[-7:]
        val_tifs = tifs[-7:]

        for xml, tif in zip(val_xmls, val_tifs):
            self.copy_if_new(xml, val_path)
            self.copy_if_new(tif, val_path)


    def move_test(self, test_path: Path, unzip_path: Path):
        test_files = (unzip_path / unzip_path.stem).glob("*")
        for test_file in test_files:
            self.copy_if_new(test_file, test_path)

class MoNuSACMover(Mover):
    def __init__(self, dataset_root: str | Path, unzip_paths: list[Path]):
        super().__init__(dataset_root, unzip_paths)
        self.expected_zips = {
            "train": "MoNuSAC_images_and_annotations",
            "val": "MoNuSAC_images_and_annotations",
            "test": "MoNuSAC Testing Data and Annotations"
        }

    def move_train(self, train_path: Path, unzip_path: Path):
        patients = list((unzip_path / unzip_path.stem).glob("*"))
        train_patients = patients[:-11]

        pooldata = [(patient, train_path) for patient in train_patients]
        with Pool() as pool:
            for _ in tqdm(pool.istarmap(self.copy_folder, pooldata), 
                            total=len(pooldata)):
                pass

    def move_val(self, val_path: Path, unzip_path: Path):
        patients = list((unzip_path / unzip_path.stem).glob("*"))
        val_patients = patients[-11:]

        pooldata = [(patient, val_path) for patient in val_patients]
        with Pool() as pool:
            for _ in tqdm(pool.istarmap(self.copy_folder, pooldata), 
                            total=len(pooldata)):
                pass

    def move_test(self, test_path: Path, unzip_path: Path):
        patients = (unzip_path / unzip_path.stem).glob("*")
        pooldata = [(patient, test_path) for patient in patients]
        with Pool() as pool:
            for _ in tqdm(pool.istarmap(self.copy_folder, pooldata), 
                            total=len(pooldata)):
                pass

class TNBCMover(Mover):
    def __init__(self, dataset_root: str | Path, unzip_paths: list[Path]):
        super().__init__(dataset_root, unzip_paths)
        self.expected_zips = {
            "train": "TNBC_NucleiSegmentation",
            "val": "TNBC_NucleiSegmentation",
            "test": "TNBC_NucleiSegmentation"
        }

    def move_train(self, train_path: Path, unzip_path: Path):
        slide_folders = sorted((unzip_path / unzip_path.stem).glob(f"**/Slide_*"))
        train_folders = slide_folders[:6]
        for train_folder in train_folders:
            self.copy_folder(train_folder, train_path)
        
        #TNBC is organised in an odd fashion, so these methods reflect it.
        train_mask_path = self.dataset_root / "masks" / "train"
        if not train_mask_path.exists():
            train_mask_path.mkdir(parents=True)

        mask_folders = sorted((unzip_path / unzip_path.stem).glob(f"**/GT_*"))
        train_mask_folders = mask_folders[:6]
        for train_mask_folder in train_mask_folders:
            self.copy_folder(train_mask_folder, train_mask_path)

    def move_val(self, val_path: Path, unzip_path: Path):
        slide_folders = sorted((unzip_path / unzip_path.stem).glob(f"**/Slide_*"))
        val_folders = [slide_folders[i] for i in [6, 9, 10]]
        for val_folder in val_folders:
            self.copy_folder(val_folder, val_path)

        val_mask_path = self.dataset_root / "masks" / "val"
        if not val_mask_path.exists():
            val_mask_path.mkdir(parents=True)

        mask_folders = sorted((unzip_path / unzip_path.stem).glob(f"**/GT_*"))
        val_mask_folders = [mask_folders[i] for i in [6, 9, 10]]
        for val_mask_folder in val_mask_folders:
            self.copy_folder(val_mask_folder, val_mask_path)

    def move_test(self, test_path: Path, unzip_path: Path):
        slide_folders = sorted((unzip_path / unzip_path.stem).glob(f"**/Slide_*"))
        test_folders = [slide_folders[i] for i in [7, 8]]
        for test_folder in test_folders:
            self.copy_folder(test_folder, test_path)
        
        test_mask_path = self.dataset_root / "masks" / "test"
        if not test_mask_path.exists():
            test_mask_path.mkdir(parents=True)

        mask_folders = sorted((unzip_path / unzip_path.stem).glob(f"**/GT_*"))
        test_mask_folders = [mask_folders[i] for i in [7, 8]]
        for test_mask_folder in test_mask_folders:
            self.copy_folder(test_mask_folder, test_mask_path)

class CryoNuSegMover(Mover):
    def __init__(self, dataset_root: str | Path, unzip_paths: list[Path]):
        super().__init__(dataset_root, unzip_paths)
        self.expected_zips = {
            "train": "segmentation-of-nuclei-in-cryosectioned-he-images",
            "val": "segmentation-of-nuclei-in-cryosectioned-he-images",
            "test": "segmentation-of-nuclei-in-cryosectioned-he-images"
        }

    def move_train(self, train_path: Path, unzip_path: Path):
        train_images = list(unzip_path.glob("tissue images/*[12].tif"))
        for train_image in train_images:
            self.copy_if_new(train_image, train_path)
        
        #TNBC is organised in an odd fashion, so these methods reflect it.
        train_mask_path = self.dataset_root / "masks" / "train"
        if not train_mask_path.exists():
            train_mask_path.mkdir(parents=True)

        for annotator in unzip_path.glob("Annotator 2*"):
            train_masks = list(annotator.glob("**/mask binary without border/*[12].png"))
            for train_mask in train_masks:
                self.copy_if_new(train_mask, train_mask_path)

    def move_val(self, val_path: Path, unzip_path: Path):
        val_images = sorted(unzip_path.glob("tissue images/*3.tif"))
        val_images = val_images[:5]
        for val_image in val_images:
            self.copy_if_new(val_image, val_path)

    
        val_mask_path = self.dataset_root / "masks" / "val"
        if not val_mask_path.exists():
            val_mask_path.mkdir(parents=True)

        for annotator in unzip_path.glob("Annotator 2*"):
            val_masks = sorted(annotator.glob("**/mask binary without border/*3.png"))
            val_masks = val_masks[:5]
            for val_mask in val_masks:
                self.copy_if_new(val_mask, val_mask_path)

    def move_test(self, test_path: Path, unzip_path: Path):
        test_images = sorted(unzip_path.glob("tissue images/*3.tif"))
        test_images = test_images[5:]
        for test_image in test_images:
            self.copy_if_new(test_image, test_path)
        
        test_mask_path = self.dataset_root / "masks" / "test"
        if not test_mask_path.exists():
            test_mask_path.mkdir(parents=True)

        for annotator in unzip_path.glob("Annotator 2*"):
            test_masks = sorted(annotator.glob("**/mask binary without border/*3.png"))
            test_masks = test_masks[5:]
            for test_mask in test_masks:
                self.copy_if_new(test_mask, test_mask_path)

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

class MoNuSACMaskGenerator(MaskGenerator):
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)
        self.xml_parser = MoNuSACXMLParser()
        self.label_map = {'Epithelial':1,
                          'Lymphocyte':2,
                          'Neutrophil':3,
                          'Macrophage':4,
                          'Ambiguous':5,}
        self.masks_1cls = self.dataset_root / "masks_1cls"
        for mask_path in [self.masks_1cls / "train", self.masks_1cls / "val", self.masks_1cls / "test"]:
            if not mask_path.exists():
                mask_path.mkdir(parents=True)
    
    def _get_pooldata(self, mask_path: Path, original_path: Path):
        patients = sorted(original_path.glob("*.xml"))
        return [(mask_path, patient) for patient in patients]
    
    def _generate_mask(self, mask_path: Path, patient: Path):
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
        for nucleus, class_label in nuclei:
            rr, cc = polygon(nucleus[:,1], nucleus[:,0], mask.shape)
            mask[rr,cc] = self.label_map[class_label]

        filename = str((mask_path / patient.stem).with_suffix(".png"))
        cv2.imwrite(filename, mask)
        
        mask_1cls = np.zeros((height, width))
        for nucleus, class_label in nuclei:
            rr, cc = polygon(nucleus[:,1], nucleus[:,0], mask_1cls.shape)
            mask_1cls[rr,cc] = 1

        filename = str((self.masks_1cls / mask_path.stem / patient.stem).with_suffix(".png"))
        cv2.imwrite(filename, mask)

class TNBCMaskGenerator(MaskGenerator):
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)

    def _get_pooldata(self, mask_path: Path, _):
        masks = list(mask_path.glob("*.png"))
        return [(mask_path, mask) for mask in masks]
    
    def _generate_mask(self, _, mask: Path):
        img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        img[img == 255] = 1
        cv2.imwrite(str(mask), img)

    def should_generate(self, mask_paths: list[Path]):
        return True

class CryoNuSegMaskGenerator(MaskGenerator):
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)

    def _get_pooldata(self, mask_path: Path, _):
        masks = list(mask_path.glob("*.png"))
        return [(mask_path, mask) for mask in masks]
    
    def _generate_mask(self, _, mask: Path):
        img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        img[img == 255] = 1
        cv2.imwrite(str(mask), img)

    def should_generate(self, mask_paths: list[Path]):
        return True


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

class MoNuSACXMLParser(XMLParser):
    def __init__(self):
        super().__init__()

    def _parse(self) -> list[np.ndarray]:
        root = self._xml_tree.getroot()
    
        nuclei = []
        annotations = root.findall("Annotation")
        for annotation in annotations:
            class_label = annotation.find("Attributes/Attribute").get("Name")
            regions = annotation.findall("Regions/Region")
            if len(regions) == 0: # Some errors in annotations
                continue
            for region in regions:
                vertices = region.findall("Vertices/Vertex")

                # Invalid region by definition
                if len(vertices) < 3:
                    continue

                nucleus = np.zeros((len(vertices), 2))
                for i, vertex in enumerate(vertices):
                    nucleus[i,0] = float(vertex.get("X"))
                    nucleus[i,1] = float(vertex.get("Y"))
                nuclei.append((nucleus, class_label))
        
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
        
        original_paths = [self.dataset_root / "original" / "train",
                          self.dataset_root / "original" / "val",
                          self.dataset_root / "original" / "test"]
        
        if not self.should_yolofy(yolofy_paths):
            print("Yolofied data already exists, skipping...")
            return

        for yolofy_path, original_path in zip(yolofy_paths, original_paths):
            if not yolofy_path.exists():
                yolofy_path.mkdir(parents=True)
            
            pooldata = self._get_pooldata(yolofy_path, original_path)
            with Pool() as pool:
                for _ in tqdm(pool.istarmap(self._yolofy, pooldata), total=len(pooldata)):
                    pass

    def should_yolofy(self, yolofy_paths: list[Path]):
        for path in yolofy_paths:
            if not path.exists() or not any(path.iterdir()):
                return True
        return False

    @abstractmethod
    def _get_pooldata(self, yolofy_path: Path, original_path: Path, *args):
        """Return data needed for mask generation as a list of tuples"""

    @abstractmethod
    def _yolofy(self, yolofy_path: Path, *args):
        """Generate a single mask for the dataset"""

class MoNuSegYolofier(Yolofier):
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)
        self.xml_parser = MoNuSegXMLParser()

    def _get_pooldata(self, yolofy_path: Path, original_path: Path):
        patient_images = sorted(original_path.glob("*.tif"))
        patient_xmls = sorted(original_path.glob("*.xml"))
        pooldata = [(yolofy_path, patient_image, patient_xml) 
                     for patient_image, patient_xml 
                     in zip(patient_images, patient_xmls)]
        return pooldata

    def _yolofy(self, yolofy_path: Path, patient_image: Path, patient_xml: Path):
        if not patient_image.exists() or not patient_xml.exists():
            return

        # YOLOv8 requires .png images
        patient_img = cv2.imread(str(patient_image))
        filename = str((yolofy_path / patient_image.stem).with_suffix(".png"))
        cv2.imwrite(filename, patient_img)

        height, width = patient_img.shape[:2]
        nuclei = self.xml_parser.parse(patient_xml)

        normalised_nuclei = []
        for nucleus in nuclei:
            nucleus[:,0] /= width
            nucleus[:,1] /= height
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
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)
        self.xml_parser = MoNuSACXMLParser()
        self.label_map = {'Epithelial': 1,
                          'Lymphocyte': 2,
                          'Neutrophil': 3,
                          'Macrophage': 4,
                          'Ambiguous': 5,}
        self.yolofy_1cls = self.dataset_root / "yolo_1cls"
        for yolofy_path in [self.yolofy_1cls / "train", self.yolofy_1cls / "val", self.yolofy_1cls / "test"]:
            if not yolofy_path.exists():
                yolofy_path.mkdir(parents=True)


    def _get_pooldata(self, yolofy_path: Path, original_path: Path):
        patient_images = sorted(original_path.glob("*.svs"))
        patient_xmls = sorted(original_path.glob("*.xml"))
        pooldata = [(yolofy_path, patient_image, patient_xml) 
                     for patient_image, patient_xml 
                     in zip(patient_images, patient_xmls)]
        return pooldata

    def _yolofy(self, yolofy_path: Path, patient_image: Path, patient_xml: Path):
        if not patient_image.exists() or not patient_xml.exists():
            return

        # YOLOv8 requires .png images
        slide = OpenSlide(str(patient_image))
        filename = str((yolofy_path / patient_image.stem).with_suffix(".png"))
        filename_1cls = str((self.yolofy_1cls / yolofy_path.stem / patient_image.stem).with_suffix(".png"))

        slide.get_thumbnail(slide.level_dimensions[0]).save(filename)
        slide.get_thumbnail(slide.level_dimensions[0]).save(filename_1cls)
        width, height = slide.level_dimensions[0]
        slide.close()

        nuclei = self.xml_parser.parse(patient_xml)
        
        normalised_contours = []
        for nucleus, class_label in nuclei:
            nucleus[:,0] /= width
            nucleus[:,1] /= height
            normalised_contours.append((nucleus, class_label))

        annotations = self.nuclei_to_annotations(normalised_contours)
        annotations_1cls = self.nuclei_to_annotations(normalised_contours, multiclass=False)

        filename = (yolofy_path / patient_image.stem).with_suffix(".txt")
        with open(filename, "w") as f:
            f.write(annotations)

        filename_1cls = (self.yolofy_1cls / yolofy_path.stem / patient_image.stem).with_suffix(".txt")
        with open(filename_1cls, "w") as f:
            f.write(annotations_1cls)

    def nuclei_to_annotations(self, nuclei: list[tuple[np.ndarray, str]], multiclass=True) -> str:
        annotations = ""
        
        for nucleus, class_label in nuclei:
            if multiclass:
                annotations += str(self.label_map[class_label])
            else:
                annotations += "0"

            for vertex in nucleus:
                annotations += f" {vertex[0]} {vertex[1]}"
            # close off the nucleus polygon (using first vertex of contour)
            annotations += f" {nucleus[0, 0]} {nucleus[0, 1]}\n"

        return annotations
    
class TNBCYolofier(Yolofier):
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)

    def _get_pooldata(self, yolofy_path: Path, original_path: Path):
        patient_imgs = sorted(original_path.glob("*.png"))
        mask_path = self.dataset_root / "masks" / original_path.stem
        patient_masks = sorted(mask_path.glob("*.png"))
        return [(yolofy_path, patient_img, patient_mask)
                for patient_img, patient_mask
                in zip(patient_imgs, patient_masks)]
    
    def _yolofy(self, yolofy_path: Path, patient_img: Path, patient_mask: Path):
        # original image is already a PNG
        shutil.copy(patient_img, (yolofy_path / patient_img.name))

        mask = cv2.imread(str(patient_mask), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        
        nuclei = cv2.findContours(mask,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        
        normalised_nuclei = []
        for nucleus in nuclei:
            float_nucleus = nucleus.astype(float)
            float_nucleus[:,:,0] /= width
            float_nucleus[:,:,1] /= height
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
    
class CryoNuSegYolofier(Yolofier):
    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)

    def _get_pooldata(self, yolofy_path: Path, original_path: Path):
        patient_imgs = sorted(original_path.glob("*.tif"))
        mask_path = self.dataset_root / "masks" / original_path.stem
        patient_masks = sorted(mask_path.glob("*.png"))
        return [(yolofy_path, patient_img, patient_mask)
                for patient_img, patient_mask
                in zip(patient_imgs, patient_masks)]
    
    def _yolofy(self, yolofy_path: Path, patient_img: Path, patient_mask: Path):
        # original image is already a PNG
        img = cv2.imread(str(patient_img))
        cv2.imwrite(str(yolofy_path / patient_img.with_suffix(".png").name), img)

        mask = cv2.imread(str(patient_mask), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        
        nuclei = cv2.findContours(mask,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        
        normalised_nuclei = []
        for nucleus in nuclei:
            float_nucleus = nucleus.astype(float)
            float_nucleus[:,:,0] /= width
            float_nucleus[:,:,1] /= height
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