from abc import ABC, abstractmethod
import gdown
from pathlib import Path
from zipfile import ZipFile
import shutil

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
            print("No download required")
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

class Unzipper:
    def __init__(self, zip_paths: list[Path]):
        self.zip_paths: list[Path] = zip_paths
        self.prefix: Path = Path("tmp_unzip")

    @property
    def prefix(self):
        return self._prefix
    
    @prefix.setter
    def prefix(self, value):
        self._prefix = value
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
        unzip_paths = [self.prefix / zip_path.with_suffix("").name 
                       for zip_path in self.zip_paths]
        return unzip_paths
    
    def unzip(self) -> list[Path]:
        for zip_path, unzip_path in zip(self.zip_paths, self.unzip_paths):
            with ZipFile(zip_path, "r") as zfile:
                zfile.extractall(unzip_path)

        return self.unzip_paths
    
class Mover(ABC):
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        pass

    @property
    def dataset_root(self):
        return self._dataset_root
    
    @dataset_root.setter
    def dataset_root(self, value):
        self._dataset_root = value

    def _move():
        pass

    def move_train():
        pass

    def move_valid():
        pass

    def move_test():
        pass