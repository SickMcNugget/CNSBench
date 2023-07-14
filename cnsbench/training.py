from abc import abstractmethod, ABC
from pathlib import Path
import re

class ConfigError(Exception):
    """Error for regex parsing of config file names"""
    pass

class Config:
    config_pattern = re.compile(r"""(?P<model>.+)
                                    _(?P<backbone>.+)
                                    _(?P<gpus>\d+)
                                    xb(?P<batch_size>\d+)
                                    -(?P<run_length>\d+)
                                    (?P<epochs_or_it>[ek])
                                    _(?P<dataset>\w+)
                                    -(?P<image_size>\d+x\d+)""", re.VERBOSE)

    def __init__(self, config_path: str | Path):
        if isinstance(config_path, str):
            config_path = Path(config_path)
        self.config_path = config_path
        self.config_dict = self.parse_config()

    def parse_config(self) -> dict:
        config_name = self.config_path.stem
        match = Config.config_pattern.match(config_name)

        if match is None:
            raise ConfigError(f"Unable to parse config: {config_name}")
        
        match_dict = match.groupdict()
        config_dict = {}
        
        config_dict["model"] = match_dict["model"]
        config_dict["backbone"] = match_dict["backbone"]
        config_dict["dataset"] = match_dict["dataset"]

        gpus = match_dict.get("gpus")
        batch_size = match_dict.get("batch_size")
        run_length = match_dict.get("run_length")
        epochs_or_iterations = match_dict.get("epochs_or_it")
        image_size = match_dict.get("image_size")

        if gpus is not None:
            config_dict["gpus"] = int(gpus)
        if batch_size is not None:
            config_dict["batch_size"] = int(batch_size)
        if run_length is not None and epochs_or_iterations is not None:
            if epochs_or_iterations == "k":
                config_dict["iterations"] = int(run_length) * 1000
            else:
                config_dict["epochs"] = int(run_length)
        if image_size is not None:
            image_size = image_size.split("x")
            image_size = tuple(int(dimension) for dimension in image_size)
            config_dict["image_size"] = image_size

        return config_dict

class Trainer(ABC):
    def __init__(self):
        self.dataset_choices = [
            "monuseg",
            "cryonuseg",
            "tnbc",
            "monusac"
        ]


    def choose_dataset(self):
        pass

    @abstractmethod
    def setup(self):
        """Prepare any dependencies for training.
        
        For example, when training MMSegmentation, datasets need to be
        registered prior to training. This method is where the registration
        should take place.
        """

    @abstractmethod
    def configure(self):
        """Configure model and training parameters.
        
        This function occurs after dataset selection and before training.
        It should be used to apply hyperparameters, logging hooks and the like.
        """

    @abstractmethod
    def train(self):
        """Train a neural network.
        
        
        """