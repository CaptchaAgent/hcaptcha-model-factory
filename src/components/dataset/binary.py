import os
import typing

import torch.utils.data as data
import yaml
from PIL import Image
from loguru import logger


class BinaryDataset(data.Dataset):
    """Binary dataset for classification"""

    def __init__(
        self,
        root: str,
        cfg_path: str = None,
        flag: str = "train",
        classes: typing.Optional[typing.List[str]] = None,
        transform: typing.Optional[typing.Callable] = None,
    ):
        """
        :param root: hook to factory/data/[task]/
        :param cfg_path: path to config file
        :param flag: train, val, test
        :param classes: list of classes
        :param transform: transform function

        file structure:
        - [dir_dataset]
            - [classes[0]]
                - [file]
            - [classes[1]]
                - [file]
            - all.yaml
            - train.yaml
            - val.yaml
            - test.yaml
        """
        self._root = root
        self._cfg_path = cfg_path or os.path.join(self._root, f"{flag}.yaml")
        self._flag = flag or os.path.basename(self._cfg_path).split(".")[0]
        self._classes = ["yes", "bad"] if classes is None else classes
        self._transform = transform

        self._init_cfg()

    def _init_cfg(self):
        if not os.path.exists(self._cfg_path):
            logger.error(f"{self._cfg_path} not found")
            raise FileNotFoundError

        if self._flag not in ["train", "val", "test", "all"]:
            logger.error(f"Invalid flag: {self._flag}")
            raise ValueError

        with open(self._cfg_path, "r") as file:
            self._cfg = yaml.safe_load(file)

        self._data = self._cfg["data"]

        logger.info(f"Dataset loaded {len(self._data)} images from {self._cfg_path}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        fname = self._data[index]["fname"]
        label = self._data[index]["label"]

        img = Image.open(fname).convert("RGB")

        if self._transform:
            img = self._transform(img)

        return img, label
