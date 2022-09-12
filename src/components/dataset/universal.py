import os
import typing

import torch
import yaml
from loguru import logger
from torch.utils.data import Dataset


class UniversalDataset(Dataset):
    """Universal dataset for classification"""

    def __init__(
        self,
        dir_dataset: str,
        cfg_path: str = None,
        flag: str = "test",
        transform: typing.Optional[typing.Callable] = None,
    ):
        """
        :param dir_dataset: hook to factory/data/[task]/
        :param cfg_path: path to config file
        :param flag: val, test
        :param transform: transform function

        file structure:
        - [dir_dataset]
            - [classes[0]]
                - [file]
            - [classes[1]]
                - [file]
            - [file(unlabeled)]
            - all.yaml
            - val.yaml
            - test.yaml
        """
        self._dir_dataset = dir_dataset
        self._cfg_path = cfg_path or os.path.join(self._dir_dataset, f"{flag}.yaml")
        self._flag = flag
        self._transform = transform

        self._init_cfg()

    def _init_cfg(self):
        if not os.path.exists(self._cfg_path):
            logger.error(f"{self._cfg_path} not found")
            raise FileNotFoundError

        if self._flag not in ["val", "test"]:  # note: train is not allowed!
            logger.error(f"Invalid flag: {self._flag}")
            raise ValueError

        with open(self._cfg_path, "r") as f:
            self._cfg = yaml.load(f, Loader=yaml.FullLoader)

        self._data = self._cfg["data"]
        self._task = {}
        for data in self._data:
            if data["cid"] not in self._task:
                self._task[data["cid"]] = []

            if self._flag == "test":
                self._task[data["cid"]].append(data["fname"])
            else:
                self._task[data["cid"]].append((data["fname"], data["label"]))

        logger.info(
            f"Dataset loaded: {self._cfg_path} with {len(self._data)} images and {len(self._task)} challenges"
        )

    def __len__(self):
        return len(self._task)

    def __getitem__(self, index):
        # get 12 images with cid=self._task[index]
        image_paths = self._task[index]
        images = []
        for image_path in image_paths:
            if self._flag == "test":
                img = torch.load(os.path.join(self._dir_dataset, image_path))
                label = None
            else:
                img_path, label = image_path
                img = torch.load(os.path.join(self._dir_dataset, img_path))
            if self._transform:
                img = self._transform(img)
            images.append((img, label))

        return images
