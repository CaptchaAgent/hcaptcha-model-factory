import os
import shutil
import time
import typing
from dataclasses import dataclass, field

import torch
import yaml


class Params:
    LOSS_FN_FOCAL = "focal"
    LOSS_FN_CROSS_ENTROPY = "cross_entropy"
    OPTIMIZER_ADAM = "adam"
    OPTIMIZER_SGD = "SGD"


@dataclass
class DataModel:
    path: str
    task_name: str
    task_type: str
    format: typing.Dict[str, int] = field(default_factory=dict)
    data: list = field(default_factory=list)

    def __post_init__(self):
        self.format = {"img_size": 64}

    def save(self):
        template = {
            "task_name": self.task_name,
            "task_type": self.task_type,
            "format": self.format,
            "data": self.data,
        }
        with open(self.path, "w") as file:
            yaml.dump(template, file)


class TaskType:
    IMAGE_LABEL_BINARY = "image_label_binary"
    IMAGE_LABEL_AREA_SELECT = "image_label_area_select"
    IMAGE_LABEL_MULTIPLE_CHOICE = "image_label_multiple_choice"


class ModelFactory:
    """Nothing is true; Everything is permitted."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 4
    EPOCHS = 200

    # Dir of ONNX output
    DIR_MODEL = "model"

    # hook to factory/data/[task]/
    FLAG_POSITIVE = "yes"
    FLAG_NEGATIVE = "bad"
    FILENAME_YAML_ALL = "all.yaml"
    FILENAME_YAML_TRAIN = "train.yaml"
    FILENAME_YAML_VAL = "val.yaml"
    FILENAME_YAML_TEST = "test.yaml"

    def __init__(
        self,
        task_name: str,
        dir_dataset: str,
        dir_model: typing.Optional[str] = None,
        epochs: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
    ):
        """

        :param task_name:
        :param dir_dataset: hook to factory/data/
        :param dir_model: hook to factory/model/
        :param epochs:
        :param batch_size:
        """
        self._task_name = task_name
        self._dir_model = os.path.join(dir_model or self.DIR_MODEL, self._task_name)
        self._dir_dataset = os.path.join(dir_dataset, self._task_name)
        self._epochs = epochs or self.EPOCHS
        self._batch_size = batch_size or self.BATCH_SIZE

        os.makedirs(self._dir_model, exist_ok=True)

        # registered datamodels
        self._datamodels: typing.List[DataModel] = []

        self._archive_previous_models()
        self._build_env()

    def _archive_previous_models(self):
        """
        1. Archive the existing model to the ::dir_last_work:: directory
        2. Delete empty archive-folders ::dir_last_work::
        """
        dir_last_work = os.path.join(self._dir_model, f"{int(time.time())}")

        for sickle in os.listdir(self._dir_model):
            path = os.path.join(self._dir_model, sickle)
            if os.path.isfile(path):
                if sickle.endswith(".onnx") or sickle.endswith(".pth"):
                    os.makedirs(dir_last_work, exist_ok=True)
                    shutil.move(path, dir_last_work)
            elif os.path.isdir(path) and sickle.isdigit() and not os.listdir(path):
                shutil.rmtree(path, ignore_errors=True)

    def _build_env(self):
        raise NotImplementedError

    def conv_pth2onnx(self, *args, **kwargs):
        """output an ONNX object"""

    def _make_datamodel(
        self,
        flag: str,
        task_name: typing.Optional[str] = None,
        task_type: typing.Optional[str] = None,
    ):
        """
        filename_flag == self.FILENAME_YAML_*
        """
        task_name = task_name or self._task_name
        task_type = task_type or TaskType.IMAGE_LABEL_BINARY
        path_yaml = os.path.join(self._dir_dataset, flag)

        model = DataModel(path=path_yaml, task_name=task_name, task_type=task_type)
        self._datamodels.append(model)

        return model

    def save_datamodels(self):
        for model in self._datamodels:
            try:
                model.save()
            except AttributeError:
                pass
