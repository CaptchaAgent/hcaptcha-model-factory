import typing

import torch


class Params:
    LOSS_FN_FOCAL = "focal"
    LOSS_FN_CROSS_ENTROPY = "cross_entropy"
    OPTIMIZER_ADAM = "adam"
    OPTIMIZER_SGD = "SGD"


class ModelFactory:
    """Nothing is true; Everything is permitted."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 4
    EPOCHS = 200

    # Dir of ONNX output
    DIR_MODEL = "model"

    # hook to factory/data/[task]/
    # FLAG_ORIGIN = "origin"
    FLAG_POSITIVE = "yes"
    # FLAG_POSITIVE_DIGIT = "0"
    FLAG_NEGATIVE = "bad"
    # FLAG_NEGATIVE_DIGIT = "1"
    FLAG_ALL = "all.yaml"
    FLAG_TRAIN = "train.yaml"
    FLAG_VAL = "val.yaml"
    FLAG_TEST = "test.yaml"

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
        self._dir_dataset = dir_dataset
        self._dir_model = dir_model or self.DIR_MODEL
        self._epochs = epochs or self.EPOCHS
        self._batch_size = batch_size or self.BATCH_SIZE

        self._build_env()

    def _build_env(self):
        raise NotImplementedError

    def register_task(self):
        pass

    def split_dataset(self):
        """Label datasets proportionally"""

    def conv_pth2onnx(self):
        """output an ONNX object"""

    def output_benchmarks(self):
        """Do not go gentle into that good night"""
