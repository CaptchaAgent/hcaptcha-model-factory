import typing

from loguru import logger

from components.config import Config
from components.utils import ToolBox
from factories.resnet import ResNet

BADCODE = {
    "а": "a",
    "е": "e",
    "e": "e",
    "i": "i",
    "і": "i",
    "ο": "o",
    "с": "c",
    "ԁ": "d",
    "ѕ": "s",
    "һ": "h",
    "у": "y",
    "р": "p",
}


def diagnose_task(task_name: typing.Optional[str]) -> typing.Optional[str]:
    """Input detection and normalization"""
    if not task_name or not isinstance(task_name, str) or len(task_name) < 2:
        raise TypeError(f"({task_name})TASK should be string type data")

    # Filename contains illegal characters
    inv = {"\\", "/", ":", "*", "?", "<", ">", "|"}
    if s := set(task_name) & inv:
        raise TypeError(f"({task_name})TASK contains invalid characters({s})")

    # Normalized separator
    rnv = {" ", ",", "-"}
    for s in rnv:
        task_name = task_name.replace(s, "_")

    # Convert bad code
    for code in BADCODE:
        task_name.replace(code, BADCODE[code])

    task_name = task_name.strip()
    logger.debug(f"Diagnose task | task_name={task_name}")

    return task_name


class Scaffold:
    _model = None

    @staticmethod
    def new():
        """
        [dev for challenger] Initialize the project directory

        Usage: python main.py new
            prompt[en] --> Please click each image containing a dog-shaped cookie
            task=`dog_shaped_cookie`

        :return:
        """
        return Scaffold.train(ToolBox.split_prompt(input("prompt[en] --> "), lang="en"))

    @staticmethod
    @logger.catch()
    def train(
        task: str, epochs: typing.Optional[int] = None, batch_size: typing.Optional[int] = None
    ):
        """
        Train the specified model and output an ONNX object

        Usage: python main.py train --task=bird_flying
        or: python main.py train --task bird_flying
        or: python main.py train --task bird_flying --epochs 100
        or: python main.py train --task bird_flying --batch_size 4

        :param task: label name
        :param epochs:
        :param batch_size:
        :return:
        """
        model = Scaffold._model or ResNet(
            task_name=diagnose_task(task),
            epochs=epochs,
            batch_size=batch_size,
            dir_dataset=Config.DIR_DATABASE,
            dir_model=Config.DIR_MODEL,
        )
        model.train()
        model.conv_pth2onnx(verbose=False)
        Scaffold._model = model

    @staticmethod
    @logger.catch()
    def val(task: str):
        """
        Detects the specified model object

        Usage: python main.py val --task=lion
        or: python main.py val --task lion

        :param task: label name
        :return:
        """
        model = Scaffold._model or ResNet(
            task_name=diagnose_task(task),
            dir_dataset=Config.DIR_DATABASE,
            dir_model=Config.DIR_MODEL,
        )
        model.val()

    @staticmethod
    def trainval(
        task: str, epochs: typing.Optional[int] = None, batch_size: typing.Optional[int] = None
    ):
        """
        Connect train and val

        Usage: python main.py trainval --task=[labelName]

        :param task: label name
        :param epochs:
        :param batch_size:
        :return:
        """
        Scaffold.train(task, epochs, batch_size)
        Scaffold.val(task)
