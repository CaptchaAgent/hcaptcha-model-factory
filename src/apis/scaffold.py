import os
import subprocess
import sys
import typing

from loguru import logger

from components.auto_label import ClusterLabeler
from components.config import Config
from components.utils import ToolBox
from factories.resnet import ResNet


class Scaffold:
    _model = None

    @staticmethod
    @logger.catch
    def new():
        """
        [dev for challenger] Initialize the project directory

        Usage: python main.py new
        ---

        >>> input("prompt[en] --> ")
            prompt[en] --> Please click each image containing a dog-shaped cookie
                => `dog_shaped_cookie`
            prompt[en] --> horse with white legs
                => `horse_with_white_legs`
            prompt[en] --> ""
                => raise TypeError
        >>> input(f"Use AI to automatically label datasets? {choices}  --> ")
            1. Copy the unbinary image files to the automatically opened folder.
            2. OkAction，Waiting for AI to automatically label.
        >>> input(f"Start automatic training? {choices} --> ")
            3. Check the results of automatic classification(Manual correction).
            4. If the error rate is too high, it is recommended to cancel the training,
            otherwise the training workflow can be continued.
        :return:
        """
        boolean_yes = "y"
        boolean_no = "n"
        choices = {boolean_yes, boolean_no}

        # Prepend the detector to avoid invalid interactions
        task = ToolBox.split_prompt(input("prompt[en] --> "), lang="en")
        task = diagnose_task(task)

        # IF AUTO-LABEL
        prompts = f"Use AI to automatically label datasets? {choices} --> "
        while (auto_label := input(prompts)) not in choices:
            continue
        if auto_label == "y":
            data_dir = os.path.join(Config.DIR_DATABASE, task)

            # Create and open un-labeled dir
            unlabel_dir = os.path.join(data_dir, "unlabel")
            os.makedirs(unlabel_dir, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(unlabel_dir)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, unlabel_dir])

            # Block the main process, waiting for manual operation
            input(
                "please put all the images in the `unlabel` folder and press any key to continue..."
            )
            ClusterLabeler(data_dir=data_dir).run()
            logger.success("Auto labeling completed")

        # IF AUTO-TRAIN
        prompts = f"Start automatic training? {choices} --> "
        while (cmd_train := input(prompts)) not in choices:
            continue
        if cmd_train == "y":
            Scaffold.train(task=task)

    @staticmethod
    @logger.catch
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
    @logger.catch
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
    @logger.catch
    def test_onnx(task: str, flag: str = "all"):
        """
        Test the ONNX model

        Usage: python main.py test_onnx --task=[labelName]

        :param task: label name
        :return:
        """
        model = Scaffold._model or ResNet(
            task_name=diagnose_task(task),
            dir_dataset=Config.DIR_DATABASE,
            dir_model=Config.DIR_MODEL,
        )
        return model.test_onnx(flag=flag)

    @staticmethod
    @logger.catch
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
    badcode = {
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
    for code, right_code in badcode.items():
        task_name.replace(code, right_code)

    task_name = task_name.strip()
    logger.debug(f"Diagnose task | task_name={task_name}")

    return task_name
