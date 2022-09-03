import os
import typing

from config import logger, ConfigT
from .train import train, conv_pth_to_onnx
from .val import val

__all__ = ["train", "conv_pth_to_onnx", "val", "split_dataset", "register_task", "diagnose_task"]


def split_dataset(task_name: str):
    return task_name


def register_task(task_name: str):
    """[PENDING] Detect and initialize the task environment"""
    dir_task = os.path.join(ConfigT.DIR_DATABASE, task_name)

    dir_task_origin = os.path.join(dir_task, ConfigT.FLAG_ORIGIN)
    dir_task_origin_positive = os.path.join(dir_task_origin, ConfigT.FLAG_POSITIVE)
    dir_task_origin_negative = os.path.join(dir_task_origin, ConfigT.FLAG_NEGATIVE)
    dir_task_train = os.path.join(dir_task, ConfigT.FLAG_TRAIN)
    dir_task_val = os.path.join(dir_task, ConfigT.FLAG_VAL)
    dir_task_test = os.path.join(dir_task, ConfigT.FLAG_TEST)

    if not os.path.exists(dir_task_origin_positive) or not os.path.exists(dir_task_origin_negative):
        raise ResourceWarning(
            f"The original dataset structure is incomplete | dir={dir_task_origin}"
        )

    for ttl in [dir_task_train, dir_task_val, dir_task_test]:
        os.makedirs(os.path.join(ttl, ConfigT.FLAG_POSITIVE_DIGIT), exist_ok=True)
        os.makedirs(os.path.join(ttl, ConfigT.FLAG_NEGATIVE_DIGIT), exist_ok=True)


def diagnose_task(task_name: typing.Optional[str]) -> typing.Optional[str]:
    """Input detection and normalization"""
    if not task_name or not isinstance(task_name, str):
        raise TypeError(f"({task_name})TASK should be string type data")

    task_name = task_name.replace(" ", "_")
    task_name = task_name.replace(",", "_")
    for code in ConfigT.BADCODE:
        task_name.replace(code, ConfigT.BADCODE[code])

    logger.debug(f"Diagnose task | task_name={task_name}")
    return task_name
