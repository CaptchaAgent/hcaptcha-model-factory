import typing

from fire import Fire

from api.scaffold import split_dataset, register_task, diagnose_task
from api.scaffold import train, conv_pth_to_onnx, val


class Scaffold:
    _COMMAND_TRAIN = "train"
    _COMMAND_VAL = "val"
    _COMMAND_TEST = "test"

    def __init__(
        self, task: str, split: typing.Optional[bool] = None, mode: typing.Optional[str] = None
    ):
        self.task = diagnose_task(task)
        register_task(self.task)

        self.split = split
        self.mode = mode

        if self.split is True:
            split_dataset(task_name=self.task)

        if self.mode is not None and isinstance(self.mode, str):
            if self._COMMAND_TRAIN in self.mode:
                self.train()
            if self._COMMAND_VAL in self.mode:
                self.val()

    @staticmethod
    def _new():
        """[NotImplemented] build new workflow"""

    @staticmethod
    def _collect():
        """[NotImplemented] claim dataset"""

    @staticmethod
    def train():
        """Train the specified model and output an ONNX object"""
        train()
        conv_pth_to_onnx()

    @staticmethod
    def val():
        """Detects the specified model object"""
        val()

    @staticmethod
    def _test():
        """NotImplemented"""


if __name__ == "__main__":
    Fire(Scaffold)
