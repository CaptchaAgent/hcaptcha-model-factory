import typing

from fire import Fire

from api.scaffold import diagnose_task
from api.scaffold.fatories import ResNet
from config import ConfigT, logger


@logger.catch()
class Scaffold:
    _COMMAND_TRAIN = "train"
    _COMMAND_VAL = "val"
    _COMMAND_TEST = "test"

    _model = None

    @staticmethod
    def train(
        task: str, epochs: typing.Optional[int] = None, batch_size: typing.Optional[int] = None
    ):
        """
        Train the specified model and output an ONNX object

        :param task: label name
        :param epochs:
        :param batch_size:
        :return:
        """
        model = Scaffold._model or ResNet(
            task_name=diagnose_task(task),
            epochs=epochs,
            batch_size=batch_size,
            dir_dataset=ConfigT.DIR_DATABASE,
            dir_model=ConfigT.DIR_MODEL,
        )
        model.train()
        model.conv_pth2onnx(verbose=False)
        Scaffold._model = model

    @staticmethod
    def val(task: str):
        """Detects the specified model object"""
        model = Scaffold._model or ResNet(
            task_name=diagnose_task(task),
            dir_dataset=ConfigT.DIR_DATABASE,
            dir_model=ConfigT.DIR_MODEL,
        )
        model.val()

    @staticmethod
    def trainval(
        task: str, epochs: typing.Optional[int] = None, batch_size: typing.Optional[int] = None
    ):
        Scaffold.train(task, epochs, batch_size)
        Scaffold.val(task)


if __name__ == "__main__":
    Fire(Scaffold)
