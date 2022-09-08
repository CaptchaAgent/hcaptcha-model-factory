import typing

from fire import Fire

from api.scaffold import diagnose_task
from api.scaffold.fatories import ResNet
from config import Config, logger
from utils import ToolBox


@logger.catch()
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
    def train(
        task: str,
        epochs: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
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
        task: str,
        epochs: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
    ):
        """
        Connect train and val

        Usage: python main.py trainval --task=[labelName]

        :param task: label name
        :param epochs:
        :param batch_size:
        :return:
        """
        Scaffold.train.__func__(task, epochs, batch_size)
        Scaffold.val.__func__(task)


if __name__ == "__main__":
    Fire(Scaffold)
