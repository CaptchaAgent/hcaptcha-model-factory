import os
import typing

from fire import Fire

from api.scaffold import diagnose_task
from api.scaffold.fatories import ResNet
from api.scaffold.auto_label import ClusterLabeler
from config import Config, logger
from utils import ToolBox


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
        task = ToolBox.split_prompt(input("prompt[en] --> "), lang="en")
        auto_label = input("auto_label? [y/n] --> ")
        if auto_label in ["y", "Y"]:
            data_dir = os.path.join(Config.DIR_DATABASE, task)
            unlabel_dir = os.path.join(data_dir, "unlabel")
            if not os.path.exists(unlabel_dir):
                os.makedirs(unlabel_dir)

            os.system(f"start {unlabel_dir}")
            input(
                "please put all the images in the `unlabel` folder and press any key to continue..."
            )

            labeler = ClusterLabeler(data_dir=data_dir)
            labeler.run()
            logger.info("Auto labeling completed")

        cmd_train = input("start to train now? [y/n] --> ")
        if cmd_train in ["y", "Y"]:
            Scaffold.train(task=task)

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
    @logger.catch()
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
        # Scaffold.train.__func__(task, epochs, batch_size)
        # Scaffold.val.__func__(task)
        Scaffold.train(task, epochs, batch_size)
        Scaffold.val(task)

    @staticmethod
    def auto_label():
        """
        Automatically label the image

        Usage: python main.py auto_label

        :return:
        """
        from api.scaffold.auto_label import AutoLabel

        AutoLabel().run()


if __name__ == "__main__":
    Fire(Scaffold)
