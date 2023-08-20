# -*- coding: utf-8 -*-
# Time       : 2023/4/22 13:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
import os.path
import shutil
import time
import webbrowser
from contextlib import suppress
from typing import Optional

import cv2
from loguru import logger

from factories.resnet import ResNet

DATA_SRC = r"E:\_GithubProjects\Sources\hcaptcha-whistleblower\src\database\rainbow_backup"
DATA_DST = "../data"
MODEL_FOLDER = "../model"
REPO_URL = "https://github.com/QIN2DIM/hcaptcha-challenger/releases/edit/model"


class Workflow:
    def __init__(self, label_name: str, deep: Optional[bool] = True):
        """

        :param label_name:
        :param deep:
        """
        self.label_name = label_name
        self.label_src = os.path.join(DATA_SRC, label_name)
        self.label_dst = os.path.join(DATA_DST, label_name)
        self.path_model = os.path.join(MODEL_FOLDER, label_name, f"{label_name}.onnx")
        self.deep_clean = deep

    def _move_dataset(self):
        """

        :return:
        """
        if self.deep_clean is True and os.path.isdir(self.label_dst):
            shutil.rmtree(self.label_dst)
        for ex in ["yes", "bad"]:
            moved_counts = 0
            label_src_ex = os.path.join(self.label_src, ex)
            label_dst_ex = os.path.join(self.label_dst, ex)
            os.makedirs(label_dst_ex, exist_ok=True)
            for fp in (samples := os.listdir(label_src_ex)):
                sample_src = os.path.join(label_src_ex, fp)
                sample_dst = os.path.join(label_dst_ex, fp)
                try:
                    with open(sample_src, "rb") as file:
                        data = file.read()
                    with open(sample_dst, "wb") as file:
                        file.write(data)
                except Exception as err:
                    logger.warning(err)
                else:
                    moved_counts += 1
            logger.success(
                f"move data - {self.label_name=} {ex=} progress=[{moved_counts}/{len(samples)}]"
            )

    def _train(self, epochs: Optional[int] = 200, batch_size: Optional[int] = 4, **kwargs):
        """

        :param epochs:
        :param batch_size:
        :param kwargs:
        :return:
        """
        from apis.scaffold import Scaffold

        epochs = kwargs.get("epochs", epochs)
        batch_size = kwargs.get("batch_size", batch_size)

        logger.info("Start automatic training after 5s ...")
        time.sleep(5)

        Scaffold.trainval(task=self.label_name, epochs=epochs, batch_size=batch_size)
        Scaffold.test_onnx(task=self.label_name)
        logger.success(f"Training complete - label_name={self.label_name}")

        # 将模型移动到显眼的位置
        shutil.copy(self.path_model, f"{self.label_name}.onnx")
        logger.success(f"Model moved successfully - label_name={self.label_name}")

    def _load_model(self):
        if (
            not os.path.isfile(self.path_model)
            or not self.path_model.endswith(".onnx")
            or not os.path.getsize(self.path_model)
        ):
            raise RuntimeError(f"ModelNotFounded - path={self.path_model}")
        return cv2.dnn.readNetFromONNX(self.path_model)

    def recur(self, per: int = 0):
        """
        RecurTraining Motion workflow
        ---------

        bird_flying         # handled label name
         ├── _inner         # recur-output
         │    ├── yes
         │    └── bad
         ├── yes            # labeled dataset for train/val
         ├── bad            # labeled dataset for train/val
         └── *.jpg          # unlabeled dataset

         1. 在一切開始前，你需要手動分類大約 100 張圖片（正反類合計），
            通過正常的 trainval 工作流獲取首個 ONNX 模型；
         2. 當你纍計獲取更多的未標注的圖片後，通過 recur 工作流使用模型進行標注（圖像二分類）；
         3. 人工檢查模型輸出，手動校準分類錯誤的極少量圖片，你可以修改標注或刪去圖片；
         4. 合并數據集，將 _inner/yes 以及 _inner/bad 的 recur 輸出合并至已分類的數據目錄；
         5. 使用合并後的數據集再次訓練。

         more: 循環往復，不斷迭代模型。
        :param per:
        :return:
        """
        # Path to the recur-output
        output_flag = int(time.time())
        output_dir_yes = os.path.join(self.label_src, f"_inner_{output_flag}/yes")
        output_dir_bad = os.path.join(self.label_src, f"_inner_{output_flag}/bad")
        # Initialize output directory
        os.makedirs(output_dir_yes, exist_ok=True)
        os.makedirs(output_dir_bad, exist_ok=True)
        # 導入上一輪迭代后的模型
        try:
            model = self._load_model()
        except RuntimeError as err:
            logger.error(err)
            return
        count = 0
        for index, fn in enumerate(img_fns := os.listdir(self.label_src)):
            # skip nested folders
            img_src = os.path.join(self.label_src, fn)
            if os.path.isfile(img_src):
                with open(img_src, "rb") as file:
                    data = file.read()
                with suppress(Exception):
                    img_dst = os.path.join(
                        output_dir_yes if ResNet.onnx_infer(model, data) else output_dir_bad, fn
                    )
                    shutil.move(img_src, img_dst)
                count += 1
            if index % 50 == 0:
                logger.debug(
                    f">> RECUR - label={self.label_name} progress=[{index}/{len(img_fns)}]"
                )
            if index and index == per:
                logger.debug(f">> Drop by per-batch - {index=}")
                break
        logger.success(f">> DONE - label={self.label_name} {count=}")
        # 清理临时文件
        for i in os.listdir(self.label_src):
            filename = os.path.join(self.label_src, i)
            if os.path.isdir(filename) and i.startswith("_inner_"):
                yes_size = os.path.getsize(os.path.join(filename, "yes"))
                bad_size = os.path.getsize(os.path.join(filename, "bad"))
                dir_size = os.path.getsize(filename)
                if not yes_size and not bad_size and not dir_size:
                    shutil.rmtree(filename)

    def run(
        self,
        train: Optional[bool] = True,
        auto_label: Optional[bool] = False,
        auto_edit: Optional[bool] = True,
        **kwargs,
    ):
        """

        :param auto_edit:
        :param train:
        :param auto_label:
        :param kwargs:
        :return:
        """
        if not os.path.isdir(self.label_src):
            logger.error(f"Missing sample data set - {self.label_name=}")
            return
        if self.label_src != self.label_dst:
            self._move_dataset()
        if train is True:
            self._train(**kwargs)
        if auto_label is True:
            self.recur()
        if auto_edit:
            webbrowser.open(REPO_URL)


class Monkey:
    @staticmethod
    def recur(task: Optional[str] = ""):
        task_list = []
        if task:
            task_list.append(task)
        for t in task_list:
            Workflow(t).recur()

    @staticmethod
    def run(task: str):
        Workflow(label_name=task).run()

    @staticmethod
    def edit():
        webbrowser.open(REPO_URL)


if __name__ == "__main__":
    from fire import Fire

    Fire(Monkey)
