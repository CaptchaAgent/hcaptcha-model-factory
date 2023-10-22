# -*- coding: utf-8 -*-
# Time       : 2023/9/27 15:28
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import onnxruntime
from hcaptcha_challenger import install, YOLOv8
from hcaptcha_challenger.components.yolo_mocker import CcYOLO
from hcaptcha_challenger.onnx.modelhub import request_resource

install(upgrade=True)


class CbYOLO(CcYOLO):
    def __init__(self, model_name: str, images_absolute_dir: Path, this_dir: Path, classes=None):
        super().__init__(model_name, images_absolute_dir, this_dir)
        self.classes = classes

    def get_model(self) -> YOLOv8 | None:
        classes = self.modelhub.ashes_of_war.get(self.model_name)
        if not classes:
            if not self.classes:
                raise AttributeError(f"Model name not found - {self.model_name=}")
            print(f">> Match model - {self.model_name=}")
            model_path = Path(self.model_name)
            if not model_path.exists():
                request_resource(self.model_url + self.model_name, model_path)
            try:
                session = onnxruntime.InferenceSession(
                    model_path, providers=onnxruntime.get_available_providers()
                )
                detector = YOLOv8.from_pluggable_model(session, self.classes)
            except Exception as err:
                print(err)
                shutil.rmtree(model_path, ignore_errors=True)
            else:
                return detector


def run():
    # model_name = "burl_head_of_the_lion_2309_yolov8s.onnx"
    model_name = "head_of_the_animal_turtle_2309_yolov8s.onnx"
    classes = ["animal-head"]
    images_dir = r"zip_dir/click_on_the_turtle_s_head_default"

    this_dir = Path(__file__).parent
    output_dir = this_dir.joinpath("yolo_mocker")

    if isinstance(images_dir, str):
        images_dir = Path(images_dir)
    images_dir = images_dir.absolute()

    ccy = CbYOLO(model_name, images_dir, output_dir, classes)
    ccy.spawn()

    if "win32" in sys.platform:
        os.startfile(ccy.output_dir)


if __name__ == "__main__":
    run()
