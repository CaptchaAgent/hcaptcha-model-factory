# -*- coding: utf-8 -*-
# Time       : 2023/9/27 15:28
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os
import sys
from pathlib import Path

from hcaptcha_challenger import install
from hcaptcha_challenger.components.yolo_mocker import CcYOLO

install(upgrade=True)

# model_name = "burl_head_of_the_lion_2309_yolov8s.onnx"
model_name = "head_of_the_animal_2310_yolov8s.onnx"
images_dir = ""

this_dir = Path(__file__).parent
output_dir = this_dir.joinpath("yolo_mocker")

if isinstance(images_dir, str):
    images_dir = Path(images_dir)
images_dir = images_dir.absolute()


def run():
    ccy = CcYOLO(model_name, images_dir, output_dir)
    ccy.spawn()

    if "win32" in sys.platform:
        os.startfile(ccy.output_dir)


if __name__ == "__main__":
    run()
