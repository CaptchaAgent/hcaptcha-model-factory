# -*- coding: utf-8 -*-
# Time       : 2023/11/9 17:33
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import io
import os
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageFilter
from hcaptcha_challenger import ResNetControl

model_path = Path("model_zoo/attention_bus.onnx")
images_dir = Path("zip_dir/challenge_bus")


def load_model():
    net = cv2.dnn.readNetFromONNX(str(model_path))
    model = ResNetControl.from_pluggable_model(net)

    return model


def refresh_output_dir():
    yes_dir = images_dir.joinpath("yes")
    bad_dir = images_dir.joinpath("bad")

    for ck in [yes_dir, bad_dir]:
        shutil.rmtree(ck, ignore_errors=True)
        ck.mkdir(exist_ok=True, parents=True)

    return yes_dir, bad_dir


def classify(image_path: Path, model):
    image = Image.open(image_path)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.1))
    img_bytes_arr = io.BytesIO()
    image.save(img_bytes_arr, format="PNG")

    result = model.binary_classify(img_bytes_arr.getvalue())

    return result[0]


def run():
    model = load_model()
    yes_dir, bad_dir = refresh_output_dir()

    for image_name in os.listdir(images_dir):
        image_path = images_dir.joinpath(image_name)
        if image_path.is_file():
            if classify(image_path, model) is True:
                to_path = yes_dir.joinpath(image_name)
            else:
                to_path = bad_dir.joinpath(image_name)

            shutil.copyfile(image_path, to_path)


if __name__ == '__main__':
    run()
