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

import cv2
import onnxruntime
from hcaptcha_challenger import install, YOLOv8
from hcaptcha_challenger.onnx.modelhub import request_resource
from tqdm import tqdm

install(upgrade=True)
model_url = "https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/"

this_dir = Path(__file__).parent
models_dir = this_dir.joinpath("tmp_models")
models_dir.mkdir(exist_ok=True)


def load_model(model_path, classes):
    session = onnxruntime.InferenceSession(
        model_path, providers=onnxruntime.get_available_providers()
    )
    detector = YOLOv8.from_pluggable_model(session, classes)
    return detector


def execute(input_dir: Path | str, model_name: str, model_path: Path, classes: list):
    def draw():
        alts = sorted(results, key=lambda x: x[-1])
        text, ps, pe, _ = alts[-1]
        image = cv2.imread(str(image_path))
        pt1 = int(ps[0]), int(ps[1])
        pt2 = int(pe[0]), int(pe[1])
        cv2.rectangle(image, pt1, pt2, (255, 0, 126), 2)
        cv2.imwrite(str(output_path), image)

    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    input_dir = input_dir.absolute()

    if not model_path.exists():
        request_resource(model_url + model_name, model_path)

    detector = load_model(model_path, classes)

    output_dir = this_dir.joinpath("yolo_mocker", input_dir.name)
    output_miss_dir = output_dir.joinpath("miss")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_miss_dir.mkdir(parents=True, exist_ok=True)

    pending_image_paths = []
    for image_name in os.listdir(input_dir):
        image_path = input_dir.joinpath(image_name)
        if image_path.is_file() and not output_dir.joinpath(image_name).exists():
            pending_image_paths.append(image_path)

    total = len(pending_image_paths)
    handle, miss = 0, 0

    with tqdm(total=total, desc=f"Labeling | ") as progress:
        for image_path in pending_image_paths:
            results = detector(image_path, shape_type="bounding_box")
            progress.update(1)
            if not results:
                output_miss_path = output_miss_dir.joinpath(image_path.name)
                shutil.copyfile(image_path, output_miss_path)
                miss += 1
                continue
            output_path = output_dir.joinpath(image_path.name)
            draw()
            handle += 1
    print(f">> Statistic - {total=} {handle=} {miss=}")

    return output_dir


def run():
    images_dir = r"zip_dir/please click on the lion's head/default"

    # model_name = "burl_head_of_the_lion_2309_yolov8s.onnx"
    # model_name = "head_of_the_animal_turtle_2309_yolov8s.onnx"
    # model_name = "head_of_the_animal_turtle_2309_yolov8s.onnx"
    # model_name = "head_of_the_meerkat_2311_yolov8n.onnx"
    model_name = "head_of_the_jaguar_mask_2309_yolov8n.onnx"
    classes = ["spiral_pattern"]

    output_dir = execute(
        input_dir=images_dir,
        model_name=model_name,
        model_path=models_dir.joinpath(model_name),
        classes=classes,
    )

    if "win32" in sys.platform:
        os.startfile(output_dir)


if __name__ == "__main__":
    run()
