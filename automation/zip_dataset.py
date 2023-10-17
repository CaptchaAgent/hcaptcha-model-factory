# -*- coding: utf-8 -*-
# Time       : 2023/9/25 13:33
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import shutil
import webbrowser
import zipfile
from pathlib import Path

import hcaptcha_challenger as solver


def zip_dataset(prompt: str):
    prompt = prompt.replace("_", " ")
    task_name = solver.prompt2task(prompt)

    project_dir = Path(__file__).parent.parent
    images_dir = project_dir.joinpath("database2309", task_name)
    zip_dir = Path(__file__).parent.joinpath("zip_dir")
    zip_dir.mkdir(exist_ok=True)

    zip_path = zip_dir.joinpath(f"{task_name}.zip")
    if zip_path.exists():
        shutil.rmtree(zip_path, ignore_errors=True)

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for root, dirs, files in os.walk(images_dir):
            tp = Path(root)
            if task_name not in tp.parent.name:
                continue
            if root.endswith("yes"):
                for file in files:
                    zip_file.write(os.path.join(root, file), f"yes/{file}")
            elif root.endswith("bad"):
                for file in files:
                    zip_file.write(os.path.join(root, file), f"bad/{file}")

    print(f">> OUTPUT - {zip_path=}")


zip_dataset(prompt="nested_largest_raccoon")
webbrowser.open(
    "https://colab.research.google.com/github/captcha-challenger/hcaptcha-model-factory/blob/main/automation/roboflow_resnet.ipynb"
)
