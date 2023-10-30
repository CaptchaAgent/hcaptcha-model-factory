# -*- coding: utf-8 -*-
# Time       : 2023/9/25 13:33
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

from hcaptcha_challenger import prompt2task, ModelHub, diagnose_task, install

CELL_TEMPLATE = """
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
GITHUB_TOKEN = "{github_token}"
task_name = "{task_name}"
onnx_archive_name = "{onnx_archive_name}"
NESTED_PROMPT = "{nested_prompt}"
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"""

NOTEBOOK = "https://colab.research.google.com/github/captcha-challenger/hcaptcha-model-factory/blob/main/automation/roboflow_resnet.ipynb"


def zip_dataset(prompt: str):
    prompt = prompt.replace("_", " ")
    task_name = prompt2task(prompt)

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
    return task_name


def print_quick_start_info(task_name: str, nested_prompt: str = ""):
    """
    task_name: like natural_landscape, nested_largest_tiger
    """
    diagnose_task(task_name)

    if task_name.startswith("nested_") and not nested_prompt:
        raise ValueError("生成嵌套类型模版需要提供其配对的提示词")

    install(upgrade=True)
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    label = task_name.replace("_", " ")

    onnx_archive_name = ""

    if not nested_prompt:
        if onnx_archive := modelhub.label_alias.get(label):
            oan = onnx_archive.replace(".onnx", "")
            v = ""
            for char in reversed(oan):
                if char.isdigit():
                    v = char + v
                else:
                    break
            if v and v.isdigit():
                v = int(v) + 1
                onnx_archive_name = f"{task_name}{str(v)}"
        else:
            onnx_archive_name = f"{task_name}2309"

    else:
        for i in modelhub.nested_categories.get(nested_prompt, []):
            print(f"{nested_prompt} => {i}")

        if nested_models := modelhub.nested_categories.get(nested_prompt, []):
            if not isinstance(nested_models, list):
                if nested_models:
                    raise TypeError(
                        f"NestedTypeError ({nested_prompt}) 的模型映射列表应该是个 List[str] 类型，但实际上是 {type(nested_models)}"
                    )
                nested_models = []
            v = ""
            for i, model_name in enumerate(nested_models):
                filter_chars = [".onnx", task_name]
                for fc in filter_chars:
                    model_name = model_name.replace(fc, "")
                if not model_name.isdigit():
                    continue
                else:
                    v = model_name
                    break
            if v and v.isdigit():
                v = int(v) + 1
                onnx_archive_name = f"{task_name}{str(v)}"
        else:
            onnx_archive_name = f"{task_name}2309"

    _t = CELL_TEMPLATE.format(
        github_token=os.getenv("GITHUB_TOKEN", ""),
        task_name=task_name,
        onnx_archive_name=onnx_archive_name,
        nested_prompt=nested_prompt,
    )

    print(_t)


def run():
    prompt = "nested_largest_squirrel"

    # 生成嵌套类型模版需要提供其配对的提示词
    # the smallest animal
    # please click on the largest animal
    nested_prompt = "please click on the largest animal"

    # 压缩数据集
    tn = zip_dataset(prompt=prompt)

    # 打印配置模版
    print_quick_start_info(task_name=tn, nested_prompt=nested_prompt)

    import webbrowser

    webbrowser.open(NOTEBOOK)


if __name__ == "__main__":
    run()
