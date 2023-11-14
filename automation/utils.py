import os
import shutil
import zipfile
from pathlib import Path
from typing import List

from hcaptcha_challenger import install, ModelHub
from loguru import logger


def parse_stander_model(modelhub: ModelHub, task_name: str) -> int | None:
    label = task_name.replace("_", " ")

    oxn: str = modelhub.label_alias.get(label, "")
    oxn = oxn.replace(".onnx", "").replace(task_name, "")

    if oxn.isdigit():
        return int(oxn) + 1


def parse_nested_model(modelhub: ModelHub, task_name: str, nested_prompt: str) -> int | None:
    nested_models: List[str] = modelhub.nested_categories.get(nested_prompt, [])
    if not isinstance(nested_models, list):
        if nested_models:
            raise TypeError(
                f"NestedTypeError ({nested_prompt}) 的模型映射列表应该是个 List[str] 类型，但实际上是 {type(nested_models)}"
            )
        nested_models = []

    for i, model_name in enumerate(nested_models):
        model_name = model_name.replace(".onnx", "").replace(task_name, "")
        if model_name.isdigit():
            return int(model_name) + 1


def gen_archive_version(task_name: str, nested_prompt: str = "") -> int:
    install(upgrade=True)
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    if not nested_prompt:
        v = parse_stander_model(modelhub, task_name)
    else:
        is_new = nested_prompt not in modelhub.nested_categories
        logger.info("Parse Nested Prompt", is_new=is_new, prompt=nested_prompt)
        for i in modelhub.nested_categories.get(nested_prompt, []):
            logger.info("Parse Nested Categories", model_sequence=i, prompt=nested_prompt)
        v = parse_nested_model(modelhub, task_name, nested_prompt)

    return v or 2309


def zip_dataset(task_name: str, binary_dir: Path, output_dir: Path):
    """

    Args:
        task_name:
        binary_dir: images_dir
        output_dir: zip_dir

    Returns:

    """
    output_dir.mkdir(exist_ok=True, parents=True)

    zip_path = output_dir.joinpath(f"{task_name}.zip")
    if zip_path.exists():
        shutil.rmtree(zip_path, ignore_errors=True)

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for root, dirs, files in os.walk(binary_dir):
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
