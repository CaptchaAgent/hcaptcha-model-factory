# -*- coding: utf-8 -*-
# Time       : 2023/9/24 15:07
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hcaptcha_challenger as solver


@dataclass
class ContinueLabeling:
    prompt: str
    images_dir: Path = field(default=Path)
    model_path: str = field(default=str)

    task_name: str = field(default=str)
    _images: List[Path] = field(default_factory=list)

    def __post_init__(self):
        self._images = [
            self.images_dir.joinpath(img_name)
            for img_name in os.listdir(self.images_dir)
            if img_name.endswith(".png")
        ]

    @classmethod
    def from_prompt(cls, prompt: str, from_dir: Path, model_dir: Path | None = None, **kwargs):
        prompt = prompt.replace("_", " ")

        # Select: model name
        task_name = solver.prompt2task(prompt)

        # Match: image database
        images_dir = from_dir.joinpath(task_name)
        if not images_dir.exists():
            raise FileNotFoundError(f"NOT Found Images_dir - {images_dir=}")

        # Match: local model
        model_path = ""
        if model_dir and isinstance(model_dir, Path):
            model_path = model_dir.joinpath(task_name, f"{task_name}.onnx")
            if not model_path.exists():
                print("NOT Found Model_Path, switch to remote model")
            else:
                model_path = str(model_path)

        return cls(
            prompt=prompt,
            task_name=task_name,
            images_dir=images_dir,
            model_path=model_path,
            **kwargs,
        )

    def mkdir(self) -> Tuple[Path, Path]:
        __formats = ("%Y-%m-%d %H:%M:%S.%f", "%Y%m%d%H%M")
        now = datetime.strptime(str(datetime.now()), __formats[0]).strftime(__formats[1])
        yes_dir = self.images_dir.joinpath(now, "yes")
        bad_dir = self.images_dir.joinpath(now, "bad")
        yes_dir.mkdir(parents=True, exist_ok=True)
        bad_dir.mkdir(parents=True, exist_ok=True)

        return yes_dir, bad_dir

    def execute(self):
        if not self._images:
            sys.exit()
        if self.model_path == "":
            solver.install(upgrade=True)

        yes_dir, bad_dir = self.mkdir()

        classifier = solver.BinaryClassifier()
        results = classifier.execute(self.prompt, self._images, model_path=self.model_path)
        for i, result in enumerate(results):
            if result is True:
                shutil.move(self._images[i], yes_dir)
            elif result is False:
                shutil.move(self._images[i], bad_dir)

        if "win32" in sys.platform:
            os.startfile(self.images_dir)


def run(prompt: str):
    project_dir = Path(__file__).parent.parent
    from_dir = project_dir.joinpath("database2309")
    model_dir = project_dir.joinpath("model")

    cl = ContinueLabeling.from_prompt(prompt, from_dir, model_dir)
    cl.execute()


if __name__ == "__main__":
    run(prompt="something_you_can_eat")
