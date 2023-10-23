# -*- coding: utf-8 -*-
# Time       : 2023/10/20 17:28
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description: zero-shot image classification
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from hcaptcha_challenger import split_prompt_message, label_cleaning
from tqdm import tqdm

project_dir = Path(__file__).parent.parent
db_dir = project_dir.joinpath("database2309")


@dataclass
class AutoLabeling:
    positive_labels: List[str] = field(default_factory=list)
    candidate_labels: List[str] = field(default_factory=list)
    images_dir: Path = field(default=Path)
    pending_tasks: List[Path] = field(default_factory=list)

    checkpoint = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    # checkpoint = "QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_224to336"

    output_dir: Path = None

    def load_zero_shot_model(self):
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        task = "zero-shot-image-classification"

        detector = pipeline(task=task, model=self.checkpoint, device=device, batch_size=8)

        return detector

    @classmethod
    def from_prompt(cls, positive_labels: List[str], candidate_labels: List[str], images_dir: Path):
        images_dir.mkdir(parents=True, exist_ok=True)

        pending_tasks: List[Path] = []
        for image_name in os.listdir(images_dir):
            image_path = images_dir.joinpath(image_name)
            if image_path.is_file():
                pending_tasks.append(image_path)

        return cls(
            positive_labels=positive_labels,
            candidate_labels=candidate_labels,
            images_dir=images_dir,
            pending_tasks=pending_tasks,
        )

    def valid(self):
        if not self.pending_tasks:
            print("No pending tasks")
            return
        if len(self.candidate_labels) <= 2:
            print(f">> Please enter at least three class names - {self.candidate_labels=}")
            return

        return True

    def mkdir(self) -> Tuple[Path, Path]:
        __formats = ("%Y-%m-%d %H:%M:%S.%f", "%Y%m%d%H%M")
        now = datetime.strptime(str(datetime.now()), __formats[0]).strftime(__formats[1])
        yes_dir = self.images_dir.joinpath(now, "yes")
        bad_dir = self.images_dir.joinpath(now, "bad")
        yes_dir.mkdir(parents=True, exist_ok=True)
        bad_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = yes_dir.parent

        return yes_dir, bad_dir

    def execute(self, limit: int | str = None):
        if not self.valid():
            return

        # Format datafolder
        yes_dir, bad_dir = self.mkdir()

        # Load zero-shot model
        detector = self.load_zero_shot_model()

        total = len(self.pending_tasks)
        desc_in = f'"{self.checkpoint}/{self.images_dir.name}"'
        if isinstance(limit, str) and limit == "all":
            limit = total
        else:
            limit = limit or total

        with tqdm(total=total, desc=f"Labeling | {desc_in}") as progress:
            for image_path in self.pending_tasks[:limit]:
                image = Image.open(image_path)

                # Binary Image classification
                predictions = detector(image, candidate_labels=self.candidate_labels)

                # Move positive cases to yes/
                # Move negative cases to bad/
                if predictions[0]["label"] in self.positive_labels:
                    output_path = yes_dir.joinpath(image_path.name)
                else:
                    output_path = bad_dir.joinpath(image_path.name)
                shutil.move(image_path, output_path)

                progress.update(1)


@dataclass
class DataGroup:
    positive_labels: List[str] | str
    joined_dirs: List[str]
    negative_labels: List[str]

    def __post_init__(self):
        if isinstance(self.positive_labels, str):
            self.positive_labels = [self.positive_labels]

    @property
    def input_dir(self):
        return db_dir.joinpath(*self.joined_dirs).absolute()

    def auto_labeling(self, **kwargs):
        pls = []
        for pl in self.positive_labels:
            pl = pl.replace("_", " ")
            pl = split_prompt_message(label_cleaning(pl), "en")
            pls.append(pl)

        candidate_labels = pls.copy()

        if isinstance(self.negative_labels, list) and len(self.negative_labels) != 0:
            candidate_labels.extend(self.negative_labels)

        al = AutoLabeling.from_prompt(pls, candidate_labels, self.input_dir)
        al.execute(limit=kwargs.get("limit"))

        return al


def edit_in_the_common_cases():
    # prompt to negative labels
    # input_dir = /[Project_dir]/database2309/*[joined_dirs]

    dg = DataGroup(
        positive_labels=["helicopter", "excavator"],
        joined_dirs=["motorized_machine"],
        negative_labels=["laptop", "chess", "plant", "natural landscape", "mountain"],
    )
    nox = dg.auto_labeling(limit=1)

    if "win32" in sys.platform and nox.output_dir:
        os.startfile(nox.output_dir)


if __name__ == "__main__":
    edit_in_the_common_cases()
