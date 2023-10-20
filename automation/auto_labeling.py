# -*- coding: utf-8 -*-
# Time       : 2023/10/20 17:28
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description: zero-shot image classification
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from hcaptcha_challenger import split_prompt_message, prompt2task, label_cleaning
from tqdm import tqdm
from transformers import pipeline


@dataclass
class AutoLabeling:
    positive_label: str = field(default=str)
    candidate_labels: List[str] = field(default_factory=list)
    images_dir: Path = field(default=Path)
    pending_tasks: List[Path] = field(default_factory=list)

    checkpoint = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "zero-shot-image-classification"

    def load_zero_shot_model(self):
        detector = pipeline(task=self.task, model=self.checkpoint, device=self.device, batch_size=8)
        return detector

    @classmethod
    def from_prompt(cls, positive_label: str, candidate_labels: List[str], images_dir: Path):
        images_dir.mkdir(parents=True, exist_ok=True)

        pending_tasks: List[Path] = []
        for image_name in os.listdir(images_dir):
            image_path = images_dir.joinpath(image_name)
            if image_path.is_file():
                pending_tasks.append(image_path)

        return cls(
            positive_label=positive_label,
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

        return yes_dir, bad_dir

    def execute(self, limit: int = None):
        if not self.valid():
            return

        # Format datafolder
        yes_dir, bad_dir = self.mkdir()

        # Load zero-shot model
        detector = self.load_zero_shot_model()

        total = len(self.pending_tasks)
        desc_in = f'"{self.checkpoint}/{self.images_dir.name}"'
        limit = limit or total

        with tqdm(total=total, desc=f"Labeling | {desc_in}") as progress:
            for image_path in self.pending_tasks[:limit]:
                image = Image.open(image_path)

                # Binary Image classification
                predictions = detector(image, candidate_labels=self.candidate_labels)

                # Move positive cases to yes/
                # Move negative cases to bad/
                if predictions[0]["label"] == self.positive_label:
                    output_path = yes_dir.joinpath(image_path.name)
                else:
                    output_path = bad_dir.joinpath(image_path.name)
                shutil.move(image_path, output_path)

                progress.update(1)

        return yes_dir.parent


def run(prompt: str, negative_labels: List[str], **kwargs):
    prompt = prompt.replace("_", " ")

    task_name = prompt2task(prompt)

    project_dir = Path(__file__).parent.parent
    images_dir = project_dir.joinpath("database2309", task_name)

    positive_label = split_prompt_message(label_cleaning(prompt), "en")
    candidate_labels = [positive_label]
    if isinstance(negative_labels, list) and len(negative_labels) != 0:
        candidate_labels.extend(negative_labels)

    al = AutoLabeling.from_prompt(positive_label, candidate_labels, images_dir)
    output_dir = al.execute(limit=kwargs.get("limit"))

    if "win32" in sys.platform:
        os.startfile(output_dir)


if __name__ == "__main__":
    run(
        prompt="vr_headset", negative_labels=["phone", "keyboard", "drone", "3d printer"], limit=500
    )
