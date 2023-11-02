# -*- coding: utf-8 -*-
# Time       : 2023/10/24 5:39
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, NoReturn

import hcaptcha_challenger as solver
from PIL import Image
from hcaptcha_challenger import DataLake, ModelHub, ZeroShotImageClassifier, register_pipline
from tqdm import tqdm

from flow_card import flow_card, flow_card_nested_animal

solver.install(upgrade=True)


@dataclass
class SubStack:
    nested_name: str
    yes_seq: List[str]
    bad_seq: List[str]

    @classmethod
    def from_tnf(cls, name: str, yes: List[str], bad: List[str]):
        return cls(nested_name=name, yes_seq=yes, bad_seq=bad)

    @staticmethod
    def kt(x):
        return f"This is a photo of the {x}"

    def _offload(self, tag: str, dirname: str, tmp_case_dir: Path, *, to_dir: Path):
        if self.kt(tag) == dirname:
            logging.info(f"refactor - name={self.nested_name} {tag=}")
            for image_name in os.listdir(tmp_case_dir):
                image_path = tmp_case_dir.joinpath(image_name)
                shutil.copyfile(image_path, to_dir.joinpath(image_name))

    def transform(self, base_dir: Path):
        logging.info(f"Startup substack - {base_dir=}")
        yes_dir = base_dir.joinpath(self.nested_name, "yes")
        bad_dir = base_dir.joinpath(self.nested_name, "bad")
        yes_dir.mkdir(parents=True, exist_ok=True)
        bad_dir.mkdir(parents=True, exist_ok=True)

        # 将多目标分类结果二次移动到 yes/bad 的混合二分类终端
        logging.info("Moving substack")
        for dirname in os.listdir(base_dir):
            if dirname == self.nested_name:
                continue
            tmp_case_dir = base_dir.joinpath(dirname)
            for tag in self.yes_seq:
                self._offload(tag, dirname, tmp_case_dir, to_dir=yes_dir)
            for tag in self.bad_seq:
                self._offload(tag, dirname, tmp_case_dir, to_dir=bad_dir)


@dataclass
class AutoLabeling:
    """
    Example:
    ---

    1. Roughly observe the distribution of the dataset and design a DataLake for the challenge prompt.
        - ChallengePrompt: "Please click each image containing an off-road vehicle"
        - positive_labels --> ["off-road vehicle"]
        - negative_labels --> ["bicycle", "car"]

    2. You can design them in batches and save them as YAML files,
    which the classifier can read and automatically DataLake

    3. Note that positive_labels is a list, and you can specify multiple labels for this variable
    if the label pointed to by the prompt contains ambiguity。

    """

    input_dir: Path = field(default_factory=Path)
    pending_tasks: List[Path] = field(default_factory=list)
    tool: ZeroShotImageClassifier = field(default_factory=ZeroShotImageClassifier)

    output_dir: Path = field(default_factory=Path)

    limit: int = field(default=1)
    """
    By default, all pictures in the specified folder are classified and moved,
    Specifies the limit used to limit the number of images for the operation.
    """

    @classmethod
    def from_datalake(cls, dl: DataLake, **kwargs):
        if not isinstance(dl.joined_dirs, Path):
            raise TypeError(
                f"The dataset joined_dirs needs to be passed in for auto-labeling. - {dl.joined_dirs=}"
            )
        if not dl.joined_dirs.exists():
            raise ValueError(f"Specified dataset path does not exist - {dl.joined_dirs=}")

        input_dir = dl.joined_dirs
        pending_tasks = []
        for image_name in os.listdir(input_dir):
            image_path = input_dir.joinpath(image_name)
            if image_path.is_file():
                pending_tasks.append(image_path)

        if (limit := kwargs.get("limit")) is None:
            limit = len(pending_tasks)
        elif not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit should be a positive integer greater than zero. - {limit=}")

        tool = ZeroShotImageClassifier.from_datalake(dl)
        return cls(tool=tool, input_dir=input_dir, pending_tasks=pending_tasks, limit=limit)

    def mkdir(self, multi: bool = False) -> Tuple[Path, Path]:
        __formats = ("%Y-%m-%d %H:%M:%S.%f", "%Y%m%d%H%M")
        now = datetime.strptime(str(datetime.now()), __formats[0]).strftime(__formats[1])
        tmp_dir = self.input_dir.joinpath(now)
        yes_dir = tmp_dir.joinpath("yes")
        bad_dir = tmp_dir.joinpath("bad")
        yes_dir.mkdir(parents=True, exist_ok=True)
        bad_dir.mkdir(parents=True, exist_ok=True)

        if multi:
            for label in self.tool.candidate_labels:
                tmp_dir.joinpath(label).mkdir(parents=True, exist_ok=True)

        self.output_dir = tmp_dir

        return yes_dir, bad_dir

    def execute(self, model, substack: Dict[str, Dict[str, List[str]]] = None, **kwargs):
        if not self.pending_tasks:
            logging.info("No pending tasks")
            return

        multi = bool(substack)
        yes_dir, bad_dir = self.mkdir(multi=multi)

        desc_in = f'"{self.input_dir.parent.name}/{self.input_dir.name}"'
        total = len(self.pending_tasks)

        logging.info(f"load {self.tool.positive_labels=}")
        logging.info(f"load {self.tool.candidate_labels=}")

        with tqdm(total=total, desc=f"Labeling | {desc_in}") as progress:
            for image_path in self.pending_tasks[: self.limit]:
                # The label at position 0 is the highest scoring target
                image = Image.open(image_path)
                results = self.tool(model, image)

                # we're dealing with multi-classification tasks here
                trusted = results[0]["label"]
                if multi:
                    bk_path = self.output_dir.joinpath(trusted, image_path.name)
                    shutil.move(image_path, bk_path)
                elif trusted in self.tool.positive_labels:
                    shutil.move(image_path, yes_dir.joinpath(image_path.name))
                else:
                    shutil.move(image_path, bad_dir.joinpath(image_path.name))

                progress.update(1)

        if multi:
            # 遍历预分类的目录名，如果是在 yes-list，复制图片到 yes/，反之则复制到 bad/
            # 跳过未匹配的目录名
            logging.info("Multi-objective datasets being processed")
            for nested_name, tnf in substack.items():
                stk = SubStack.from_tnf(nested_name, yes=tnf["yes"], bad=tnf["bad"])
                stk.transform(base_dir=self.output_dir)


def check_card(pending_card: list) -> NoReturn | bool:
    require_keys = ["positive_labels", "negative_labels", "joined_dirs"]
    ok = True
    for i, card in enumerate(pending_card):
        for rk in require_keys:
            if rk not in card or not isinstance(card[rk], list):
                logging.error(f"card 缺少必要的键值对 - {i=} key={rk} {card=}")
                ok = False
            elif len(card[rk]) == 0:
                logging.error(f"card 的必要信息不能为空 - {i=} key={rk} {require_keys=}")
                ok = False
            elif len(card[rk]) < 2:
                logging.error(f"CLIP 模型在二分类任务上的准确率可能会比较糟糕 - {i=} ")
                ok = False

            for label in card[rk]:
                if not isinstance(label, str):
                    logging.error(
                        f"card 的 require_keys 的值必须是 List[str] 类型 - {i=} {label=} {type(label)=}"
                    )
                    ok = False
                if len(label) <= 1:
                    logging.error(f"这可能是一个异常输入 - {i=} {label=}")
                    ok = False

    if not ok:
        raise ValueError("card 存在致命的写法问题，请逐一排查后再启动程序！")
    return True


def run(suffix_filter: str, cards: list, base_dirname: str = "database2309"):
    if not suffix_filter:
        return

    logging.info("正在检查运行配置")
    check_card(cards)
    logging.info("运行配置准确无误！")

    images_dir = Path(__file__).parent.parent.joinpath(base_dirname)

    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    # Make sure you have torch and transformers installed and
    # the NVIDIA graphics card is available
    model = register_pipline(modelhub, fmt="transformers")

    for card in cards:
        # Filter out the task cards we care about
        if suffix_filter not in card["joined_dirs"]:
            continue
        # Generating a dataclass from serialized data
        dl = DataLake(
            positive_labels=card["positive_labels"],
            negative_labels=card["negative_labels"],
            joined_dirs=images_dir.joinpath(*card["joined_dirs"]),
        )
        # Starts an automatic labeling task
        al = AutoLabeling.from_datalake(dl)
        al.execute(model, **card)
        # Automatically open output directory
        if "win32" in sys.platform and al.output_dir.is_dir():
            os.startfile(al.output_dir)


if __name__ == "__main__":
    logging.info(f"Loading {len(flow_card)=}")
    logging.info(f"Loading {len(flow_card_nested_animal)=}")

    run("s1_bird", cards=flow_card_nested_animal)
