# -*- coding: utf-8 -*-
# Time       : 2023/9/26 14:03
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import yaml
from github import Github, Auth
from github.GitReleaseAsset import GitReleaseAsset
from github.Repository import Repository
from hcaptcha_challenger import ModelHub
from hcaptcha_challenger.onnx.modelhub import request_resource
from loguru import logger

if not os.getenv("GITHUB_TOKEN"):
    logger.warning("Skip model deployment, miss GITHUB TOKEN")
    sys.exit()


@dataclass
class Objects:
    branches: Dict[str, Any]
    circle_seg: str
    nested_categories: Dict[str, List[str]]
    ashes_of_war: Dict[str, Any]
    label_alias: Dict[str, Any]

    @classmethod
    def from_modelhub(cls, modelhub: ModelHub):
        data = yaml.safe_load(modelhub.objects_path.read_text(encoding="utf8"))
        return cls(
            **{
                key: (data[key] if val.default == val.empty else data.get(key, val.default))
                for key, val in inspect.signature(cls).parameters.items()
            }
        )

    def to_yaml(self, path: Path | None = None):
        path = path or Path("objects-tmp.yaml")
        with open(path, "w", encoding="utf8") as file:
            yaml.safe_dump(self.__dict__, file, sort_keys=False, allow_unicode=True)
        return path

    @staticmethod
    def to_asset(repo: Repository, data_tmp_path: Path, message: str = ""):
        content = data_tmp_path.read_bytes()
        message = message or f"Automated deployment @ utc {datetime.utcnow()}"
        remote_path = "src/objects.yaml"
        sha = repo.get_contents(path=remote_path).sha
        return repo.update_file(
            branch="main", path=remote_path, message=message, content=content, sha=sha
        )


def upgrade_objects(modelhub: ModelHub):
    objects_url = (
        "https://raw.githubusercontent.com/QIN2DIM/hcaptcha-challenger/main/src/objects.yaml"
    )
    request_resource(objects_url, modelhub.objects_path)


class Annotator:
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    repo = Github(auth=auth).get_repo("QIN2DIM/hcaptcha-challenger")

    def __init__(self, asset_id: int, matched_label: str = ""):
        self._asset_id = asset_id
        self._matched_label = matched_label

        self._asset: GitReleaseAsset | None = None

        self.modelhub = ModelHub.from_github_repo()

        upgrade_objects(modelhub=self.modelhub)

        self.modelhub.parse_objects()

        self.data: Objects = Objects.from_modelhub(modelhub=self.modelhub)

    @property
    def asset(self):
        if not self._asset:
            self._asset = self.repo.get_release_asset(self._asset_id)
        return self._asset

    @staticmethod
    def parse_resnet_label(asset_name: str) -> str | None:
        """
        asset_name: dog2312.onnx chess_piece2309.onnx
        """
        onnx_archive = asset_name.replace(".onnx", "")
        i_end = -1
        for i, s in enumerate(onnx_archive):
            if s.isdigit():
                i_end = i
                break
        label = onnx_archive[:i_end]
        label = label.replace("_", " ")
        return label

    def handle_resnet_objects(self):
        onnx_archive = self.asset.name.replace(".onnx", "")
        matched_label = self._matched_label or self.parse_resnet_label(self.asset.name)
        old_onnx_archive = self.modelhub.label_alias.get(matched_label)

        # Match: create new case
        if not old_onnx_archive:
            self.data.label_alias[onnx_archive] = {"en": [matched_label]}
        # Match: update old case
        else:
            i18n_mapping = self.data.label_alias[old_onnx_archive].copy()
            del self.data.label_alias[old_onnx_archive]
            self.data.label_alias[onnx_archive] = i18n_mapping

    def handle_yolov8_objects(self):
        pass

    def flush_remote_objects(self):
        data_tmp_path = self.data.to_yaml()

        res = self.data.to_asset(
            self.repo,
            message=f"ci(annotator): update model `{self.asset.name}`",
            data_tmp_path=data_tmp_path,
        )

        logger.success(f"upgrade objects", response=res)

        os.remove(data_tmp_path)

    def execute(self):
        logger.debug(f"capture asset", name=self.asset.name, url=self.asset.browser_download_url)

        # Match: ResNet MoE models
        if "yolov8" not in self.asset.name:
            self.handle_resnet_objects()
        else:
            self.handle_yolov8_objects()
            return

        self.flush_remote_objects()
