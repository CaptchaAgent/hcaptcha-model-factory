# -*- coding: utf-8 -*-
# Time       : 2023/9/26 14:03
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import inspect
import os
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import yaml
from github import Github, Auth
from github.GitReleaseAsset import GitReleaseAsset
from github.Repository import Repository
from hcaptcha_challenger import ModelHub, handle
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
    datalake: dict

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
        data = yaml.safe_dump(self.__dict__, sort_keys=False, allow_unicode=True)
        path.write_text(data, encoding="utf8", newline="\n")
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

    def handle_nested_objects(self, model_pending: str):
        """
        Match nested cases:
        - the largest animal
        - the smallest animal
        """
        bond_nested_prompt = handle(self._matched_label)
        if not bond_nested_prompt:
            raise ValueError("Nested model requires binding prompt")

        # nested_largest_dog2309.onnx nested_largest_elephant2309.onnx
        prefix_tag_pending = self.parse_resnet_label(model_pending)

        # Match: 已注册的嵌套类型（bond_nested_prompt）
        if nested_models := self.modelhub.nested_categories.get(bond_nested_prompt, []):
            # prompt已注册但被错误赋值
            if not isinstance(nested_models, list):
                # 如果存在确切的值，则返回错误
                if nested_models:
                    raise TypeError(
                        f"NestedTypeError ({bond_nested_prompt}) 的模型映射列表应该是个 List[str] 类型，但实际上是 {type(nested_models)}"
                    )
                # 如果prompt存在但未被赋有效值，则尝试恢复程序重建秩序
                nested_models = []
            # 查询 prompt 对应的模型匹配列表，更新「同项模型」的版本索引
            idx_old_points: List[int] = []
            for i, model_name in enumerate(nested_models):
                prefix_tag_in_the_slot = self.parse_resnet_label(model_name)
                if prefix_tag_in_the_slot == prefix_tag_pending:
                    idx_old_points.append(i)
            # 若 prompt 对应的模型匹配列表找不到「同项模型」更旧的版本，则直接插入新的模型
            for i in idx_old_points:
                nested_models.pop(i)
            nested_models.append(model_pending)
        # Match: 未注册的嵌套模型
        else:
            nested_models = [model_pending]

        # 恢复嵌套模型的上下文，更新模型索引
        self.data.nested_categories[bond_nested_prompt] = nested_models

    def flush_remote_objects(self):
        """
        导出 YAML 文件，上传到仓库
        """
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
        if "yolov8" in self.asset.name:
            return
        if "nested_" in self.asset.name:
            self.handle_nested_objects(self.asset.name)
        else:
            self.handle_resnet_objects()

        self.flush_remote_objects()


def rolling_upgrade(asset_id=None, matched_label: str = ""):
    """
    当上传 nested 模型时需要指定该模型绑定的嵌套类型
    """
    if not asset_id:
        return

    try:
        annotator = Annotator(asset_id, matched_label=matched_label)
        annotator.execute()
        webbrowser.open(Annotator.repo.html_url)
    except Exception as err:
        logger.warning(err)


def find_asset_id(name_prefix: str):
    """如果工作流在滚动更新前中断，可以通过此函数根据模型名前缀匹配到资源的 asset_id"""
    repo = Annotator.repo
    modelhub_title = "ONNX ModelHub"

    for release in repo.get_releases():
        if release.title != modelhub_title:
            continue
        for asset in release.get_assets():
            if not asset.name.startswith(name_prefix):
                continue
            print(asset.name, asset.id)
            break
