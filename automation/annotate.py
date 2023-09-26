# -*- coding: utf-8 -*-
# Time       : 2023/9/26 14:03
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import sys
from contextlib import suppress

import yaml
from github import Github, Auth
from hcaptcha_challenger import ModelHub
from loguru import logger


def upload_test_model():
    if not os.getenv("GITHUB_TOKEN"):
        logger.warning("Skip model deployment, miss GITHUB TOKEN")
        sys.exit()

    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    repo = Github(auth=auth).get_repo("QIN2DIM/test-action")
    release_title = "firetox v2"

    for release in repo.get_releases():
        if release.title != release_title:
            continue
        for p in ["dog2312.onnx", "chess_piece2309.onnx"]:
            with suppress(Exception):
                asset = release.upload_asset(path=p)
                print(asset.name, asset.id)


def parse_label(asset_name: str) -> str | None:
    """
    asset_name: dog2312.onnx chess_piece2309.onnx
    """
    onnx_archive = asset_name.replace(".onnx", "")
    i_end = -1
    for i, s in enumerate(onnx_archive):
        if s.digit():
            i_end = i
            break
    label = onnx_archive[:i_end]
    label = label.replace("_", " ")
    return label


def quick_annotation(asset_id: int, matched_label: str = ""):
    # 127749692 dog2312.onnx
    # 127749694 chess_piece2309.onnx
    if not os.getenv("GITHUB_TOKEN"):
        logger.warning("Skip model deployment, miss GITHUB TOKEN")
        return

    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    repo = Github(auth=auth).get_repo("QIN2DIM/test-action")

    asset = repo.get_release_asset(asset_id)
    logger.success(f">> CAPTURE - name={asset.name} url={asset.browser_download_url}")

    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    data = yaml.safe_load(modelhub.objects_path.read_text(encoding="utf8"))

    # if "yolov8" not in asset.name:
    #     onnx_archive = asset.name.replace(".onnx", "")
    #     matched_label = matched_label or parse_label(asset.name)
    #     old_onnx_archive = modelhub.label_alias.get(matched_label)
    #     if not old_onnx_archive:
    #         data["label_alias"][onnx_archive] = {"en": [matched_label]}
    #     else:
    #         i18n_mapping = data["label_alias"][old_onnx_archive].copy()
    #         del data["label_alias"][old_onnx_archive]
    #         data["label_alias"][onnx_archive] = i18n_mapping

    with open("format_objects.yaml", "w", encoding="utf8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


for aid in [127749692, 127749694]:
    quick_annotation(aid)
