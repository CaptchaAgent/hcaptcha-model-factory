# -*- coding: utf-8 -*-
# Time       : 2023/8/21 0:15
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
# https://github.com/QIN2DIM/hcaptcha-challenger/releases/edit/model
import os
import shutil
from pathlib import Path

from github import Auth
from github import Github
from github.GithubException import GithubException
from loguru import logger

from apis.scaffold import Scaffold

project_dir = Path(__file__).parent.parent

binary_dir = project_dir.joinpath("database2309")
factory_data_dir = project_dir.joinpath("data")
model_dir = project_dir.joinpath("model")

auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
repo = Github(auth=auth).get_repo("QIN2DIM/hcaptcha-challenger")
modelhub_title = "ONNX ModelHub"


def quick_train():
    # Copy the classified data into the dataset
    for task_name in focus_flags:
        to_dir = factory_data_dir.joinpath(task_name)
        shutil.rmtree(to_dir, ignore_errors=True)
        to_dir.mkdir(parents=True, exist_ok=True)

        # Flush dataset
        src_dir = binary_dir.joinpath(task_name)
        for hook in ["yes", "bad"]:
            shutil.copytree(src_dir.joinpath(hook), to_dir.joinpath(hook))

        # train
        Scaffold.train(task=task_name)


def quick_development():
    for label, onnx_archive in focus_flags.items():
        model_path = model_dir.joinpath(label, f"{label}.onnx")
        pending_onnx_path = model_dir.joinpath(label, f"{onnx_archive}.onnx")
        shutil.copy(model_path, pending_onnx_path)
        for release in repo.get_releases():
            if release.title != modelhub_title:
                continue
            try:
                res = release.upload_asset(path=str(pending_onnx_path))
            except GithubException as err:
                if err.status == 422:
                    logger.error(
                        f"The model file already exists, please manually replace the file with the same name - url={repo.releases_url}",
                        url=repo.releases_url,
                    )
            except Exception as err:
                logger.error(err)
            else:
                logger.success(
                    "Model file uploaded successfully", name=res.name, url=res.browser_download_url
                )


if __name__ == "__main__":
    # fmt:off
    focus_flags = {
        # "cat": "cat2310",
        # "mountain": "mountain2309",
        # "motorcycle": "motorcycle2309"
        # "excavator": "excavator2309"
        # "outdoor_gear": "outdoor_gear2309",
        # "vending_machine": "vending_machine2309",
        # "cup_of_hot_chocolate": "cup_of_hot_chocolate2309",
        # "fox": "fox2309",
        # "furniture": "furniture2309",
        # "helicopter": "helicopter2310",
        # "motor_vehicle": "motor_vehicle2309",
        # "chess_piece": "chess_piece2309"
        # "robot": "robot2309",
        "dog": "dog2309"
    }
    # fmt:on

    quick_train()
    quick_development()
