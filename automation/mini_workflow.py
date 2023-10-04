# https://github.com/QIN2DIM/hcaptcha-challenger/releases/edit/model
import os
import shutil
from pathlib import Path

from github import Auth
from github import Github
from github.GithubException import GithubException
from loguru import logger

from apis.scaffold import Scaffold
from automation.rolling_upgrade import rolling_upgrade

project_dir = Path(__file__).parent.parent

binary_dir = project_dir.joinpath("database2309")
factory_data_dir = project_dir.joinpath("data")
model_dir = project_dir.joinpath("model")


def quick_train():
    if not focus_flags:
        logger.warning("Skip model training, miss focus-flags")
        return

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


def quick_development() -> int | None:
    if not os.getenv("GITHUB_TOKEN"):
        logger.warning("Skip model deployment, miss GITHUB TOKEN")
        return

    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    repo = Github(auth=auth).get_repo("QIN2DIM/hcaptcha-challenger")
    modelhub_title = "ONNX ModelHub"

    for label, onnx_archive in focus_flags.items():
        model_path = model_dir.joinpath(label, f"{label}.onnx")
        pending_onnx_path = model_dir.joinpath(label, f"{onnx_archive}.onnx")
        shutil.copy(model_path, pending_onnx_path)
        for release in repo.get_releases():
            if release.title != modelhub_title:
                continue
            try:
                asset = release.upload_asset(path=str(pending_onnx_path))
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
                    f"Model file uploaded successfully - name={asset.name} url={asset.browser_download_url}"
                )
                return asset.id


if __name__ == "__main__":
    # After Annotating, edit `focus_flags`
    # - Copy from: `[PROJECT]/database2309/<diagnosed_label_name>`
    # - Paste to: `[PROJECT]/data/<diagnosed_label_name>`
    # - Output to: `[PROJECT]/model/<diagnosed_label_name>/<model_name[flag].onnx>`

    # fmt:off
    focus_flags = {
        # "<diagnosed_label_name>": "<model_name[flag]>"
        # "steak": "steak2309",
        # "sushi": "sushi2309",
        # "pizza": "pizza2301",
        # "main_course": "main_course2309",
        # "something_you_can_eat": "something_you_can_eat2314",
        # "trees": "trees2309",
        # "industrial_scene": "industrial_scene2311",
        # "steak":"steak2310",
        # "sushi": "sushi2310",
        # "car_without_a_roof": "car_without_a_roof2309",
        "gnome": "gnome2309"
    }
    # fmt:on

    quick_train()
    aid = quick_development()
    rolling_upgrade(aid)
