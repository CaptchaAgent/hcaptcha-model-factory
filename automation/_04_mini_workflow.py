"""
# https://github.com/QIN2DIM/hcaptcha-challenger/releases/edit/model
"""
import os
import shutil
import time
import webbrowser
from enum import Enum
from pathlib import Path

from github import Github
from github.Auth import Token
from github.GithubException import GithubException
from hcaptcha_challenger import diagnose_task
from loguru import logger
from pydantic import BaseModel, field_validator, ValidationInfo, Field

from factories import ResNet
from utils import gen_archive_version, zip_dataset

GH_MODELHUB_TITLE = "ONNX ModelHub"
GH_DEPLOYMENT_REPO = "QIN2DIM/hcaptcha-challenger"

DIRNAME_TASK = "database2309"
DIRNAME_FACTORY_DATA = "data"
DIRNAME_FACTORY_MODEL = "model"

CELL_TEMPLATE = """
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
GITHUB_TOKEN = "{github_token}"
task_name = "{task_name}"
onnx_archive_name = "{archive_name}"
NESTED_PROMPT = "{nested_prompt}"
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"""

NOTEBOOK = (
    "https://colab.research.google.com/github/captcha-challenger/hcaptcha-model-factory/blob/main/automation"
    "/roboflow_resnet.ipynb"
)


class NestedPrompts(str, Enum):
    Unset = ""
    TheLargestAnimal = "the largest animal"
    TheSmallestAnimal = "the smallest animal"
    ElectronicDevice = "electronic device"
    MachineThatFlies = "machine that flies"
    TheSmallestAnimalSpecies = "the smallest animal species"
    TheLargestAnimalInRealLife = "the largest animal in real life"
    TheSmallestAnimalInRealLife = "the smallest animal in real life"


class ModelVersion(int, Enum):
    UNSET = 0
    DEFAULT = 2309


class WorkFlow(BaseModel):
    project_dir: Path
    task_name: str
    nested_prompt: NestedPrompts | str = Field(default=NestedPrompts.Unset, validate_default=True)
    archive_version: ModelVersion | int = Field(default=ModelVersion.UNSET)

    binary_dir: Path = Field(default=Path)
    factory_data_dir: Path = Field(default=Path)
    factory_model_dir: Path = Field(default=Path)

    train__: bool = Field(default=False)
    deploy__: bool = Field(default=False)
    rolling_upgrade__: bool = Field(default=False)
    gh_token: str | None = os.getenv("GITHUB_TOKEN")

    @field_validator("task_name")
    def check_task(cls, v: str):
        if not v.strip():
            raise ValueError("Empty task_name passed in")
        return diagnose_task(v)

    @field_validator("nested_prompt")
    def check_nested_prompt(cls, v: str, info: ValidationInfo):
        v = v.strip()

        # ::RULE<1>::
        task_name = info.data["task_name"]
        if task_name.startswith("nested_") and not v:
            raise ValueError("Nested model requires binding prompt, but the nested_prompt is null")
        if not task_name.startswith("nested_") and v:
            v = NestedPrompts.Unset

        # ::RULE<2>::
        t2p = {
            "nested_smallest_": "smallest",
            "nested_largest_": "largest",
            "nested_electronic_device": "electronic device",
        }
        for t, p in t2p.items():
            if task_name.startswith(t) and p not in v:
                raise ValueError(
                    f"Please check if the task_name and prompt match - "
                    f"\ntask_name: {task_name} "
                    f"\nyour nested prompt: {v} "
                    f"\nforce_rule: {p}"
                )

        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.binary_dir = self.project_dir.joinpath(DIRNAME_TASK)
        self.factory_data_dir = self.project_dir.joinpath(DIRNAME_FACTORY_DATA)
        self.factory_model_dir = self.project_dir.joinpath(DIRNAME_FACTORY_MODEL)

        self.binary_dir.mkdir(parents=True, exist_ok=True)
        if not (dataset_dir := self.binary_dir.joinpath(self.task_name)).exists():
            raise FileNotFoundError(
                f"We can't find the dataset that the task_name points - task_name={self.task_name}"
                f"\nDataset should be in this directory {dataset_dir}"
            )

        if not self.archive_version:
            self.archive_version = gen_archive_version(self.task_name, self.nested_prompt)

    def _train_model(self, **kwargs):
        # Shuffle configuration
        to_dir = self.factory_data_dir.joinpath(self.task_name)
        shutil.rmtree(to_dir, ignore_errors=True)
        to_dir.mkdir(parents=True, exist_ok=True)

        # Copy dataset
        src_dir = self.binary_dir.joinpath(self.task_name)
        for hook in ["yes", "bad"]:
            shutil.copytree(src_dir.joinpath(hook), to_dir.joinpath(hook))

        # train
        model = ResNet(
            task_name=self.task_name,
            epochs=kwargs.get("epochs"),  # default to 200
            batch_size=kwargs.get("batch_size"),  # default to 4
            dir_dataset=str(self.factory_data_dir),
            dir_model=str(self.factory_model_dir),
        )
        model.train()
        model.conv_pth2onnx(verbose=False)

    def _deploy_model(self, version: int | None = None) -> int | None:
        version = version or self.archive_version
        model_archive_dir = self.factory_model_dir.joinpath(self.task_name)

        model_path_raw = model_archive_dir.joinpath(f"{self.task_name}.onnx")
        model_path_version = model_archive_dir.joinpath(f"{self.task_name}{version}.onnx")
        shutil.copy(model_path_raw, model_path_version)

        repo = Github(auth=Token(self.gh_token)).get_repo(GH_DEPLOYMENT_REPO)

        for release in repo.get_releases():
            if release.title != GH_MODELHUB_TITLE:
                continue
            try:
                asset = release.upload_asset(path=str(model_path_version))
            except GithubException as err:
                if err.status == 422:
                    version += 1
                    logger.warning(
                        "The model file already exists, "
                        "Attempting to submit model via self-increasing archive_version",
                        url=repo.releases_url,
                        next_version=version,
                    )
                    if abs(version - self.archive_version) < 5:
                        time.sleep(3)
                        return self._deploy_model(version)
            except Exception as err:
                logger.error(err)
            else:
                self.archive_version = version
                logger.success(
                    f"Model file uploaded successfully",
                    name=asset.name,
                    url=asset.browser_download_url,
                )
                return asset.id

    def _rolling_upgrade(self, asset_id: int | None = None, matched_label: str = ""):
        """当上传 nested 模型时需要指定该模型绑定的嵌套类型"""

        def lookup_aid_by_task_name() -> int | None:
            archive_model_name = f"{self.task_name}{self.archive_version}"

            repo = Github(auth=Token(self.gh_token)).get_repo(GH_DEPLOYMENT_REPO)
            for release in repo.get_releases():
                if release.title != GH_MODELHUB_TITLE:
                    continue
                for asset in release.get_assets():
                    if not asset.name.startswith(archive_model_name):
                        continue
                    logger.success("lookup aid by task_name", name=asset.name, aid=asset.id)
                    return asset.id

        asset_id = asset_id or lookup_aid_by_task_name()
        if not asset_id:
            logger.error("Failed to get the correct resource ID", asset_id=asset_id)
            return

        from _annotator import Annotator

        try:
            annotator = Annotator(asset_id, matched_label=matched_label)
            annotator.execute()
            webbrowser.open(Annotator.repo.html_url)
        except Exception as err:
            logger.warning(err)

    def create(self, train: bool = True, deploy: bool = True, rolling_upgrade: bool = True):
        if (deploy or rolling_upgrade) and not self.gh_token:
            logger.warning("Skip model deployment", reason="miss env GITHUB_TOKEN")
            return

        self.train__ = train
        self.deploy__ = deploy
        self.rolling_upgrade__ = rolling_upgrade
        return self

    def to_colab(self):
        zip_dataset(
            task_name=self.task_name,
            binary_dir=self.binary_dir,
            output_dir=Path(__file__).parent.joinpath("zip_dir"),
        )
        print(self.archive_version)
        _t = CELL_TEMPLATE.format(
            github_token=os.getenv("GITHUB_TOKEN", ""),
            task_name=self.task_name,
            archive_name=f"{self.task_name}{self.archive_version}",
            nested_prompt=f"{self.nested_prompt}",
        )

        print(_t)
        print(f"Open In Colab -> {NOTEBOOK}")

    def run(self, **kwargs):
        if self.train__:
            logger.info("Preview", task_name=self.task_name)
            logger.info("Preview", model_archive_name=f"{self.task_name}{self.archive_version}")
            logger.info("Preview", nested_prompt=f"{self.nested_prompt}")
            try:
                input("--> PRESS ANY KEY TO START TRAINING")
                self._train_model(**kwargs)
            except KeyboardInterrupt:
                return

        asset_id: int | None = None
        if self.deploy__:
            asset_id = self._deploy_model()

        if self.rolling_upgrade__:
            self._rolling_upgrade(asset_id, self.nested_prompt)

        return self


workflow = WorkFlow(
    project_dir=Path(__file__).parent.parent,
    task_name="nested_smallest_lion",
    nested_prompt=NestedPrompts.TheSmallestAnimalInRealLife,
    archive_version=ModelVersion.UNSET,
)


def run():
    # 本地运行
    # workflow.create(train=True, deploy=True, rolling_upgrade=True).run()

    # 输出 Colab 启动配置
    workflow.to_colab()


if __name__ == "__main__":
    run()
