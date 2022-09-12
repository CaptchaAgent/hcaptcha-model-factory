import copy
import os
import random
import shutil
import time
import typing

import torch
import torchvision
import yaml
from loguru import logger
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from components.dataset import BinaryDataset
from components.losses import FocalLoss
from components.nn import ResNetMini
from components.utils import ToolBox
from .kernel import ModelFactory, Params

_ACTION_NAME = "ResNetMini"


class ResNet(ModelFactory):
    # {{< Training >}}
    BATCH_SIZE = 4
    EPOCHS = 200
    FOCAL_LOSS_GAMMA = 2.0
    LR = 0.001
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.6
    LR_WEIGHT_DECAY = 0.0005
    LR_MOMENTUM = 0.9
    LOSS_FN = Params.LOSS_FN_FOCAL  # "focal" | "cross_entropy"
    OPT_FLAG = Params.OPTIMIZER_ADAM  # "sgd" | "adam"

    # {{< Checkpoint >}}
    SAVE_INTERVAL = 50
    LOG_INTERVAL = 100

    # {{< Dataset Spliter >}}
    RATIO_TRAIN = 0.8
    RATIO_VAL = 0.2
    RATIO_TEST = 0.0

    USE_BEST_MODEL = False
    USE_IMG_TRANSFORM = True

    def _build_env(self):
        dir_last_work = os.path.join(self._dir_model, self._task_name, f"{int(time.time())}")
        self._dir_model = os.path.join(self._dir_model, self._task_name)
        os.makedirs(self._dir_model, exist_ok=True)

        # Archive the existing model to the `dir_last_work[TIMESTAMP]` directory
        # Delete empty archive-folders `dir_last_work[TIMESTAMP]`
        for sickle in os.listdir(self._dir_model):
            path = os.path.join(self._dir_model, sickle)
            if os.path.isfile(path):
                if sickle.endswith(".onnx") or sickle.endswith(".pth"):
                    os.makedirs(dir_last_work, exist_ok=True)
                    shutil.move(path, dir_last_work)
            elif os.path.isdir(path) and sickle.isdigit() and not os.listdir(path):
                shutil.rmtree(path, ignore_errors=True)

        # Init workspace
        self._dir_dataset = os.path.join(self._dir_dataset, self._task_name)

        # Check && Split Dataset(though yaml)
        dict_dataset_all = {
            "task_name": self._task_name,
            "task_type": "image_label_binary",
            "format": {"img_size": 64},
            "data": [],
        }
        dict_dataset_train = copy.deepcopy(dict_dataset_all)
        dict_dataset_val = copy.deepcopy(dict_dataset_all)
        dict_dataset_test = copy.deepcopy(dict_dataset_all)

        dir_dataset_yes = os.path.join(self._dir_dataset, self.FLAG_POSITIVE)
        dir_dataset_bad = os.path.join(self._dir_dataset, self.FLAG_NEGATIVE)
        for hook in [dir_dataset_yes, dir_dataset_bad]:
            if not os.path.isdir(hook):
                raise FileNotFoundError(
                    f"The structure of the dataset is incomplete | dir={os.path.dirname(hook)}"
                )
            if (hook_size := len(os.listdir(hook))) < 2:
                raise ResourceWarning(
                    "The data set size does not reach the standard | "
                    f"dir={hook} threshold=[{hook_size}/{2}]"
                )
            for fn in os.listdir(hook):
                src_path_img = os.path.join(hook, fn)
                if not os.path.isfile(src_path_img):
                    raise FileNotFoundError(f"{src_path_img} is not a file")
                if not ToolBox.is_image(src_path_img):
                    raise ValueError(f"{src_path_img} is not a image file")

                image_info = {"fname": src_path_img, "label": 1 if hook == dir_dataset_yes else 0}
                dict_dataset_all["data"].append(image_info)
                operator = random.uniform(0, 1)
                if operator < self.RATIO_TRAIN:
                    dict_dataset_train["data"].append(image_info)
                elif operator < self.RATIO_TRAIN + self.RATIO_VAL:
                    dict_dataset_val["data"].append(image_info)
                else:
                    dict_dataset_test["data"].append(image_info)

        # Save dataset info
        path_yaml_dataset_all = os.path.join(self._dir_dataset, self.FILENAME_YAML_ALL)
        path_yaml_dataset_train = os.path.join(self._dir_dataset, self.FILENAME_YAML_TRAIN)
        path_yaml_dataset_val = os.path.join(self._dir_dataset, self.FILENAME_YAML_VAL)
        path_yaml_dataset_test = os.path.join(self._dir_dataset, self.FILENAME_YAML_TEST)
        with open(path_yaml_dataset_all, "w") as file:
            yaml.dump(dict_dataset_all, file)
        with open(path_yaml_dataset_train, "w") as file:
            yaml.dump(dict_dataset_train, file)
        with open(path_yaml_dataset_val, "w") as file:
            yaml.dump(dict_dataset_val, file)
        with open(path_yaml_dataset_test, "w") as file:
            yaml.dump(dict_dataset_test, file)

    @staticmethod
    def _get_dataset(dir_dataset: str, flag: str, with_augment: bool) -> Dataset:
        # transform with data augmentation
        img_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomChoice(
                    [
                        torchvision.transforms.RandomRotation(30),
                        torchvision.transforms.GaussianBlur(3),
                        torchvision.transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                        ),
                        torchvision.transforms.RandomResizedCrop(size=64, scale=(0.8, 1.2)),
                    ]
                ),
                torchvision.transforms.Resize(size=64),
                torchvision.transforms.ToTensor(),
            ]
        )

        # transform without data augmentation
        img_transform_no_augment = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=64), torchvision.transforms.ToTensor()]
        )

        transform_ = img_transform if with_augment else img_transform_no_augment
        return BinaryDataset(root=dir_dataset, flag=flag, transform=transform_)

    def _get_optimizer(self, model: nn.Module, opt: str) -> Optimizer:
        opt_map = {
            Params.OPTIMIZER_SGD: torch.optim.SGD(
                model.parameters(),
                lr=self.LR,
                momentum=self.LR_MOMENTUM,
                weight_decay=self.LR_WEIGHT_DECAY,
            ),
            Params.OPTIMIZER_ADAM: torch.optim.Adam(
                model.parameters(), lr=self.LR, weight_decay=self.LR_WEIGHT_DECAY
            ),
        }
        optimizer = opt_map.get(opt)
        if not optimizer:
            raise ValueError("optimizer must be sgd or adam")
        return optimizer

    def _get_criterion(self, loss_fn: str) -> typing.Union[FocalLoss, CrossEntropyLoss]:
        criterion_map = {
            Params.LOSS_FN_FOCAL: FocalLoss(gamma=self.FOCAL_LOSS_GAMMA),
            Params.LOSS_FN_CROSS_ENTROPY: nn.CrossEntropyLoss(),
        }
        criterion = criterion_map.get(loss_fn)
        if not criterion:
            raise ValueError("loss must be focal or cross_entropy")
        return criterion

    def _save_trained_model(self, model: nn.modules, fn_model_pth: str):
        path_model_pth = os.path.join(self._dir_model, fn_model_pth)
        torch.save(model.state_dict(), path_model_pth)
        logger.success(
            ToolBox.runtime_report(
                motive="SAVE",
                action_name=_ACTION_NAME,
                task=self._task_name,
                fn=fn_model_pth,
                path=path_model_pth,
            )
        )

    def _train(
        self,
        model: nn.modules,
        epochs: int,
        data_loader: DataLoader,
        optimizer: Optimizer,
        criterion: typing.Union[FocalLoss, CrossEntropyLoss],
        lrs: StepLR,
    ):
        best_model = copy.deepcopy(model)
        best_acc = 0

        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            for i, (img, label) in enumerate(data_loader):
                img = img.to(self.DEVICE)
                label = label.to(self.DEVICE)
                optimizer.zero_grad()
                out = model(img)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

                if (i + 1) % self.LOG_INTERVAL == 0:
                    logger.debug(
                        ToolBox.runtime_report(
                            motive="ECHO",
                            action_name=_ACTION_NAME,
                            task=self._task_name,
                            device=self.DEVICE,
                            epoch=f"[{epoch + 1}/{epochs}]",
                            iter=i + 1,
                            loss=f"{loss.item():.4f}",
                        )
                    )
                total_loss += loss.item()
                total_acc += torch.sum(torch.argmax(out, dim=1) == label).item()

            lrs.step()
            dataset_length = len(data_loader.dataset)  # noqa
            logger.debug(
                ToolBox.runtime_report(
                    motive="TRAIN",
                    action_name=_ACTION_NAME,
                    task=self._task_name,
                    device=self.DEVICE,
                    epoch=f"[{epoch + 1}/{epochs}]",
                    avg_loss=f"{total_loss / dataset_length:.4f}",
                    avg_acc=f"{total_acc / dataset_length:.4f}",
                )
            )

            if total_acc / dataset_length >= best_acc:
                best_acc = total_acc / dataset_length
                best_model = copy.deepcopy(model)

            if (epoch + 1) % self.SAVE_INTERVAL == 0:
                fn_model_pth = f"{self._task_name}_{epoch + 1}.pth"
                self._save_trained_model(model, fn_model_pth)
                self.val(fn_model_pth)

        model = best_model if self.USE_BEST_MODEL else model

        # Change object
        return model

    def _val(self, model: nn.modules, data_loader: DataLoader):
        total_acc = 0
        for i, (img, label) in enumerate(data_loader):
            img = img.to(self.DEVICE)
            label = label.to(self.DEVICE)
            out = model(img)
            pred = torch.argmax(out, dim=1)
            total_acc += torch.sum(pred == label).item()

        dataset_length = len(data_loader.dataset)  # noqa
        logger.success(
            ToolBox.runtime_report(
                motive="VAL",
                action_name=_ACTION_NAME,
                task=self._task_name,
                size=dataset_length,
                total_acc=f"{total_acc / dataset_length:.4f}",
            )
        )

    def val(self, path_model: typing.Optional[str] = None):
        """

        :param path_model: filename or absPath of model.pth
        :return:
        """
        if path_model is None:
            path_model = os.path.join(self._dir_model, f"{self._task_name}.pth")
        if not os.path.isabs(path_model):
            path_model = os.path.join(self._dir_model, path_model)
        if not os.path.isfile(path_model):
            logger.error(f"ModelNotFound | path={path_model}")
            return False

        model = ResNetMini(3, 2)
        model.load_state_dict(torch.load(path_model))
        model.eval()
        model = model.to(self.DEVICE)

        data = self._get_dataset(self._dir_dataset, "val", with_augment=False)
        data_loader = DataLoader(data, batch_size=self.BATCH_SIZE, shuffle=False)

        self._val(model, data_loader=data_loader)

    def train(self):
        model = ResNetMini(3, 2)
        model.train()
        model.to(device=self.DEVICE)

        logger.info(
            ToolBox.runtime_report(
                motive="TRAIN",
                action_name=_ACTION_NAME,
                total_params=sum(p.numel() for p in model.parameters()),
                model=model,
            )
        )

        optimizer = self._get_optimizer(model, opt=self.OPT_FLAG)
        criterion = self._get_criterion(loss_fn=self.LOSS_FN)
        lrs = torch.optim.lr_scheduler.StepLR(optimizer, self.LR_STEP_SIZE, self.LR_GAMMA)
        data = self._get_dataset(self._dir_dataset, "train", with_augment=True)
        data_loader = DataLoader(data, batch_size=self._batch_size, shuffle=True)

        self._train(
            model=model,
            epochs=self._epochs,
            data_loader=data_loader,
            optimizer=optimizer,
            criterion=criterion,
            lrs=lrs,
        )

        self._save_trained_model(model, fn_model_pth=f"{self._task_name}.pth")
        self.conv_pth2onnx(model=model, verbose=True)

    def conv_pth2onnx(self, model: nn.modules = None, verbose: bool = False, *args, **kwargs):
        path_model_pth = os.path.join(self._dir_model, f"{self._task_name}.pth")
        path_model_onnx = os.path.join(self._dir_model, f"{self._task_name}.onnx")

        if model is None:
            model = ResNetMini(3, 2)
            model.load_state_dict(torch.load(path_model_pth))
        else:
            model = model.cpu()
        model.eval()
        torch.onnx.export(
            model, torch.randn(1, 3, 64, 64), path_model_onnx, verbose=verbose, export_params=True
        )
