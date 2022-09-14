import copy
import os
import random
import typing
import warnings
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
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
        dataset_all = self._make_datamodel(flag=self.FILENAME_YAML_ALL)
        dataset_train = self._make_datamodel(flag=self.FILENAME_YAML_TRAIN)
        dataset_val = self._make_datamodel(flag=self.FILENAME_YAML_VAL)
        dataset_test = self._make_datamodel(flag=self.FILENAME_YAML_TEST)

        # Check && Split Dataset(though yaml)
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
                if not os.path.isfile(src_path_img) or not ToolBox.is_image(src_path_img):
                    warnings.warn(f"It's not a image file, {src_path_img}", category=RuntimeWarning)
                    continue
                label = 0 if hook == dir_dataset_yes else 1
                image_info = {"fname": src_path_img, "label": label}
                dataset_all.data.append(image_info)
                if (operator := random.uniform(0, 1)) < self.RATIO_TRAIN:
                    dataset_train.data.append(image_info)
                elif operator < self.RATIO_TRAIN + self.RATIO_VAL:
                    dataset_val.data.append(image_info)
                else:
                    dataset_test.data.append(image_info)

        # Save dataset info
        self.save_datamodels()

    @staticmethod
    def _get_dataset(
        dir_dataset: str, flag: str, with_augment: bool, need_transform: bool = True
    ) -> Dataset:
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

        if need_transform:
            transform_ = img_transform if with_augment else img_transform_no_augment
        else:
            transform_ = None
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
            total_tp = 0
            total_tn = 0
            total_fp = 0
            total_fn = 0

            for i, (img, label, _) in enumerate(data_loader):
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
                total_tp += torch.sum((torch.argmax(out, dim=1) == 1) & (label == 1)).item()
                total_tn += torch.sum((torch.argmax(out, dim=1) == 0) & (label == 0)).item()
                total_fp += torch.sum((torch.argmax(out, dim=1) == 1) & (label == 0)).item()
                total_fn += torch.sum((torch.argmax(out, dim=1) == 0) & (label == 1)).item()

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
                    avg_f1=f"{2 * total_tp / (2 * total_tp + total_fp + total_fn):.4f}",
                    avg_precision=f"{total_tp / (total_tp + total_fp):.4f}",
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
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0

        for _, (img, label, _) in enumerate(data_loader):
            img = img.to(self.DEVICE)
            label = label.to(self.DEVICE)
            out = model(img)
            pred = torch.argmax(out, dim=1)
            total_acc += torch.sum(pred == label).item()
            total_tp += torch.sum((pred == 1) & (label == 1)).item()
            total_tn += torch.sum((pred == 0) & (label == 0)).item()
            total_fp += torch.sum((pred == 1) & (label == 0)).item()
            total_fn += torch.sum((pred == 0) & (label == 1)).item()

        dataset_length = len(data_loader.dataset)  # noqa
        logger.success(
            ToolBox.runtime_report(
                motive="VAL",
                action_name=_ACTION_NAME,
                task=self._task_name,
                size=dataset_length,
                total_acc=f"{total_acc / dataset_length:.4f}",
                total_f1=f"{2 * total_tp / (2 * total_tp + total_fp + total_fn):.4f}",
                total_precision=f"{total_tp / (total_tp + total_fp):.4f}",
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

    def conv_pth2onnx(self, model: nn.modules = None, verbose: bool = False, **kwargs):
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

    @staticmethod
    def _onnx_infer(net, img_arr):
        img_arr = np.frombuffer(img_arr, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        img = cv2.resize(img, (64, 64))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        out = net.forward()
        if not np.argmax(out, axis=1)[0]:
            return True
        return False

    @staticmethod
    def _get_latest_onnx_model(dir_model: str, model_prefix: str):
        models = glob(os.path.join(dir_model, f"**/{model_prefix}*.onnx"), recursive=True)
        if not models:
            return None
        return max(models, key=os.path.getctime)

    def test_onnx(self, flag: str = "all"):
        path_model_onnx = self._get_latest_onnx_model(self._dir_model, self._task_name)

        logger.debug(f"path_model_onnx={path_model_onnx}")

        if path_model_onnx is None or not os.path.isfile(path_model_onnx):
            logger.error(f"ModelNotFound | path={path_model_onnx}")
            return False

        model = cv2.dnn.readNetFromONNX(path_model_onnx)
        logger.debug(f"ModelLoaded | path={path_model_onnx}")

        dataset = self._get_dataset(
            self._dir_dataset, flag=flag, with_augment=False, need_transform=False
        )

        total_acc = 0
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0

        for idx, (_, label, fname) in enumerate(dataset):
            with open(fname, "rb") as file:
                img_arr = file.read()
            pred = self._onnx_infer(model, img_arr=img_arr)
            pred = 0 if pred else 1

            # logger.debug(f"pred={pred} | label={label} | fname={fname}")

            total_acc += 1 if pred == label else 0
            total_tp += 1 if (pred == 1) & (label == 1) else 0
            total_tn += 1 if (pred == 0) & (label == 0) else 0
            total_fp += 1 if (pred == 1) & (label == 0) else 0
            total_fn += 1 if (pred == 0) & (label == 1) else 0

            if (idx + 1) % self.LOG_INTERVAL == 0:
                logger.info(f"TestOnnx | {idx + 1}/{len(dataset)} | acc={total_acc / (idx + 1)}")

        dataset_length = len(dataset)
        logger.debug(
            ToolBox.runtime_report(
                motive="TEST_ONNX",
                action_name=_ACTION_NAME,
                task=self._task_name,
                avg_acc=f"{total_acc / dataset_length:.4f}",
                avg_f1=f"{2 * total_tp / (2 * total_tp + total_fp + total_fn):.4f}",
                avg_precision=f"{total_tp / (total_tp + total_fp):.4f}",
            )
        )
