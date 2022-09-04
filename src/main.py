import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil
import torch
import torch.nn as nn
import torchvision
import copy

import cv2
from PIL import Image

from config import Config

from nn.resnet_mini import ResNetMini
from losses import FocalLoss


def split_data():
    yes_path = Config.origin_data_path / "yes"
    bad_path = Config.origin_data_path / "bad"

    train_path = Config.train_data_path
    val_path = Config.val_data_path
    test_path = Config.test_data_path

    for img in yes_path.iterdir():
        p_rand = np.random.rand()
        if p_rand < Config.train_ratio:
            shutil.copy(img, train_path / "0")
        elif p_rand < Config.train_ratio + Config.val_ratio:
            shutil.copy(img, val_path / "0")
        else:
            shutil.copy(img, test_path / "0")

    for img in bad_path.iterdir():
        p_rand = np.random.rand()
        if p_rand < Config.train_ratio:
            shutil.copy(img, train_path / "1")
        elif p_rand < Config.train_ratio + Config.val_ratio:
            shutil.copy(img, val_path / "1")
        else:
            shutil.copy(img, test_path / "1")


def train():
    model = ResNetMini(3, 2)
    total_params = sum(p.numel() for p in model.parameters())

    model.train()
    model.to(Config.device)

    print("model:", model)
    print("total params:", total_params)

    if Config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=Config.lr,
            momentum=Config.lr_momentum,
            weight_decay=Config.lr_weight_decay,
        )
    elif Config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=Config.lr, weight_decay=Config.lr_weight_decay
        )
    else:
        raise ValueError("optimizer must be sgd or adam")

    if Config.loss_fn == "focal":
        criterion = FocalLoss(gamma=Config.focal_loss_gamma)
    elif Config.loss_fn == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("loss must be focal or cross_entropy")

    lrs = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.lr_step_size, gamma=Config.lr_gamma
    )

    data = torchvision.datasets.ImageFolder(Config.train_data_path, transform=Config.img_transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=Config.batch_size, shuffle=True)
    print(f"{len(data)} images")
    epochs = Config.epochs

    best_model = copy.deepcopy(model)
    best_acc = 0

    # train with focal loss
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for i, (img, label) in enumerate(data_loader):
            img = img.to(Config.device)
            label = label.to(Config.device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if (i + 1) % Config.log_interval == 0:
                print(f"epoch: {epoch + 1}, iter: {i + 1}, loss: {loss.item():.4f}")
            total_loss += loss.item()
            total_acc += torch.sum(torch.argmax(out, dim=1) == label).item()

        lrs.step()
        print(
            f"epoch: {epoch + 1}, avg loss: {total_loss / len(data):.4f}, avg acc: {total_acc / len(data):.4f}"
        )

        if total_acc / len(data) >= best_acc:
            best_acc = total_acc / len(data)
            best_model = copy.deepcopy(model)

        if (epoch + 1) % Config.save_interval == 0:
            _path = Config.model_path.parent / (
                Config.model_path.stem.split(".")[0] + f"_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), _path)
            print(f"save model to {_path}")
            val(_path)

    if Config.use_best_model:
        model = best_model

    torch.save(model.state_dict(), Config.model_path)
    model = model.cpu()
    model.eval()
    torch.onnx.export(
        model, torch.randn(1, 3, 64, 64), Config.model_onnx_path, verbose=True, export_params=True
    )


def val(model_path=None):
    model_path = model_path or Config.model_path
    model = ResNetMini(3, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(Config.device)
    data = torchvision.datasets.ImageFolder(
        Config.val_data_path, transform=Config.img_transform_no_augment
    )
    data_loader = torch.utils.data.DataLoader(data, batch_size=Config.batch_size, shuffle=False)
    print(f"{len(data)} images")
    total_acc = 0
    for i, (img, label) in enumerate(data_loader):
        img = img.to(Config.device)
        label = label.to(Config.device)
        out = model(img)
        pred = torch.argmax(out, dim=1)
        total_acc += torch.sum(pred == label).item()
    print(f"total acc: {total_acc / len(data):.4f}")


def test_single_img(model, img):
    img = Config.img_transform(img)
    img = img.unsqueeze(0)
    img = img.to(Config.device)
    out = model(img)
    pred = torch.argmax(out, dim=1)
    # print(f'pred: {pred.item()}')
    if pred.item() == 0:
        return 0
    else:
        return 1


def test_single():
    model = ResNetMini(3, 2)
    model.load_state_dict(torch.load(Config.model_path))
    model.eval()
    model = model.to(Config.device)
    img = cv2.imread(Config.test_img_path)
    img = cv2.resize(img, (64, 64))
    img = Image.fromarray(img)
    print(test_single_img(model, img))


def transfer_model():
    model = ResNetMini(3, 2)
    model.load_state_dict(torch.load(Config.model_path))
    model.eval()
    torch.onnx.export(
        model, torch.randn(1, 3, 64, 64), Config.model_onnx_path, verbose=False, export_params=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()

    if args.split:
        split_data()

    if "train" in args.mode:
        train()
        transfer_model()

    if "val" in args.mode:
        val()
