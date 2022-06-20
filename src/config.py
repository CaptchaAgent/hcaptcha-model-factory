import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def mkdir(path, remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif remove:
        shutil.rmtree(path)
        os.makedirs(path)


class config:
    remv = False
    task_name = "seaplane"
    data_path = os.path.join("..", "data", task_name)
    labeled_data_path = os.path.join(data_path, "labeled")

    mkdir(data_path)
    mkdir(labeled_data_path, remove=remv)

    n_classes = 2
    classes = [str(i) for i in range(n_classes)]

    for class_ in classes:
        mkdir(os.path.join(labeled_data_path, class_), remove=remv)

    label_file_path = os.path.join(labeled_data_path, "label.txt")

    lr = 0.01
    epochs = 100
    batch_size = 16
    log_interval = 100

    # transform with data augmentation
    data_augmentation_tr = [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.GaussianBlur(3),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        torchvision.transforms.RandomResizedCrop(size=64, scale=(0.8, 1.2)),
    ]

    img_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomChoice(data_augmentation_tr),
            torchvision.transforms.Resize(size=64),
            torchvision.transforms.ToTensor(),
        ]
    )

    model_path = "model.pth"
    model_onnx_path = "model.onnx"
