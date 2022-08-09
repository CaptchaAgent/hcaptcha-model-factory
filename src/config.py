import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random


from pathlib import Path


def mkdir(path, remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif remove:
        shutil.rmtree(path)
        os.makedirs(path)


class config:
    remv = True
    task_name = "lion_with_closed_eyes"
    data_path = Path(os.path.join("..", "data", task_name))

    origin_data_path = data_path / "origin"
    # labeled_data_path = os.path.join(data_path, "labeled")
    train_data_path = data_path / "train"
    val_data_path = data_path / "val"
    test_data_path = data_path / "test"

    seed = 233

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # split train/val/test data
    train_ratio = 0.8
    val_ratio = 0.2
    test_ratio = 0.0

    mkdir(data_path)
    mkdir(train_data_path, remove=remv)
    mkdir(val_data_path, remove=remv)
    mkdir(test_data_path, remove=remv)

    # best model
    use_best_model = False

    # data classes
    n_classes = 2
    classes = [str(i) for i in range(n_classes)]

    for class_ in classes:
        mkdir(os.path.join(train_data_path, class_), remove=remv)
        mkdir(os.path.join(val_data_path, class_), remove=remv)
        mkdir(os.path.join(test_data_path, class_), remove=remv)

    # data label file
    label_file_path = os.path.join(train_data_path, "label.txt")

    # training settings
    lr = 0.001
    lr_step_size = 30
    lr_gamma = 0.6
    lr_weight_decay = 0.0005
    lr_momentum = 0.9
    epochs = 200
    batch_size = 4
    loss_fn = "focal"  # "focal" or "cross_entropy"
    optimizer = "adam"  # "sgd" | "adam"
    focal_loss_gamma = 2.0

    # log and checkpoint settings
    save_interval = 50
    log_interval = 100

    # data augmentation
    data_augmentation_tr = [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.GaussianBlur(3),
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        ),
        torchvision.transforms.RandomResizedCrop(size=64, scale=(0.8, 1.2)),
    ]

    # transform with data augmentation
    img_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomChoice(data_augmentation_tr),
            torchvision.transforms.Resize(size=64),
            torchvision.transforms.ToTensor(),
        ]
    )

    # transform without data augmentation
    img_transform_no_augment = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=64), torchvision.transforms.ToTensor()]
    )

    # model path
    model_dir = Path(os.path.join("..", "model"))
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / f"{task_name}.pth"
    model_onnx_path = model_dir / f"{task_name}.onnx"
