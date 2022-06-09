import os
import shutil
import sys

import torchvision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mkdir(path, remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif remove:
        shutil.rmtree(path)
        os.makedirs(path)


class Config:
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

    lr = 0.005
    epochs = 20
    batch_size = 16
    log_interval = 100

    img_transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Grayscale(num_output_channels=1),
            # torchvision.transforms.GaussianBlur(kernel_size=3),
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
        ]
    )

    model_path = "model.pth"
    model_onnx_path = "model.onnx"
