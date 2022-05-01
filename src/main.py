import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import cv2
from PIL import Image

from config import config

from nn.resnet_mini import ResNetMini


def train():
    model = ResNetMini(3, 2)
    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    print('model:', model)

    data = torchvision.datasets.ImageFolder(config.labeled_data_path,
                                            transform=config.img_transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=True)
    print(f'{len(data)} images')
    epochs = config.epochs

    # train with focal loss
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for i, (img, label) in enumerate(data_loader):
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if (i + 1) % config.log_interval == 0:
                print(f'epoch: {epoch + 1}, iter: {i + 1}, loss: {loss.item():.4f}')
            total_loss += loss.item()
            total_acc += torch.sum(torch.argmax(out, dim=1) == label).item()
        print(
            f'epoch: {epoch + 1}, avg loss: {total_loss / len(data):.4f}, avg acc: {total_acc / len(data):.4f}'
        )

    torch.save(model.state_dict(), config.model_path)
    model = model.cpu()
    model.eval()
    torch.onnx.export(model,
                      torch.randn(1, 3, 64, 64),
                      config.model_onnx_path,
                      verbose=True,
                      export_params=True)


def test_single(model, img):

    img = config.img_transform(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    out = model(img)
    pred = torch.argmax(out, dim=1)
    # print(f'pred: {pred.item()}')
    if pred.item() == 0:
        return 0
    else:
        return 1


def test():
    model = ResNetMini(3, 2)


def transfer_model():
    model = ResNetMini(3, 2)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    torch.onnx.export(model,
                      torch.randn(1, 3, 64, 64),
                      config.model_onnx_path,
                      verbose=False,
                      export_params=True)


if __name__ == '__main__':
    train()
    transfer_model()