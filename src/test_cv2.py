import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from config import Config, mkdir


def val_cv2(img_path):
    img_list = os.listdir(img_path)
    # print(img_list)

    model = cv2.dnn.readNetFromONNX(Config.model_onnx_path)
    acc = 0

    for img_name in img_list:
        # print(img_name)
        img = cv2.imread(os.path.join(img_path, img_name))
        img = cv2.resize(img, (64, 64))

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        out = model.forward()
        # print(out.shape)
        # print(out)
        label = np.argmax(out, axis=1)[0]
        # print(label)
        if label == 1:
            acc += 1
        # break

    print(f'{img_path}\n err: {len(img_list) - acc} acc: {acc / len(img_list)}')


if __name__ == '__main__':
    val_img_path = [
        os.path.join(Config.data_path, 'val', 'airplane in the sky flying left', 'yes'),
        os.path.join(Config.data_path, 'val', 'airplane in the sky flying left', 'bad'),
        os.path.join(Config.data_path, 'val', 'airplanes in the sky that are flying to the right',
                     'yes'),
        os.path.join(Config.data_path, 'val', 'airplanes in the sky that are flying to the right',
                     'bad'),
    ]
    for val_img_path_ in val_img_path:
        val_cv2(val_img_path_)

    test_img_path = os.path.join(Config.data_path, 'test')
    for class_ in Config.classes:
        mkdir(os.path.join(test_img_path, class_), remove=True)
