# generate yolo dataset
import os
import cv2
import random
import numpy as np
import ezkfg as ez
from loguru import logger
from hashlib import md5


def adjust_brightness_contrast(img, brightness=0, contrast=0):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b = brightness
    c = contrast
    if b > 0:
        shadow = b
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + b
    alpha_b = 0.8
    beta_b = shadow

    alpha_c = 1.0 + c
    beta_c = 0

    img = cv2.convertScaleAbs(img, alpha=alpha_c, beta=beta_c)
    img = cv2.convertScaleAbs(img, alpha=alpha_b, beta=beta_b)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def style_transfer(img, img_sz=64, brightness=-5, contrast=0.1, show=False):
    h, w = img.shape[:2]
    size = min(h, w)

    # random crop
    start_h = np.random.normal(0, 1) * (h - size) // 4
    start_w = np.random.normal(0, 1) * (w - size) // 4

    start_h = max(0, int(start_h))
    start_w = max(0, int(start_w))

    img = img[start_h : start_h + size, start_w : start_w + size, :]
    img = cv2.resize(img, (512, 512))

    res = cv2.xphoto.oilPainting(img, 2, 1)
    res = cv2.detailEnhance(res, sigma_s=50, sigma_r=0.8)
    res = adjust_brightness_contrast(res, brightness=brightness, contrast=contrast)

    # denoise
    res = cv2.fastNlMeansDenoisingColored(res, None, 10, 10, 7, 21)

    res = cv2.resize(res, (img_sz, img_sz), interpolation=cv2.INTER_AREA)

    if show:
        cv2.imshow("img", res)
        cv2.waitKey(0)

    return res


class OnSelectAnimalDatasetGen(object):
    def __init__(self, cfg):
        self._bg_path = cfg["bg_path"]
        self._cls_path = cfg["cls_path"]
        self._classes = cfg["classes"]
        self._sig_num_min = cfg["sig_num_min"]
        self._sig_num_max = cfg["sig_num_max"]
        self._sig_size = cfg["sig_size"]
        self._bg_size = cfg["bg_size"]
        self._tot_num = cfg["tot_num"]
        self._save_path = cfg["save_path"]
        self._feather_size = cfg["feather_size"]

        self._bg_imgs = os.listdir(self._bg_path)
        self._cls_imgs = {}
        for _cls in self._classes:
            self._cls_imgs[_cls] = os.listdir(os.path.join(self._cls_path, _cls))
        self._cls2id = {cls: i for i, cls in enumerate(self._classes)}

    def _generate(self):
        bg_img = self._generate_bg()
        sig_imgs = []
        sig_ids = []

        for i in range(random.randint(self._sig_num_min, self._sig_num_max)):
            sig_img, sig_id = self._generate_sig()
            sig_imgs.append(sig_img)
            sig_ids.append(sig_id)

        # paste sig to bg, random position, save label
        sig_pos = []

        # check overlap
        def _check_overlap(pos_x, pos_y):
            for pos in sig_pos:
                if abs(pos[0] - pos_x) < (self._sig_size - 4 * self._feather_size) * np.sqrt(
                    2
                ) and abs(pos[1] - pos_y) < (self._sig_size - 4 * self._feather_size) * np.sqrt(2):
                    return False
            return True

        retry = 0
        for sig_img, sig_id in zip(sig_imgs, sig_ids):
            retry = 0
            pos_x, pos_y = np.random.randint(0, self._bg_size - self._sig_size), np.random.randint(
                0, self._bg_size - self._sig_size
            )
            while not _check_overlap(pos_x, pos_y):
                pos_x, pos_y = np.random.randint(
                    0, self._bg_size - self._sig_size
                ), np.random.randint(0, self._bg_size - self._sig_size)
                retry += 1
                if retry > 10:
                    logger.warning("Cannot find a proper position, retrying...")
                    return self._generate()

            sig_pos.append((pos_x, pos_y))

        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2RGBA)
        sig_labels = []

        # circle mask with edge feathering
        mask = np.zeros((self._sig_size, self._sig_size, 4), dtype=np.uint8)
        mask = cv2.circle(
            mask,
            (self._sig_size // 2, self._sig_size // 2),
            self._sig_size // 2 - self._feather_size,
            (255, 255, 255, 255),
            -1,
        )
        mask = cv2.GaussianBlur(mask, (self._feather_size, self._feather_size), 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)

        for sig_img, sig_id, pos in zip(sig_imgs, sig_ids, sig_pos):
            sig_img = cv2.cvtColor(sig_img, cv2.COLOR_RGB2RGBA)

            # paste
            bg_img[
                pos[0] : pos[0] + self._sig_size, pos[1] : pos[1] + self._sig_size, :
            ] = sig_img * (mask / 255) + bg_img[
                pos[0] : pos[0] + self._sig_size, pos[1] : pos[1] + self._sig_size, :
            ] * (
                1 - mask / 255
            )
            cx = (pos[0] + pos[0] + self._sig_size) / 2 / self._bg_size
            cy = (pos[1] + pos[1] + self._sig_size) / 2 / self._bg_size
            w = self._sig_size / self._bg_size
            h = self._sig_size / self._bg_size
            sig_labels.append([self._cls2id[sig_id], cx, cy, w, h])

        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGBA2RGB)
        return bg_img, sig_labels

    def _generate_bg(self):
        bg_img = cv2.imread(os.path.join(self._bg_path, np.random.choice(self._bg_imgs)))
        bg_img = style_transfer(bg_img, img_sz=self._bg_size, brightness=10, contrast=0.2)
        # increase green color brightness
        bg_img[:, :, 1] = bg_img[:, :, 1] * 1.1
        bg_img[:, :, 1] = np.clip(bg_img[:, :, 1], 0, 255)

        return bg_img

    def _generate_sig(self, cls_id=None):
        if cls_id is None:
            cls_id = np.random.choice(self._classes)
        sig_img = cv2.imread(
            os.path.join(self._cls_path, cls_id, np.random.choice(self._cls_imgs[cls_id]))
        )
        sig_img = style_transfer(sig_img, img_sz=self._sig_size, brightness=-3, contrast=0.5)
        return sig_img, cls_id

    def generate(self):
        os.makedirs(os.path.join(self._save_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self._save_path, "labels"), exist_ok=True)

        data_cfg = {"classes": self._classes, "nc": len(self._classes)}
        ez.save(data_cfg, os.path.join(self._save_path, "data.yaml"))

        for i in range(self._tot_num):
            img, label = self._generate()
            img_name = md5(img).hexdigest()
            cv2.imwrite(os.path.join(self._save_path, "images", f"{img_name}.jpg"), img)
            with open(os.path.join(self._save_path, "labels", f"{img_name}.txt"), "w") as file:
                for l in label:
                    file.write(f"{l[0]} {l[1]} {l[2]} {l[3]} {l[4]}\n")

            logger.info(f"Generated {i+1}/{self._tot_num} images")


if __name__ == "__main__":
    root = "G:\\dataset\\animal2\\animals\\animals\\bear"
    img = cv2.imread(os.path.join(root, "3dbd23430d.jpg"))

    img = style_transfer(img, show=True)
