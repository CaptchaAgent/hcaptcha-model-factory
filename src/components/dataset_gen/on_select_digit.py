# generate yolo dataset
import os
import cv2
import random
import numpy as np
import ezkfg as ez
from loguru import logger
from hashlib import md5


def random_color(c_min=64, c_max=255):
    return (
        np.random.randint(c_min, c_max),
        np.random.randint(c_min, c_max),
        np.random.randint(c_min, c_max),
    )


class OnSelectDigitDatasetGen(object):
    def __init__(self, cfg):
        self._bg_path = cfg["bg_path"]
        self._cls_path = cfg["cls_path"]
        self._classes = cfg["classes"]
        self._sig_num_min = cfg["sig_num_min"]
        self._sig_num_max = cfg["sig_num_max"]
        self._sig_size = cfg["sig_size"]
        self._bg_size = cfg["bg_size"] + self._sig_size
        self._final_size = cfg["bg_size"]
        self._tot_num = cfg["tot_num"]
        self._save_path = cfg["save_path"]
        self._feather_size = cfg["feather_size"]

        self._cls2id = {_cls: i for i, _cls in enumerate(self._classes)}

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
                if abs(pos[0] - pos_x) < (self._sig_size // 30) and abs(pos[1] - pos_y) < (
                    self._sig_size // 30
                ):
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

        # bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2RGBA)
        sig_labels = []

        for sig_img, sig_id, pos in zip(sig_imgs, sig_ids, sig_pos):
            # sig_img = cv2.cvtColor(sig_img, cv2.COLOR_RGB2RGBA)

            # paste
            # sig_img.size is [w,h,3] mask is any RGB channel > 0
            np_sig_img = np.array(sig_img)
            mask = np.where(np_sig_img[:, :, 0] > 0, 1, 0)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            bg_img[pos[1] : pos[1] + self._sig_size, pos[0] : pos[0] + self._sig_size] = (
                bg_img[pos[1] : pos[1] + self._sig_size, pos[0] : pos[0] + self._sig_size]
                * (1 - mask)
                + np_sig_img * mask
            ).astype(np.uint8)

            cx = (pos[0] + pos[0] + self._sig_size - self._sig_size // 2) / 2 / self._final_size
            cy = (pos[1] + pos[1] + self._sig_size - self._sig_size // 2) / 2 / self._final_size
            w = self._sig_size / self._final_size
            h = self._sig_size / self._final_size
            sig_labels.append([sig_id, cx, cy, w, h])

        # bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGBA2RGB)
        # center crop
        bg_img = bg_img[
            self._sig_size // 2 : self._sig_size // 2 + self._final_size,
            self._sig_size // 2 : self._sig_size // 2 + self._final_size,
            :,
        ]

        # from np.array to cv2
        bg_img = bg_img.astype(np.uint8)

        return bg_img, sig_labels

    def _generate_bg(self):
        bg_img = np.zeros((self._bg_size, self._bg_size, 3), dtype=np.uint8)
        # random color
        color = random_color(c_min=0, c_max=45)
        bg_img[:, :] = color
        return bg_img

    def _generate_sig(self, cls_id=None):
        if cls_id is None:
            cls_id = np.random.choice(self._classes)
        # write a digit on a blank image
        sig_img = np.zeros((self._sig_size, self._sig_size, 3), dtype=np.uint8)

        # random color
        color = random_color()

        # write digit
        # random font
        font_family = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]
        font = np.random.choice(font_family)
        font_scale = 1
        thickness = np.random.randint(4, 8)
        text = str(cls_id)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        font_scale = min(self._sig_size / text_size[0], self._sig_size / text_size[1])
        # thickness = max(1, int(font_scale))
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        text_x = (self._sig_size - text_size[0]) // 2
        text_y = (self._sig_size + text_size[1]) // 2
        cv2.putText(sig_img, text, (text_x, text_y), font, font_scale, color, thickness)

        # random rotation
        angle = np.random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((self._sig_size / 2, self._sig_size / 2), angle, 1)
        sig_img = cv2.warpAffine(sig_img, M, (self._sig_size, self._sig_size))

        # perspective distortion
        pts1 = np.float32(
            [[0, 0], [self._sig_size, 0], [0, self._sig_size], [self._sig_size, self._sig_size]]
        )
        pts2 = np.float32(
            [
                [np.random.randint(0, 10), np.random.randint(0, 10)],
                [self._sig_size - np.random.randint(0, 10), np.random.randint(0, 10)],
                [np.random.randint(0, 10), self._sig_size - np.random.randint(0, 10)],
                [
                    self._sig_size - np.random.randint(0, 10),
                    self._sig_size - np.random.randint(0, 10),
                ],
            ]
        )
        M = cv2.getPerspectiveTransform(pts1, pts2)
        sig_img = cv2.warpPerspective(sig_img, M, (self._sig_size, self._sig_size))

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
    cfg = ez.load("configs/on_select_digit.yaml")
    dsg = OnSelectDigitDatasetGen(cfg)
    # bg_img = dsg._generate_bg()
    # cv2.imshow("bg_img", bg_img)
    # cv2.waitKey(0)

    # sig_img, sig_id = dsg._generate_sig()
    # cv2.imshow("sig_img", sig_img)
    # cv2.waitKey(0)

    bg_img, sig_labels = dsg._generate()
    print(sig_labels)
    cv2.imshow("bg_img", bg_img)
    cv2.waitKey(0)
