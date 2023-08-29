# generate yolo dataset
import os
import cv2
import random
import numpy as np
import ezkfg as ez
from loguru import logger
from hashlib import md5


class OnSelectDatasetGen(object):
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
    
    def _generate(self):
        pass



    def _generate_bg(self):
        pass


    def _generate_sig(self, cls_id=None):
        pass

    def generate(self):
        os.makedirs(os.path.join(self._save_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self._save_path, "labels"), exist_ok=True)
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