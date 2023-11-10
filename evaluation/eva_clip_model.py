# -*- coding: utf-8 -*-
# Time       : 2023/11/9 19:52
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description: Check Multi choices
from pathlib import Path

from hcaptcha_challenger import register_pipline, ModelHub
from PIL import Image

candidates = ['kitchen', 'bedroom', 'living_room']
image_path = Path("zip_dir/multi.jpg")

modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()

model = register_pipline(modelhub, fmt="onnx")
results = model([Image.open(image_path)], candidates)
for result in results:
    print(result)
