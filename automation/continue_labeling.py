# -*- coding: utf-8 -*-
# Time       : 2023/9/24 15:07
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import hcaptcha_challenger as solver

prompt = "dog"

__formats = ("%Y-%m-%d %H:%M:%S.%f", "%Y%m%d%H%M")
now = datetime.strptime(str(datetime.now()), __formats[0]).strftime(__formats[1])

task_name = solver.prompt2task(prompt)

project_dir = Path(__file__).parent.parent
from_dir = project_dir.joinpath("database2309", task_name)
yes_dir = from_dir.joinpath(now, "yes")
bad_dir = from_dir.joinpath(now, "bad")

images = [
    from_dir.joinpath(img_name) for img_name in os.listdir(from_dir) if img_name.endswith(".png")
]

if not images:
    sys.exit()

solver.install(upgrade=True)

yes_dir.mkdir(parents=True, exist_ok=True)
bad_dir.mkdir(parents=True, exist_ok=True)

classifier = solver.BinaryClassifier()
results = classifier.execute(prompt, images)
for i, result in enumerate(results):
    if result is True:
        shutil.move(images[i], yes_dir)
    elif result is False:
        shutil.move(images[i], bad_dir)

if "win32" in sys.platform:
    os.startfile(from_dir)
