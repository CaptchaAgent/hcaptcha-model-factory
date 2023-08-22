# -*- coding: utf-8 -*-
# Time       : 2023/8/21 0:15
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import shutil
from pathlib import Path

from apis.scaffold import Scaffold

rainbow_dir = Path("../../../Sources/hcaptcha-whistleblower/database2023/rainbow_backup")
factory_data_dir = Path(r"../data")

for task_name in ["diamond_bracelet"]:
    to_dir = factory_data_dir.joinpath(task_name)
    shutil.rmtree(to_dir, ignore_errors=True)
    to_dir.mkdir(mode=777, parents=True, exist_ok=True)

    # Flush dataset
    src_dir = rainbow_dir.joinpath(task_name)
    for hook in ["yes", "bad"]:
        shutil.copytree(src_dir.joinpath(hook), to_dir.joinpath(hook))

    # train
    Scaffold.train(task=task_name)
