# -*- coding: utf-8 -*-
# Time       : 2023/8/21 0:15
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import shutil
from pathlib import Path

from apis.scaffold import Scaffold
from hcaptcha_whistleblower.settings import project

# fmt:off
focus_flags = {
    # "palm_tree",
    # "dolphin",
    # "camera"
}
# fmt:on

binary_dir = project.binary_backup_dir
factory_data_dir = Path(__file__).parent.parent.joinpath("data")

# Copy the classified data into the dataset
for task_name in focus_flags:
    to_dir = factory_data_dir.joinpath(task_name)
    shutil.rmtree(to_dir, ignore_errors=True)
    to_dir.mkdir(mode=777, parents=True, exist_ok=True)

    # Flush dataset
    src_dir = binary_dir.joinpath(task_name)
    for hook in ["yes", "bad"]:
        shutil.copytree(src_dir.joinpath(hook), to_dir.joinpath(hook))

    # train
    Scaffold.train(task=task_name)
