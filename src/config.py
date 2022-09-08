import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torchvision
import random
from os.path import dirname, join

from pathlib import Path
from utils import ToolBox


def mkdir(path, remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif remove:
        shutil.rmtree(path)
        os.makedirs(path)


class Config:
    # set random seed
    seed = 233
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # hook to factory/
    PROJECT_ROOT = dirname(dirname(__file__))
    # hook to factory/data/
    DIR_DATABASE = join(PROJECT_ROOT, "data")
    # hook to factory/model/
    DIR_MODEL = join(PROJECT_ROOT, "model")
    # hook to factory/logs/
    DIR_LOG = join(PROJECT_ROOT, "logs")


logger = ToolBox.init_log(
    error=join(Config.DIR_LOG, "error.log"), runtime=join(Config.DIR_LOG, "runtime.log")
)
