import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import random
from os.path import dirname, join

from utils import ToolBox


class Config:
    # set random seed
    SEED = 233

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

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
