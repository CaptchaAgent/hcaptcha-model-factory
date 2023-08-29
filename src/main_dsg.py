import ezkfg as ez
import os
from loguru import logger
from components.dataset_gen.on_select_animal import OnSelectAnimalDatasetGen
from components.dataset_gen.on_select_digit import OnSelectDigitDatasetGen

if __name__ == "__main__":
    # cfg_path = os.path.join("configs", "on_select_animal.yaml")
    cfg_path = os.path.join("configs", "on_select_digit.yaml")
    cfg = ez.load(cfg_path)

    # clss = os.listdir(cfg["cls_path"])
    # cfg.update({"classes": clss})
    # cfg.update({"tot_num": 1000})
    # cfg.update({"sig_num_min": 4})
    # cfg.update({"sig_num_max": 5})

    # cfg.dump(cfg_path)
    logger.info(cfg)

    dsg = OnSelectDigitDatasetGen(cfg)
    dsg.generate()
