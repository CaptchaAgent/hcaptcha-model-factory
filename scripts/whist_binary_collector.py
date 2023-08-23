# -*- coding: utf-8 -*-
# Time       : 2023/8/22 0:26
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from hcaptcha_whistleblower import run_binary_collector
from hcaptcha_whistleblower.settings import SiteKey

# 初始化聚焦标签，采集器仅会下载关注的以及未编排在 objects.yaml 中的数据
# [prompts] --> [train label]
focus_labels = {
    "camera": "camera",
    "diamond bracelet": "diamond_bracelet",
    "dolphin": "dolphin",
    "red panda": "red_panda",
    "palm tree": "palm_tree",
}

sitekey = SiteKey.cloud_horse

if __name__ == "__main__":
    run_binary_collector(sitekey=sitekey, focus_labels=focus_labels, silence=False)
