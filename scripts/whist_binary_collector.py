# -*- coding: utf-8 -*-
# Time       : 2023/8/22 0:26
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from hcaptcha_whistleblower import run_binary_collector
from hcaptcha_whistleblower.settings import SiteKey

if __name__ == "__main__":
    run_binary_collector(sitekey=SiteKey.epic, silence=False)
