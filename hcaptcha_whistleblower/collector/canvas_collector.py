# -*- coding: utf-8 -*-
# Time       : 2023/8/16 2:27
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import hashlib
import json
import os
from dataclasses import dataclass

import requests
from hcaptcha_challenger.agents.selenium import get_challenge_ctx
from loguru import logger
from selenium.common.exceptions import InvalidArgumentException

from hcaptcha_whistleblower.guarder import BinaryClaimer
from hcaptcha_whistleblower.settings import project


@dataclass
class CanvasCollector(BinaryClaimer):
    img_url = ""
    img_path = ""
    canvas_backup_dir = project.canvas_backup_dir

    def get_challenge_prompt(self):
        """通过OCR或其他技术获知需要框选的目标ID"""

    def get_img_url(self, ctx) -> str:
        """针对图像区域选择挑战，拦截网络流，返回图片下载链接"""
        log_type = "performance"

        try:
            logs_ = ctx.get_log(log_type)
        except InvalidArgumentException:
            pass
        else:
            for log_ in logs_:
                message: dict = json.loads(log_.get("message", ""))
                message: dict = message.get("message", {})
                _params: dict = message.get("params", {})
                _response: dict = _params.get("response", {})
                _url: str = _response.get("url", "")
                _type: str = _params.get("type", "")
                _content_length: str = _response.get("headers", {}).get("content-length", "1")
                if (
                    _type.lower() == "image"
                    and _url
                    and not _url.endswith(".svg")
                    and int(_content_length) > 12060
                ):
                    self.img_url = _url
                    logger.debug("最大容错临界", content_length=_content_length)
                    break
        finally:
            return self.img_url

    def download_images(self):
        if not self.img_url.startswith("https://"):
            return

        resp = requests.get(self.img_url)
        logger.debug(f"{resp.headers}")

        content = resp.content
        filename = f"{hashlib.md5(content).hexdigest()}.jpg"
        self.img_path = os.path.join(self.canvas_backup_dir, filename)

        with open(self.img_path, "wb") as file:
            file.write(content)

        head = requests.head(self.img_url)
        logger.debug(f"{head.headers}")

    def _hacking_dataset(self, ctx):
        self.get_img_url(ctx)
        self.download_images()
        self.refresh_hcaptcha(ctx)


def run_canvas_collector(sitekey: str = "ace50dd0-0d68-44ff-931a-63b670c7eed7", r: int = 5):
    logger.info("startup canvas collector", type="canvas")

    # 采集数据集 | 自动解包数据集
    ct = CanvasCollector.from_modelhub(tmp_dir=project.binary_backup_dir)
    ct.sitekey = sitekey
    ct.modelhub.pull_objects()

    # 确保所有任务进度得以同步
    ctx = get_challenge_ctx(silence=False, lang="en")
    try:
        ct.claim(ctx, retries=r)
    finally:
        logger.success("采集器退出")
        ctx.quit()
