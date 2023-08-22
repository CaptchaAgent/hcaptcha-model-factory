# -*- coding: utf-8 -*-
# Time       : 2022/7/16 17:42
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from hcaptcha_challenger.agents.skeleton import Status
from loguru import logger
from selenium.common.exceptions import WebDriverException

from hcaptcha_whistleblower.guarder.guarder import GuarderV2


@dataclass
class BinaryClaimer(GuarderV2):
    boolean_tags = ["yes", "bad"]

    """
    # 1. 添加 label_alias
    # ---------------------------------------------------
    # 不在 alias 中的挑战将被跳过

    # 2. 创建彩虹键
    # ---------------------------------------------------
    # 彩虹表中的 rainbow key

    # 3. 创建挑战目录
    # ---------------------------------------------------
    # 遇到新挑战时，先手动创建 rainbow_backup/challengeName/
    # 再在这个目录下分别创建 yes 和 bad 两个文件夹
    """

    def download_images(self):
        # 当遇到的 prompts 不在手动编排的 focus firebird 字典也不在已发布的 firebird 字典中时
        # 可以认为遇到了意外的 binary challenge 或 Others type challenge
        if (
            self._label in self.firebird.focus_labels
            or self._label not in self.modelhub.label_alias
        ):
            super().download_images()

    def claim(self, ctx, retries=5):
        """定向采集数据集"""
        loop_times = -1
        start = time.time()

        while loop_times < retries:
            loop_times += 1
            # 有头模式下自动最小化
            try:
                ctx.get(self.monitor_site)
            except WebDriverException as err:
                if "ERR_PROXY_CONNECTION_FAILED" in err.msg:
                    logger.warning(err.msg)
                    ctx.close()
                    time.sleep(30)
                    continue
                raise err
            ctx.minimize_window()
            # 激活 Checkbox challenge
            self.anti_checkbox(ctx)
            for _ in range(random.randint(5, 8)):
                # 更新挑战框架 | 任务提前结束或进入失败
                if self.switch_to_challenge_frame(ctx) in [
                    Status.CHALLENGE_SUCCESS,
                    Status.CHALLENGE_REFRESH,
                ]:
                    loop_times -= 1
                    break
                # 勾取数据集 | 跳过非聚焦挑战
                self.hacking_dataset(ctx)
                # 随机休眠 | 降低请求频率
                time.sleep(random.uniform(1, 2))
            # 解包数据集 | 每间隔运行3分钟解压一次数据集
            if time.time() - start > 180:
                # self.unpack()
                start = time.time()

    def _unpack(self, dst_dir: Path, from_name: str):
        """
        將 _challenge 中的内容解壓到目標路徑

        :param from_name: 自定義標簽名
        :param dst_dir: rainbow_backup/<label>/
        :return:
        """
        # rainbow_backup/_challenge
        src_dir = self.challenge_dir

        # 标记已有的内容
        _exists_files = {}
        for _, _, files in os.walk(dst_dir):
            for fn in files:
                _exists_files.update({fn: "*"})

        # 清洗出包含標簽名的文件夾緩存
        # 1. 拼接挑戰圖片的絕對路徑
        # 2. 读取二进制流编成hash文件名
        # 3. 写到目标路径
        samples = set()
        for tmp_challenge_dir in os.listdir(src_dir):
            if tmp_challenge_dir.endswith(".png"):
                continue
            if from_name != tmp_challenge_dir.split("_", 1)[-1]:
                continue
            tmp_focus_dir = src_dir.joinpath(tmp_challenge_dir)
            for img_name in os.listdir(tmp_focus_dir):
                img_path = tmp_focus_dir.joinpath(img_name)
                data = img_path.read_bytes()
                filename = f"{hashlib.md5(data).hexdigest()}.png"

                # 过滤掉已存在的文件，无论是 yes|bad|pending
                if not _exists_files.get(filename):
                    dst_dir.joinpath(filename).write_bytes(data)
                    samples.add(filename)

        return len(samples)
