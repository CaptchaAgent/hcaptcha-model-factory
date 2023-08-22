# -*- coding: utf-8 -*-
# Time       : 2022/7/15 21:00
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from hcaptcha_challenger.agents.selenium import get_challenge_ctx
from loguru import logger

from hcaptcha_whistleblower.guarder import BinaryClaimer
from hcaptcha_whistleblower.settings import SiteKey
from hcaptcha_whistleblower.settings import project


@dataclass
class BinaryCollector(BinaryClaimer):
    """hCAPTCHA Focus Challenge Collector"""

    def claim(self, ctx, retries=5):
        if not self.firebird.focus_labels:
            logger.warning("聚焦挑战为空，白名单模式失效，将启动备用的黑名单模式运行采集器")
        logger.debug("focus", sitekey=self.sitekey)
        logger.debug("focus", monitor=self.monitor_site)
        logger.debug("focus", label_alias=self.firebird.focus_labels.values())
        with suppress(KeyboardInterrupt):
            super().claim(ctx, retries)

    def unpack(self):
        """
        解构彩虹表，自动分类，去重，拷贝

        FROM: rainbow_backup/_challenge
        TO: rainbow_backup/[*challengeName]

        :return:
        """
        channel_dirs = set(self.firebird.focus_labels.values())
        for tag in self.boolean_tags:
            for channel_dir in channel_dirs:
                tmp = project.binary_backup_dir.joinpath(f"{channel_dir}/{tag}")
                tmp.mkdir(777, parents=True, exist_ok=True)

        flag2counts = {}
        fullback_labels = self.firebird.focus_labels.copy()
        fullback_labels.update({v: v for v in fullback_labels.values()})
        for prompt_name, flag in fullback_labels.items():
            dst_dir = project.binary_backup_dir.joinpath(flag)
            samples_len = self._unpack(dst_dir=dst_dir, from_name=prompt_name)
            if flag in flag2counts:
                flag2counts[flag] += samples_len
            else:
                flag2counts[flag] = samples_len

        for flag, num in flag2counts.items():
            if not num:
                continue
            logger.success(f"UNPACK", flag=flag, count=num)


def run_binary_collector(sitekey: str = SiteKey.epic, silence=False, r: int = 50):
    """根据label定向采集数据集"""
    logger.info("startup collector", type="binary")

    # 采集数据集 | 自动解包数据集
    cc = BinaryCollector.from_modelhub(tmp_dir=project.binary_backup_dir)
    cc.sitekey = sitekey
    cc.modelhub.pull_objects()

    # 退出任务前执行最后一次解包任务
    # 确保所有任务进度得以同步
    ctx = get_challenge_ctx(lang="en", silence=silence)
    try:
        cc.claim(ctx, retries=r)
        cc.unpack()
    finally:
        logger.success("采集器退出")
        ctx.quit()


def unpack_cache():
    cc = BinaryCollector.from_modelhub(tmp_dir=project.binary_backup_dir)
    cc.modelhub.pull_objects()
    cc.unpack()
