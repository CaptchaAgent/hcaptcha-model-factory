# -*- coding: utf-8 -*-
# Time       : 2022/7/15 21:00
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Optional

from hcaptcha_challenger.agents.selenium import get_challenge_ctx
from loguru import logger

from hcaptcha_whistleblower.guarder import BinaryClaimer
from hcaptcha_whistleblower.settings import SiteKey
from hcaptcha_whistleblower.settings import project

GETTER_RETRIES = 500


@dataclass
class BinaryCollector(BinaryClaimer):
    """hCAPTCHA Focus Challenge Collector"""

    def claim(self, ctx, retries=5):
        if not self._label_alias:
            logger.warning("聚焦挑战为空，白名单模式失效，将启动备用的黑名单模式运行采集器")
        logger.debug("focus", sitekey=self.sitekey)
        logger.debug("focus", monitor=self.monitor_site)
        logger.debug("focus", label_alias=self._label_alias.values())
        with suppress(KeyboardInterrupt):
            super().claim(ctx, retries)

    def unpack(self):
        statistics_: Optional[dict] = super().unpack()
        for flag in statistics_:
            count = statistics_[flag]
            if count:
                logger.success(f"UNPACK [{flag}] --> count={count}")


def run_binary_collector(sitekey: str = SiteKey.epic, silence=False):
    """根据label定向采集数据集"""
    logger.info("startup collector")

    # 采集数据集 | 自动解包数据集
    cc = BinaryCollector.from_modelhub(tmp_dir=project.binary_backup_dir)
    cc.sitekey = sitekey
    cc.modelhub.pull_objects()

    # 退出任务前执行最后一次解包任务
    # 确保所有任务进度得以同步
    ctx = get_challenge_ctx(lang="en", silence=silence)
    try:
        cc.claim(ctx, retries=GETTER_RETRIES)
        cc.unpack()
    finally:
        logger.success("采集器退出")
        ctx.quit()


def unpack_cache():
    cc = BinaryCollector.from_modelhub()
    cc.unpack()
