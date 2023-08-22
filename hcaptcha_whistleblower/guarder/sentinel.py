# -*- coding: utf-8 -*-
# Time       : 2022/7/17 4:17
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional, Union, Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from hcaptcha_challenger.agents.skeleton import Status
from loguru import logger

from hcaptcha_whistleblower.guarder import GuarderV2
from hcaptcha_whistleblower.settings import firebird, SiteKey


@dataclass
class Sentinel(GuarderV2):
    """hCAPTCHA New Challenge Sentinel"""

    def lock_challenge(self):
        """缓存加锁，防止新挑战的重复警报"""
        self._label_alias[self._label] = self._label

        # TODO: {{<- MongoDB Sync ->}}

        return self._label

    def unlock_challenge(self, keys: Union[Tuple[str, str], str]):
        """
        弹出缓存锁，允许新挑战的重复警报
        :param keys: 需要弹出的 label，通常需要传入 clean_label 以及 raw_label
        :return:
        """
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            self._label_alias.pop(key)

        # TODO: {{<- MongoDB Offload ->}}

    def tango(self, ctx, timer: Optional[int] = None):
        loop_times = -1
        retry_times = 3 if not timer else 100
        trigger = time.time()
        while loop_times < retry_times:
            loop_times += 1
            # 有头模式下自动最小化
            ctx.get(self.monitor_site)
            ctx.minimize_window()
            # 激活 Checkbox challenge
            self.anti_checkbox(ctx)
            for _ in range(random.randint(5, 8)):
                # 更新挑战框架 | 任务提前结束或进入失败
                if self.switch_to_challenge_frame(ctx) in [
                    Status.CHALLENGE_SUCCESS,
                    Status.CHALLENGE_REFRESH,
                ]:
                    break
                # 正常响应 | 已标记的挑战标签
                if not self.checking_dataset(ctx):
                    time.sleep(random.uniform(3, 5))
                    continue
                # 拉响警报 | 出现新的挑战
                # self.broadcast_alert_information()
                firebird.flush()
                if timer and time.time() - trigger > timer:
                    logger.info(f"Drop by outdated - upto={timer}")
                    return


@logger.catch
def tango(sitekey: str | None = None, timer: int = 300):
    sitekey = sitekey or SiteKey.shuffle()

    with Sentinel.from_modelhub() as sentinel:
        sentinel.sitekey = sitekey
        logger.info("build sentinel", at=sentinel.monitor_site)
        sentinel.tango(sentinel.ctx_session, timer=timer)


def deploy_sentinel():
    scheduler = BackgroundScheduler()
    job_name = "sentinel"
    logger.info("deploy scheduler", job_name=job_name, interval_seconds=600)

    # [⚔] Deploying Sentinel-Monitored Scheduled Tasks
    scheduler.add_job(
        func=tango,
        trigger=IntervalTrigger(seconds=600, timezone="Asia/Shanghai", jitter=5),
        name=job_name,
    )

    # [⚔] Gracefully run scheduler.
    scheduler.start()
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, EOFError):
        scheduler.shutdown(wait=False)
        logger.debug("Received keyboard interrupt signal.")


def run_sentinel(sitekey: str | None = None, timer: int = 300):
    logger.info("run sentinel")
    tango(sitekey=sitekey, timer=timer)
