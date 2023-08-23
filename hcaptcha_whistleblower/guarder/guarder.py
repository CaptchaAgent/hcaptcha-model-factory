# -*- coding: utf-8 -*-
# Time       : 2022/7/16 7:13
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from dataclasses import dataclass

from hcaptcha_challenger.agents.exceptions import ChallengePassed
from hcaptcha_challenger.agents.selenium import SeleniumAgent
from hcaptcha_challenger.agents.selenium import get_challenge_ctx
from hcaptcha_challenger.components.prompt_handler import BAD_CODE
from loguru import logger
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
    ElementNotInteractableException,
)
from selenium.webdriver.common.by import By

from hcaptcha_whistleblower.settings import Firebird


@dataclass
class GuarderV2(SeleniumAgent):
    silence: bool = False
    lang: str = "en"

    ctx_session = None

    _sitekey: str = "ace50dd0-0d68-44ff-931a-63b670c7eed7"
    monitor_site: str = f"https://accounts.hcaptcha.com/demo?sitekey={_sitekey}"

    firebird: Firebird = None

    def __post_init__(self):
        self.firebird = Firebird.from_static()

    @property
    def sitekey(self):
        return self._sitekey

    @sitekey.setter
    def sitekey(self, s: str | None):
        if s and isinstance(s, str):
            self._sitekey = s
            self.monitor_site = f"https://accounts.hcaptcha.com/demo?sitekey={s}"

    def __enter__(self):
        self.ctx_session = get_challenge_ctx(silence=self.silence, lang=self.lang)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.ctx_session:
                self.ctx_session.quit()
        except AttributeError:
            pass

    def flush_firebird(self, label: str):
        if label.lower().startswith("please click on the"):
            logger.info("Pass challenge", label=label, case="NotBinaryChallenge")
            return
        map_to = diagnose_task(label)
        self.firebird.to_json({label: map_to})
        self.firebird.flush()

        logger.success("将遇到的新挑战刷入运行时任务队列", label=label, map_to=map_to)

    def hacking_dataset(self, ctx):
        """Collector solution"""
        try:
            self.get_label(ctx)
            if "please click on the" in self._label.lower():
                refresh_hcaptcha(ctx)
                logger.warning("Pass challenge", label=self._label, case="NotBinaryChallenge")
                return
            self.mark_samples(ctx)
            self.download_images()
            refresh_hcaptcha(ctx)
        except (ChallengePassed, ElementClickInterceptedException):
            ctx.refresh()
        except StaleElementReferenceException:
            return
        except WebDriverException as err:
            logger.exception(err)
        finally:
            ctx.switch_to.default_content()

    def checking_dataset(self, ctx):
        """Sentinel solution"""
        try:
            # 进入挑战框架 | 开始执行相关检测任务
            self.get_label(ctx)
            # 拉起预警服务
            if not self.firebird.focus_labels.get(self._label):
                self.mark_samples(ctx)
                self.tactical_retreat()
                self.flush_firebird(self._label)
                return True
            # 在内联框架中刷新挑战
            refresh_hcaptcha(ctx)
        except (ChallengePassed, TimeoutException):
            ctx.refresh()
        except WebDriverException as err:
            logger.exception(err)
        finally:
            ctx.switch_to.default_content()


def diagnose_task(words: str) -> str:
    """from challenge prompt to model name"""
    origin = words
    if not words or not isinstance(words, str) or len(words) < 2:
        raise TypeError(f"({words})TASK should be string type data")

    # Filename contains illegal characters
    inv = {"\\", "/", ":", "*", "?", "<", ">", "|"}
    if s := set(words) & inv:
        raise TypeError(f"({words})TASK contains invalid characters({s})")

    # Normalized separator
    rnv = {" ", ",", "-"}
    for s in rnv:
        words = words.replace(s, "_")

    for code, right_code in BAD_CODE.items():
        words.replace(code, right_code)

    words = words.strip()
    logger.debug(f"diagnose task", origin=origin, to=words)

    return words


def refresh_hcaptcha(ctx) -> bool | None:
    try:
        return ctx.find_element(By.XPATH, "//div[@class='refresh button']").click()
    except (NoSuchElementException, ElementNotInteractableException):
        return False
