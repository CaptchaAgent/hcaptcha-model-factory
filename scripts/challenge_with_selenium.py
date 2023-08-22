# -*- coding: utf-8 -*-
# Time       : 2022/9/23 17:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import time
from pathlib import Path

import hcaptcha_challenger as solver
from hcaptcha_challenger.agents.exceptions import ChallengePassed
from hcaptcha_challenger.agents.selenium import ArmorUtils
from hcaptcha_challenger.agents.selenium import SeleniumAgent
from hcaptcha_challenger.agents.selenium import get_challenge_ctx
from hcaptcha_challenger.agents.skeleton import Status
from loguru import logger

from hcaptcha_whistleblower.settings import SiteKey

# Init local-side of the ModelHub
solver.install()

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")

test_url = f"https://accounts.hcaptcha.com/demo?sitekey={SiteKey.epic}"


@logger.catch
def hit_challenge(ctx, challenger: SeleniumAgent, retries: int = 2) -> bool | None:
    if ArmorUtils.face_the_checkbox(ctx):
        challenger.anti_checkbox(ctx)

    for _ in range(retries):
        try:
            if (resp := challenger.anti_hcaptcha(ctx)) is None:
                ArmorUtils.refresh(ctx)
                time.sleep(1)
                continue
            if resp == Status.CHALLENGE_SUCCESS:
                return True
        except ChallengePassed:
            return True
        ArmorUtils.refresh(ctx)
        time.sleep(1)


def bytedance():
    # New Challenger
    challenger = SeleniumAgent.from_modelhub(tmp_dir=tmp_dir)

    # Replace selenium.webdriver.Chrome with CTX
    with get_challenge_ctx(silence=False) as ctx:
        ctx.get(test_url)
        hit_challenge(ctx=ctx, challenger=challenger)

        sp = tmp_dir.joinpath(f"bytedance - headless .png")
        logger.success("quit tasks", save_screenshot=sp)
        ctx.save_screenshot(sp)
        time.sleep(3)


if __name__ == "__main__":
    bytedance()
