from __future__ import annotations

import asyncio
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path

from hcaptcha_challenger.agents import Malenia, AgentT
from hcaptcha_challenger.utils import SiteKey
from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

collected = []
per_times = 60
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")

# sitekey = "58366d97-3e8c-4b57-a679-4a41c8423be3"
sitekey = SiteKey.epic


async def collete_datasets(context: ASyncContext):
    page = await context.new_page()
    agent = AgentT.from_page(page=page, tmp_dir=tmp_dir)

    sitelink = SiteKey.as_sitelink(sitekey)
    await page.goto(sitelink)

    logger.info("startup collector", url=sitelink)

    await agent.handle_checkbox()

    for pth in range(1, per_times + 1):
        with suppress(Exception):
            t0 = time.time()
            label = await agent.collect()
            te = f"{time.time() - t0:.2f}s"
            probe = list(agent.qr.requester_restricted_answer_set.keys())
            mixed_label = probe[0] if len(probe) > 0 else label
            collected.append(mixed_label)
            print(f">> COLLETE - progress=[{pth}/{per_times}] timeit={te} {label=} {probe=}")

        await page.wait_for_timeout(500)
        fl = page.frame_locator(agent.HOOK_CHALLENGE)
        await fl.locator("//div[@class='refresh button']").click()


@logger.catch
async def bytedance():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="en-US")
        await Malenia.apply_stealth(context)
        await collete_datasets(context)
        await context.close()

    print(f"\n>> RESULT - {Counter(collected)=}")


if __name__ == "__main__":
    asyncio.run(bytedance())
