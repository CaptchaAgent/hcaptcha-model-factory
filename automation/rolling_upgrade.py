# -*- coding: utf-8 -*-
# Time       : 2023/10/1 22:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import webbrowser

from loguru import logger

from annotator import Annotator


def rolling_upgrade(asset_id=None):
    if not asset_id:
        return

    try:
        annotator = Annotator(asset_id)
        annotator.execute()
        webbrowser.open(Annotator.repo.html_url)
    except Exception as err:
        logger.warning(err)


if __name__ == "__main__":
    rolling_upgrade()
