# -*- coding: utf-8 -*-
# Time       : 2023/3/19 19:46
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description: 查看 prompt 被切割前后的样子
from __future__ import annotations

import re

from hcaptcha_challenger.components.prompt_handler import split_prompt_message
from loguru import logger


def launch(prompt: str, lang: str | None = None):
    # Filename contains illegal characters
    inv = {"\\", "/", ":", "*", "?", "<", ">", "|"}
    if s := set(prompt) & inv:
        raise TypeError(f"({prompt})TASK contains invalid characters({s})")

    if lang is None:
        lang = "zh" if re.compile("[\u4e00-\u9fa5]+").search(prompt) else "en"
    flag = split_prompt_message(prompt, lang)

    # Normalized separator
    rnv = {" ", ",", "-"}
    name = flag
    for s in rnv:
        name = name.replace(s, "_")

    logger.success(f'"{flag}": "{name}",  # {prompt}')


def cases():
    prompts = [
        "请单击包含水下乌龟的每个图像",
        "请点击包含水下鱼的每张图片",
        "请点击每张包含花盆中枯死和干燥植物的图片",
        "请点击每个包含玩具兔子的图片",
        "请点击每张包含ー只飞翔的鸟的图片",
        "请点击每个包含破碎玻璃瓶的图片",
        "请单击每个包含白腿马的图像。",
        "请点击草丛中包含兔子的每张图片",
        "请点击草丛中包含乌龟的每张图片",
        "请点击每张包含会议室的图片",
        "请点击每张包含类似瓷器设计图案的茶杯的图片",
        "请点击每张包含兔子在水中游泳的图片",
        "请点击每个包含玩具屋的图片",
        "请点击每张包含被鲜花包围的企鹅的图片",
        "请单击每个包含水中鸭子的图像",
        "请点击每张包含岩石后面企鹅的图片",
        "请单击每个包含冰上企鹅的图片",
        "请单击每个包含盆中仙人掌的图像",
        "请点击每张包含蜜蜂在花朵附近飞翔的图片",
        "请点击包含人们通常戴在头上的物品的每张图片",
        "请点击每张包含海滩房屋的图片",
        "请单击每个包含沙中仙人掌的图像",
        "请点击每张包含危险海域的图片",
        "请单击包含人们通常脚上穿着的物品的每张图片",
        "请单击包含人们通常穿在裤子上的物品的每张图片",
        "请点击每张包含冷饮的图片",
        "请点击树枝上包含蜂鸟的每图片",
        "请点击每张包含骑自行车的骷髅的图片",
        "请点击每张包含鬼屋图画的图片",
        "请单击每个包含篮子中橙子的图片",
        "Please click each image containing a rabbіt.",
        "Please click each image containing oranges in a basket",
        "Please click each image containing a skeleton ridding a bicycle",
        "Please click each image containing lemons in a basket",
        "请点击每张海滩上包含一只鸡的图片",
        "请单击每个包含树上的鸡的图像",
        "请点击每个包含树上松鼠的图像",
        "Please click each image containing a banana in a basket",
        "请点击每张包含被鲜花包围的刺猬的图片",
        "请单击每个包含树上苹果的图像",
        "请点击每个包含篮子里的草莓的图片",
        "请点击每张包含蝴蝶的图片",
        "Please click each image containing a butterfly",
        "Please click each image containing an apple on a tree",
        "Please click each image containing a strawberry in a basket",
        "Please click each image containing a daisy",
        "请点击所有包含高层建筑的图片",
        "请单击每个包含花园中红玫瑰的图像",
        "Please click each image containing red roses in a garden",
        "请点击田野中包含向日葵的每张图片",
        "请点击每张包含打高尔夫球的图片",
        "Please click each image containing someone playing golf",
        "请单击每个包含盘子上的鸡蛋的图像",
        "请点击每张包含戴墨镜的狗的图片",
        "请点击每张包含有人打曲棍球的图片",
    ]

    for prompt_ in prompts:
        launch(prompt_)


if __name__ == "__main__":
    cases()
