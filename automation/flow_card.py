# -*- coding: utf-8 -*-
# Time       : 2023/10/26 2:58
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
# Run `assets_manager.py` to get test data from GitHub issues

flow_card = [
    {
        "positive_labels": ["off-road vehicle"],
        "negative_labels": ["car", "bicycle"],
        "joined_dirs": ["off_road_vehicle"],
    },
    {
        "positive_labels": ["bicycle"],
        "negative_labels": ["car", "off-road vehicle"],
        "joined_dirs": ["bicycle"],
    },
    {
        "positive_labels": ["furniture", "chair"],
        "negative_labels": ["guitar", "keyboard", "game tool", "headphones"],
        "joined_dirs": ["furniture"],
    },
    {
        "positive_labels": ["sedan car"],
        "negative_labels": ["bicycle", "off-road vehicle"],
        "joined_dirs": ["sedan_car"],
    },
    {
        "positive_labels": ["turtle"],
        "negative_labels": ["horse", "bear", "giraffe", "dolphins"],
        "joined_dirs": ["please_click_on_the_smallest_animal", "nested_smallest_turtle"],
    },
    {
        "positive_labels": ["dog"],
        "negative_labels": ["frog", "hedgehog", "squirrel", "hummingbird"],
        "joined_dirs": ["please_click_on_the_largest_animal", "nested_largest_dog"],
    },
    # multi classification for nested prompt
    {
        "positive_labels": ["dog", "fox"],
        "negative_labels": ["crab", "bird", "dragonfly", "ant"],
        "joined_dirs": ["the_largest_animal"],
        # = ↑↑ = 和常规情况一样，先对整体数据集进行多目标分类
        # = ↓↓ = 再根据具体的 yes/bad 映射关系进行数据集二次移动
        "substack": {
            "nested_largest_dog": {"yes": ["dog"], "bad": ["crab", "bird", "dragonfly", "ant"]},
            "nested_largest_fox": {"yes": ["fox"], "bad": ["crab", "bird", "dragonfly", "ant"]},
        },
    },
]
