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
]
