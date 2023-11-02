# -*- coding: utf-8 -*-
# Time       : 2023/10/26 2:58
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
# Run `assets_manager.py` to get test data from GitHub issues
import logging
import sys

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        "positive_labels": ["red panda"],
        "negative_labels": ["cactus", "door", "guinea pig", "meerkat", "bird"],
        "joined_dirs": ["red_panda"],
    },
    {
        "positive_labels": ["natural landscape", "Mountain", "forest"],
        "negative_labels": [
            "chess",
            "laptop",
            "helicopter",
            "meerkat",
            "roller coaster",
            "Recreational facilities",
        ],
        "joined_dirs": ["natural_landscape"],
    },
    {
        "positive_labels": ["keyboard"],
        "negative_labels": ["panda", "goat", "headphones", "bird", "trunk"],
        "joined_dirs": ["keyboard"],
    },
]

flow_card_nested_animal = [
    {
        "positive_labels": ["panda"],
        "negative_labels": ["raccoon", "dog", "meerkat", "koala"],
        "joined_dirs": ["the_largest_animal", "l1_panda"],
        "substack": {
            "nested_largest_panda": {
                "yes": ["panda"],
                "bad": ["raccoon", "dog", "meerkat", "koala"],
            }
        },
    },
    {
        "positive_labels": ["horse"],
        "negative_labels": ["elephant", "whale"],
        "joined_dirs": ["the_smallest_animal", "s1_horse"],
        "substack": {"nested_smallest_horse": {"yes": ["horse"], "bad": ["elephant", "whale"]}},
    },
    {
        "positive_labels": ["bird"],
        "negative_labels": ["ladybug", "butterfly", "dragonfly", "bees", "crab", "frog", "ant"],
        "joined_dirs": ["the_largest_animal", "l1_bird"],
        "substack": {
            "nested_largest_bird": {
                "yes": ["bird"],
                "bad": ["ladybug", "butterfly", "dragonfly", "bees", "crab", "frog", "ant"],
            }
        },
    },
    {
        "positive_labels": ["bird"],
        "negative_labels": ["panda", "giraffe", "dolphins", "lion"],
        "joined_dirs": ["the_smallest_animal", "s1_bird"],
        "substack": {
            "nested_smallest_bird": {
                "yes": ["bird"],
                "bad": ["panda", "giraffe", "dolphins", "lion"],
            }
        },
    },
    {
        "positive_labels": ["penguin"],
        "negative_labels": ["ant", "crab", "ladybug"],
        "joined_dirs": ["the_largest_animal", "l1_penguin"],
        "substack": {
            "nested_largest_penguin": {
                "yes": ["penguin"],
                "bad": ["ant", "crab", "ladybug"],
            }
        }
    }
]
