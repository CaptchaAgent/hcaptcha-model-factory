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
        "positive_labels": ["forest"],
        "negative_labels": ["hedgehog", "city building", "laptop"],
        "joined_dirs": ["forest"],
    },
    {
        "positive_labels": ["car"],
        "negative_labels": [
            "flower",
            "suit",
            "plant",
            "table",
            "guitar",
            "animal",
            "tree",
            "bicycle",
            "bus",
            "machine that flies",
            "door",
            "water vehicle",
        ],
        "joined_dirs": ["car"],
    },
    {
        "positive_labels": ["city street"],
        "negative_labels": ["electronic device", "plant", "laptop", "forest"],
        "joined_dirs": ["city_street"],
    },
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
    {
        "positive_labels": ["microwave"],
        "negative_labels": ["panda", "goat", "headphones", "computer", "office places"],
        "joined_dirs": ["microwave"],
    },
    {
        "positive_labels": ["forklift"],
        "negative_labels": ["butterfly", "bird", "cable"],
        "joined_dirs": ["industrial_machinery"],
    },
    {
        "positive_labels": ["lion", "kangaroo"],
        "negative_labels": ["butterfly", "shark", "whale", "dolphin", "jeyllyfish"],
        "joined_dirs": ["land_animal"],
    },
    {
        "positive_labels": ["animal", "fish", "dog", "bird"],
        "negative_labels": [
            "robot",
            "boat",
            "cables",
            "circuit board",
            "forklift",
            "excavator",
            "bus",
            "vehicle",
        ],
        "joined_dirs": ["animal"],
    },
    {
        "positive_labels": ["flying animal", "bird", "owl"],
        "negative_labels": ["elephant", "lion", "land animal", "whale", "dolphin", "water animal"],
        "joined_dirs": ["flying_animal"],
    },
    {
        "positive_labels": ["water animal", "jellyfish"],
        "negative_labels": ["land animal", "flying animal", "butterfly", "bird"],
        "joined_dirs": ["water_animal"],
    },
    {
        "positive_labels": ["clock"],
        "negative_labels": ["animal", "suitcase", "guitar"],
        "joined_dirs": ["clock"],
    },
    {
        "positive_labels": ["motorized machine", "tractor"],
        "negative_labels": ["plant", "flower", "mountain", "laptop", "river", "natural", "chess"],
        "joined_dirs": ["motorized_machine"],
    },
    {
        "positive_labels": ["electric components", "cable"],
        "negative_labels": ["animal", "boat", "bird", "butterfly"],
        "joined_dirs": ["electric_components"],
    },
    {
        "positive_labels": ["cable"],
        "negative_labels": [
            "animal",
            "boat",
            "bird",
            "butterfly",
            "circuit board",
            "forklift",
            "excavator",
        ],
        "joined_dirs": ["electric_cables"],
    },
    {
        "positive_labels": ["volcano"],
        "negative_labels": ["plant", "stage"],
        "joined_dirs": ["volcano"],
    },
    {
        "positive_labels": ["land vehicle", "truck"],
        "negative_labels": [
            "flying vehicle",
            "helicopter",
            "water vehicle",
            "airplane",
            "boat",
            "seaplane",
        ],
        "joined_dirs": ["land_vehicle"],
    },
    {
        "positive_labels": ["machine that flies", "propeller"],
        "negative_labels": ["water vehicle", "land vehicle", "boat", "truck"],
        "joined_dirs": ["machine_that_flies"],
    },
    {
        "positive_labels": ["server"],
        "negative_labels": ["cat", "elephant", "airplane"],
        "joined_dirs": ["server"],
    },
    {
        "positive_labels": ["movie theater"],
        "negative_labels": ["flashlight", "house", "playground", "tape"],
        "joined_dirs": ["movie_theater"],
    },
    {
        "positive_labels": ["smartphone"],
        "negative_labels": ["glass", "pen", "car", "clothes", "chair", "suit"],
        "joined_dirs": ["smartphone"],
    },
    {
        "positive_labels": ["suit"],
        "negative_labels": [
            "vehicle",
            "paper",
            "factory",
            "machine",
            "neon light",
            "notebook",
            "boat",
        ],
        "joined_dirs": ["business_suit"],
    },
    {
        "positive_labels": ["dragon"],
        "negative_labels": ["cat", "smartphone", "tiger", "elephant", "bird"],
        "joined_dirs": ["dragon"],
    },
    {
        "positive_labels": ["hatchback car"],
        "negative_labels": ["cat", "server", "elephant"],
        "joined_dirs": ["hatchback_car"],
    },
    {
        "positive_labels": ["panda"],
        "negative_labels": [
            "keyboard",
            "dragon",
            "dog",
            "bird",
            "elephant",
            "suitcase",
            "electronic device",
            "goat",
        ],
        "joined_dirs": ["panda"],
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
        "negative_labels": [
            "water animal",
            "land animal",
            "panda",
            "giraffe",
            "dolphins",
            "lion",
            "cow",
            "hedgehog",
            "dog",
            "fox",
            "cat",
        ],
        "joined_dirs": ["the_smallest_animal", "s1_bird"],
        "substack": {
            "nested_smallest_bird": {
                "yes": ["bird"],
                "bad": ["panda", "giraffe", "dolphins", "lion", "dog", "fox", "cat"],
            }
        },
    },
    {
        "positive_labels": ["penguin"],
        "negative_labels": ["ant", "crab", "ladybug"],
        "joined_dirs": ["the_largest_animal", "l1_penguin"],
        "substack": {
            "nested_largest_penguin": {"yes": ["penguin"], "bad": ["ant", "crab", "ladybug"]}
        },
    },
    {
        "positive_labels": ["zebra"],
        "negative_labels": ["elephant", "whale"],
        "joined_dirs": ["the_smallest_animal", "s1_zebra"],
        "substack": {"nested_smallest_zebra": {"yes": ["zebra"], "bad": ["elephant", "whale"]}},
    },
    {
        "positive_labels": ["turtle"],
        "negative_labels": ["horse", "lion", "giraffe", "shark", "dog"],
        "joined_dirs": ["the_smallest_animal", "s1_turtle"],
        "substack": {
            "nested_smallest_turtle": {
                "yes": ["turtle"],
                "bad": ["horse", "lion", "giraffe", "shark", "dog"],
            }
        },
    },
    {
        "positive_labels": ["capybara"],
        "negative_labels": ["bird", "chicken", "elephant", "crab"],
        "joined_dirs": ["the_largest_animal", "l1_capybara"],
        "substack": {
            "nested_smallest_capybara": {
                "yes": ["capybara"],
                "bad": ["bird", "chicken", "elephant", "crab"],
            }
        },
    },
    {
        "positive_labels": ["cactus", "conference room"],
        "negative_labels": ["bed", "snowman", "snow house"],
        "joined_dirs": ["images_that_appear_warmer_in_comparison_to_other", "w1_cactus"],
        "substack": {
            "nested_warmer_cactus": {"yes": ["cactus"], "bad": ["bed", "conference room"]},
            "nested_warmer_conference_room": {
                "yes": ["conference room"],
                "bad": ["snowman", "snow house"],
            },
        },
    },
    {
        "positive_labels": ["mouse"],
        "negative_labels": ["door", "pen", "chair"],
        "joined_dirs": ["electronic_device", "e1_mouse"],
        "substack": {
            "nested_electronic_device_mouse": {"yes": ["mouse"], "bad": ["door", "pen", "chair"]}
        },
    },
    {
        "positive_labels": ["helicopter"],
        "negative_labels": ["truck", "boat", "water vehicle", "land vehicle"],
        "joined_dirs": ["machine_that_flies"],
        "substack": {
            "nested_machine_that_flies_helicopter": {
                "yes": ["helicopter"],
                "bad": ["truck", "boat", "water vehicle", "land vehicle"],
            }
        },
    },
    {
        "positive_labels": ["circuit board"],
        "negative_labels": ["animal", "forklift"],
        "joined_dirs": ["electronic_device", "e1_circuit_board"],
        "substack": {
            "nested_electronic_device_circuit_board": {
                "yes": ["circuit board"],
                "bad": ["animal", "forklift"],
            }
        },
    },
    {
        "positive_labels": ["squirrel"],
        "negative_labels": ["bee", "ladybug"],
        "joined_dirs": ["the_largest_animal_in_real_life", "l2_squirrel"],
        "substack": {"nested_largest_squirrel": {"yes": ["squirrel"], "bad": ["bee", "ladybug"]}},
    },
    {
        "positive_labels": ["owl"],
        "negative_labels": ["land animal", "water animal"],
        "joined_dirs": ["the_smallest_animal_species", "s2_owl"],
        "substack": {"nested_smallest_owl": {"yes": ["squirrel"], "bad": ["bee", "ladybug"]}},
    },
    {
        "positive_labels": ["lion"],
        "negative_labels": ["cat", "dog", "goat", "fox", "ladybug", "bee"],
        "joined_dirs": ["the_largest_animal_in_real_life", "l2_lion"],
        "substack": {
            "nested_largest_lion": {
                "yes": ["lion"],
                "bad": ["cat", "dog", "goat", "fox", "ladybug", "bee"],
            }
        },
    },
    {
        "positive_labels": ["lion"],
        "negative_labels": ["elephant", "whale", "panda"],
        "joined_dirs": ["the_smallest_animal_in_real_life", "s2_lion"],
        "substack": {
            "nested_smallest_lion": {"yes": ["lion"], "bad": ["elephant", "whale", "panda"]}
        },
    },
]
