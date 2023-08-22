# -*- coding: utf-8 -*-
# Time       : 2023/8/21 0:32
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import json
from pathlib import Path

sp = Path("../logs/serialize.log")
label_set = set()

try:
    with open(sp, "r", encoding="utf8") as file:
        for line in file:
            data = json.loads(line)
            if label := data["record"]["extra"].get("name"):
                label_set.add(label)
except FileNotFoundError as err:
    print(err)

label_set = sorted(label_set)
for ls in label_set:
    print(ls)
