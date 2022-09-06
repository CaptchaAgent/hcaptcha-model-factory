import typing

from loguru import logger

BADCODE = {
    "а": "a",
    "е": "e",
    "e": "e",
    "i": "i",
    "і": "i",
    "ο": "o",
    "с": "c",
    "ԁ": "d",
    "ѕ": "s",
    "һ": "h",
    "у": "y",
    "р": "p",
}


def diagnose_task(task_name: typing.Optional[str]) -> typing.Optional[str]:
    """Input detection and normalization"""
    if not task_name or not isinstance(task_name, str) or len(task_name) < 2:
        raise TypeError(f"({task_name})TASK should be string type data")

    # Filename contains illegal characters
    inv = {"\\", "/", ":", "*", "?", "<", ">", "|"}
    if s := set(task_name) & inv:
        raise TypeError(f"({task_name})TASK contains invalid characters({s})")

    # Normalized separator
    rnv = {" ", ",", "-"}
    for s in rnv:
        task_name = task_name.replace(s, "_")

    # Convert bad code
    for code in BADCODE:
        task_name.replace(code, BADCODE[code])

    task_name = task_name.strip()
    logger.debug(f"Diagnose task | task_name={task_name}")

    return task_name
