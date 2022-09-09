import re
import sys
import typing

from loguru import logger


class ToolBox:
    """Portable Toolbox"""

    IMAGE_EXT = ["jpg", "jpeg", "png", "ppm", "bmp", "tif", "tiff", "webp", "jfif"]

    @staticmethod
    def init_log(**sink_path):
        """Initialize loguru log information"""
        event_logger_format = (
            "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | "
            "<lvl>{level}</lvl> - "
            # "<c><u>{name}</u></c> | "
            "{message}"
        )
        logger.remove()
        logger.add(
            sink=sys.stdout,
            colorize=True,
            level="DEBUG",
            format=event_logger_format,
            diagnose=False,
        )
        if sink_path.get("error"):
            logger.add(
                sink=sink_path.get("error"),
                level="ERROR",
                rotation="1 week",
                encoding="utf8",
                diagnose=False,
            )
        if sink_path.get("runtime"):
            logger.add(
                sink=sink_path.get("runtime"),
                level="DEBUG",
                rotation="20 MB",
                retention="20 days",
                encoding="utf8",
                diagnose=False,
            )
        return logger

    @staticmethod
    def runtime_report(
        action_name: str, motive: str = "RUN", message: str = "", **params
    ) -> str:
        """格式化输出"""
        flag_ = f">> {motive} [{action_name}]"
        if message != "":
            flag_ += f" {message}"
        if params:
            flag_ += " - "
            flag_ += " ".join([f"{i[0]}={i[1]}" for i in params.items()])

        return flag_

    @staticmethod
    def split_prompt(prompt_message: str, lang: str = "en") -> typing.Optional[str]:
        if prompt_message and isinstance(prompt_message, str):
            prompt_message = prompt_message.strip()
            return {
                "zh": re.split(r"[包含 图片]", prompt_message)[2][:-1].replace("的每", "")
                if "包含" in prompt_message
                else prompt_message,
                "en": re.split(r"containing a", prompt_message)[-1][1:]
                .strip()
                .replace(".", "")
                if "containing" in prompt_message
                else prompt_message,
            }.get(lang)

    @staticmethod
    def is_image(filename: str) -> typing.Optional[bool]:
        """Check if the file is an image file"""

        return filename.split(".")[-1] in ToolBox.IMAGE_EXT if filename else None
