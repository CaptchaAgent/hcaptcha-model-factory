"""
- DOCS: https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN?lang=zh-CN
- 机器人应用：https://open.feishu.cn/document/home/interactive-session-based-robot/create-app-request-permission
- 发送消息：https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create
- 开发者后台：https://open.feishu.cn/app?lang=zh-CN
"""
# -*- coding: utf-8 -*-
# Time       : 2022/7/15 20:48
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import json
import os
import time
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import Optional, Union
from urllib.request import getproxies

import pytz
import requests
from loguru import logger
from requests.exceptions import ConnectionError
from requests_toolbelt import MultipartEncoder


class TypeMessage:
    """发送消息 msg_type"""

    text = "text"
    post = "post"
    image = "image"
    file = "file"
    audio = "audio"
    media = "media"
    interactive = "interactive"
    share_chat = "share_chat"
    share_user = "share_user"


class LarkT:
    """guards"""

    def __init__(self, app_id: str, app_secret: str, debug: Optional[bool] = True):
        self.app_id = app_id
        self.app_secret = app_secret
        self.debug = debug

        self.action_name = "LarkT"
        self.access_token = ""
        self.expire = int(time.time())

    @staticmethod
    def _postman(
        url: str,
        headers: dict,
        body: Optional[dict] = None,
        data: Union[dict, MultipartEncoder] = None,
        params: Optional[dict] = None,
        files: Optional[dict] = None,
    ):
        """
        请求范式
        :param url: HTTP URL
        :param body: 请求体
        :param headers: 请求头
        :return: 响应体
        """
        _data: dict = {}
        _err = None
        try:
            resp = requests.post(
                url,
                json=body,
                headers=headers,
                data=data,
                params=params,
                files=files,
                proxies=getproxies(),
            )
        except ConnectionError:
            pass
        except Exception as err:
            logger.exception(err)
        else:
            try:
                _data = resp.json()
            except JSONDecodeError:
                pass
            else:
                code = _data.get("code")
                if code is not None:
                    if isinstance(code, int) and code != 0:
                        _err = _data.get("msg", "")
        finally:
            return _data, _err

    def require_tenant_access_token(self) -> Optional[str]:
        """
        更新应用 Token

        企业自建应用通过此接口获取 tenant_access_token，调用接口获取企业资源时，
        需要使用 tenant_access_token 作为授权凭证。

        - token 有效期为 2 小时，在此期间调用该接口 token 不会改变。
        - 当 token 有效期小于 30 分的时候，再次请求获取 token 的时候，会生成一个新的 token，与此同时老的 token 依然有效。

        https://open.feishu.cn/document/ukTMukTMukTM/ukDNz4SO0MjL5QzM/auth-v3/auth/tenant_access_token_internal

        :return: tenant_access_token
        """
        # Returns an existing token
        if self.access_token and self.expire > int(time.time()) + 60:
            return self.access_token

        # Require new token
        data, err = self._postman(
            url="https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            body={"app_id": self.app_id, "app_secret": self.app_secret},
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        if err or not data:
            logger.exception(err)

        self.access_token = f"Bearer {data.get('tenant_access_token', '')}"
        self.expire = data.get("expire", self.expire)

        return self.access_token

    def get_chat_id(self):
        """
        GET 获取群聊 ID

        https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/chat/list

        :return:
        """


class LarkAPI(LarkT):
    def __init__(self, app_id: str, app_secret: str, debug: Optional[bool] = True):
        super().__init__(app_id, app_secret, debug=debug)
        self.action_name = "LarkAPI"

    def translate(
        self, text: str, source_language: str = "zh", target_language: str = "en"
    ) -> Optional[str]:
        """
        文本翻译

        https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/ai/translation-v1/text/translate
        :param text:
        :param source_language:
        :param target_language:
        :return:
        """
        data, err = self._postman(
            url="https://open.feishu.cn/open-apis/translation/v1/text/translate",
            body={
                "source_language": source_language,
                "target_language": target_language,
                "text": text,
            },
            headers={
                "Authorization": self.require_tenant_access_token(),
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        if not data or err:
            logger.debug(
                "文本翻译异常", text=text, src=source_language, dst=target_language, response=data
            )
        return data.get("data", {}).get("text", "")

    def upload_files(self, file_type: str, file_path):
        file_type_list = ["opus", "mp4", "pdf", "doc", "xls", "ppt", "stream"]
        if file_type not in file_type_list:
            return logger.debug("文件格式异常", type=file_type, select_type=file_type_list)

        filename = os.path.basename(file_path)
        form = {"file_type": file_type, "file_name": filename, "file": open(file_path, "rb")}
        multi_form = MultipartEncoder(form)

        data, err = self._postman(
            url="https://open.feishu.cn/open-apis/im/v1/files",
            headers={
                "Authorization": self.require_tenant_access_token(),
                "Content-Type": multi_form.content_type,
            },
            data=multi_form,
        )
        if err or not data:
            logger.debug("上传文件异常", file_type=file_type, file_path=file_path, response=data)
        return data

    def upload_img(self, image_path: str):
        """
        上传图片

        https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/image/create

        :param image_path:
        :return:
        """
        if not os.path.exists(image_path):
            return logger.debug("文件不存在", image_path=image_path)

        form = {"image_type": "message", "image": (open(image_path, "rb"))}
        multi_form = MultipartEncoder(form)

        data, err = self._postman(
            url="https://open.feishu.cn/open-apis/im/v1/images",
            headers={
                "Authorization": self.require_tenant_access_token(),
                "Content-Type": multi_form.content_type,
            },
            data=multi_form,
        )
        if err or not data:
            logger.debug("图片上传失败", image_path=image_path, response=data)
        return data

    def send_group_msg(self, chat_id: str, content: dict, msg_type: str = TypeMessage.text):
        """
        发送群聊消息

        https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create
        :return:
        """
        msg_type_list = [
            "text",
            "post",
            "image",
            "file",
            "audio",
            "media",
            "sticker",
            "interactive",
            "share_chat",
            "share_user",
        ]
        if msg_type not in msg_type_list:
            return logger.debug("消息格式异常", type=msg_type, select_type=msg_type_list)

        data, err = self._postman(
            url="https://open.feishu.cn/open-apis/im/v1/messages",
            headers={
                "Authorization": self.require_tenant_access_token(),
                "Content-Type": "application/json; charset=utf-8",
            },
            params={"receive_id_type": "chat_id"},
            body={"receive_id": chat_id, "content": json.dumps(content), "msg_type": msg_type},
        )
        if err or not data:
            logger.debug("群消息发送失败", data=data)
        return data


class LarkAlert(LarkAPI):
    # Template content tag
    plain_text = "plain_text"
    lark_md = "lark_md"
    div = "div"
    hr = "hr"
    tag_img = "img"

    # 模版基础信息
    project_name = "hcaptcha-challenger"
    project_link = "https://github.com/QIN2DIM/hcaptcha-challenger"
    project_md = f"[{project_name}]({project_link})"

    def __init__(self, app_id: str, app_secret: str):
        super().__init__(app_id, app_secret, debug=True)
        self.action_name = "LarkAlert"

    def fire_card(self, chat_id: str, hcaptcha_link: str, label_name: str, challenge_img_key: str):
        """
        发送 <New Challenge> 消息卡片

        - 发送消息：https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create
        - 卡片模版：https://open.feishu.cn/tool/cardbuilder

        :param chat_id:
        :param hcaptcha_link: Demo links with sitekey
        :param label_name: Clean label
        :param challenge_img_key: use lark to upload challenge image and get the key of the resource
        :return:
        """

        header_title = "1 级警报 - New Challenge"

        sender_date = str(datetime.now(pytz.timezone("Asia/Shanghai"))).split(".")[0]

        hcaptcha_demo = "hCAPTCHA演示"
        hcaptcha_md = f"[{hcaptcha_demo}]({hcaptcha_link})"

        template = {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "red",
                "title": {"content": header_title, "tag": self.plain_text},
            },
            "elements": [
                {
                    "fields": [
                        {
                            "is_short": True,
                            "text": {"content": f"**🕐 时间：**\n {sender_date}", "tag": self.lark_md},
                        },
                        {
                            "is_short": True,
                            "text": {
                                "content": f"**📋 项目：**\n {self.project_md}",
                                "tag": self.lark_md,
                            },
                        },
                        {"is_short": False, "text": {"content": " ", "tag": self.plain_text}},
                        {
                            "is_short": True,
                            "text": {
                                "content": f"**🚳 挑战链接：**\n {hcaptcha_md}",
                                "tag": self.lark_md,
                            },
                        },
                        {
                            "is_short": True,
                            "text": {"content": f"**🔖 挑战标签：**\n {label_name}", "tag": self.lark_md},
                        },
                    ],
                    "tag": self.div,
                },
                {"tag": self.hr},
                {
                    "alt": {"content": "", "tag": self.plain_text},
                    "img_key": challenge_img_key,
                    "tag": self.tag_img,
                },
            ],
        }

        return self.send_group_msg(chat_id, content=template, msg_type=TypeMessage.interactive)


def broadcast_alert_information(app_id, app_secret):
    """广播预警信息"""
    lark = LarkAlert(app_id, app_secret)

    # 缓存加锁
    # key = lock_challenge()

    # 检查挑战截图的路径
    # challenge_img_key = ""
    # if os.path.exists(path_screenshot):
    #     resp_upload_img = lark.upload_img(path_screenshot)
    #     challenge_img_key = resp_upload_img.get("data", {}).get("image_key", "")
    #     path_screenshot = ""

    # 向研发部门群组发送报警卡片
    # resp_fire_card = lark.fire_card(
    #     chat_id=config.lark.chat_id,
    #     hcaptcha_link=self.monitor_site,
    #     label_name=self._label_alias.get(self._label, self._label),
    #     challenge_img_key=challenge_img_key,
    # )

    # 推送失败 | 解锁缓存，继续挑战
    # if resp_fire_card.get("code", -1) != 0:
    #     self.unlock_challenge(key)
    #     return logger.error(
    #         "Lark 报警消息推送失败",
    #         challenge_img_key=challenge_img_key,
    #         path_screenshot=self.path_screenshot,
    #         response=resp_fire_card,
    #     )
    # 推送成功 | 保持缓存锁定状态，避免重复警报
    return logger.success("Lark 报警消息推送成功")
