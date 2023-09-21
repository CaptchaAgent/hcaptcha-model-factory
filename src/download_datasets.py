# -*- coding: utf-8 -*-
# Time       : 2023/8/22 0:28
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description: 将 hcaptcha-challenger 中的缓存解包到对应的待分类文件夹
import os
import shutil
import sys
import zipfile
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

project_dir = Path(__file__).parent.parent
to_dir = project_dir.joinpath("database2309")

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.76"
}
client = httpx.Client(headers=headers, follow_redirects=True)
prefix = "https://github.com/captcha-challenger/hcaptcha-whistleblower/releases/download/automation-archive/"


def get_download_links(issue_url: str):
    res = client.get(issue_url, headers=headers, follow_redirects=True)
    soup = BeautifulSoup(res.text, "html.parser")
    link_tags = soup.find_all("a", href=lambda href: href and prefix in href)
    for link in link_tags:
        real_link = link["href"]
        if not isinstance(real_link, str):
            continue
        real_link = real_link.strip()
        if not real_link.endswith(".zip"):
            continue
        yield real_link


def merge(fd: Path, td: Path):
    td_img_names = {img for _, _, ina in os.walk(td) for img in ina}
    count = 0
    for i in os.listdir(fd):
        if i not in td_img_names:
            shutil.move(fd.joinpath(i), td.joinpath(i))
            count += 1
    td.joinpath("yes").mkdir(exist_ok=True)
    td.joinpath("bad").mkdir(exist_ok=True)
    print(f">> 合并数据集 - {count=} to_dir={td} {sources=}")


def unpack_datasets(source: str, from_dir: str = ""):
    if not from_dir:
        return

    fd = Path(from_dir).joinpath(source)
    if not fd.exists():
        return

    # Normalized separator
    task_name = source
    rnv = {" ", ",", "-"}
    for s in rnv:
        task_name = task_name.replace(s, "_")

    td = to_dir.joinpath(task_name)
    td.mkdir(parents=True, exist_ok=True)

    merge(fd, td)

    return td


def download_datasets(issue_url):
    # https://github.com/captcha-challenger/hcaptcha-whistleblower/releases/download/automation-archive/image_label_binary.industrial.tool.or.machinery.202309201646375863.zip
    td: Path | None = None

    print(f">> Parse issue url - link={issue_url}")
    for url in get_download_links(issue_url):
        res = client.get(url, headers=headers, follow_redirects=True)
        zip_path = to_dir.joinpath(f"{url.split('/')[-1]}")
        zip_path.write_bytes(res.content)

        task_name = ' '.join(url.split('/')[-1].split('.')[1:-2])
        rnv = {" ", ",", "-"}
        for s in rnv:
            task_name = task_name.replace(s, "_")

        td = to_dir.joinpath(task_name)
        td.mkdir(exist_ok=True)

        td_tmp = to_dir.joinpath(f"{task_name}.tmp")
        td_tmp.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td_tmp)

        merge(td_tmp, td)

        os.remove(zip_path)
        shutil.rmtree(td_tmp, ignore_errors=True)

    return td


if __name__ == "__main__":
    sources: str = "https://github.com/QIN2DIM/hcaptcha-challenger/issues/725"

    target_dir = ""
    local_from_dir = (
        r"E:\_GithubProjects\Sources\hcaptcha-challenger\automation\tmp_dir\image_label_binary"
    )
    if not sources.startswith("https://"):
        target_dir = unpack_datasets(sources, local_from_dir)
    else:
        target_dir = download_datasets(sources)

    # Annotate your images
    if "win32" in sys.platform and target_dir:
        os.startfile(target_dir)
