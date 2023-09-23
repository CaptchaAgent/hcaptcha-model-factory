import os
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

import httpx
from bs4 import BeautifulSoup
from httpx import Client


@dataclass
class AssetsManager:
    sources: str = field(default=str)

    client: Client = field(default=Client)

    this_dir = Path(__file__).parent
    project_dir = Path(__file__).parent.parent
    to_dir = project_dir.joinpath("database2309")
    cache_path = this_dir.joinpath("assets_cache.txt")
    local_from_dir: Path = this_dir.joinpath("tmp_dir/image_label_binary")

    _cached_assets: Set[str] = field(default_factory=set)

    def __post_init__(self):
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.76"
        }
        self.client = httpx.Client(headers=headers, follow_redirects=True)
        self._cached_assets = set()

    def __del__(self):
        text = "\n".join(self._cached_assets)
        self.cache_path.write_text(text)

    @classmethod
    def from_sources(cls, s: str):
        return cls(sources=s)

    def execute(self):
        self.load_assets_cache()

        if not self.sources.startswith("https://"):
            target_dir = self.unpack_datasets(self.sources, self.local_from_dir)
        else:
            target_dir = self.download_datasets(self.sources)

        # Annotate your images
        if "win32" in sys.platform and target_dir:
            os.startfile(target_dir)

    def load_assets_cache(self):
        if self.cache_path:
            self._cached_assets = set(self.cache_path.read_text(encoding="utf8").split("\n"))

    def unpack_datasets(self, source: str, from_dir: Path):
        fd = from_dir.joinpath(source)
        if not fd.exists():
            return

        # Normalized separator
        task_name = source
        rnv = {" ", ",", "-"}
        for s in rnv:
            task_name = task_name.replace(s, "_")

        td = self.to_dir.joinpath(task_name)
        td.mkdir(parents=True, exist_ok=True)

        self.merge(fd, td)

        return td

    def download_datasets(self, issue_url):
        # https://github.com/captcha-challenger/hcaptcha-whistleblower/releases/download/automation-archive/image_label_binary.industrial.tool.or.machinery.202309201646375863.zip
        td: Path | None = None

        print(f">> Parse issue url - link={issue_url}")
        for url in self.get_download_links(issue_url):
            if url in self._cached_assets:
                continue
            self._cached_assets.add(url)
            res = self.client.get(url)
            zip_path = self.to_dir.joinpath(f"{url.split('/')[-1]}")
            zip_path.write_bytes(res.content)

            task_name = " ".join(url.split("/")[-1].split(".")[1:-2])
            rnv = {" ", ",", "-"}
            for s in rnv:
                task_name = task_name.replace(s, "_")

            td = self.to_dir.joinpath(task_name)
            td.mkdir(exist_ok=True)

            td_tmp = self.to_dir.joinpath(f"{task_name}.tmp")
            td_tmp.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(td_tmp)

            self.merge(td_tmp, td)

            os.remove(zip_path)
            shutil.rmtree(td_tmp, ignore_errors=True)

        return td

    def get_download_links(self, issue_url: str):
        prefix = "https://github.com/captcha-challenger/hcaptcha-whistleblower/releases/download/automation-archive/"

        res = self.client.get(issue_url)
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

    def merge(self, fd: Path, td: Path):
        td_img_names = {img for _, _, ina in os.walk(td) for img in ina}
        count = 0
        for i in os.listdir(fd):
            if i not in td_img_names:
                shutil.move(fd.joinpath(i), td.joinpath(i))
                count += 1
        td.joinpath("yes").mkdir(exist_ok=True)
        td.joinpath("bad").mkdir(exist_ok=True)
        print(f">> Merge datasets - {count=} to_dir={td} {self.sources=}")


def run():
    sources = "https://github.com/QIN2DIM/hcaptcha-challenger/issues/734"
    am = AssetsManager.from_sources(sources)
    am.execute()


if __name__ == "__main__":
    run()
