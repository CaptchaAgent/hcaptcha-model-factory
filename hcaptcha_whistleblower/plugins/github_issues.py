# -*- coding: utf-8 -*-
# Time       : 2023/3/21 17:50
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description: https://docs.github.com/en/rest/issues/issues
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Literal

from httpx import Client

ISSUE_TEMPLATE_CHALLENGE_GENERAL = """
### Prompt[en]

{prompt}

### New types of challenge

New prompt (for ex. Please select all the 45th President of the US)

### Sitekey

{sitekey}

### Sitelink

{sitelink}

### Screenshot of the challenge

![Screenshot of {prompt}]({screenshot_url})

"""


@dataclass
class GitHubAPI:
    token: str
    owner: str
    repo: str

    client: Client = None

    @classmethod
    def from_env(cls, token: str | None = None, owner: str | None = None, repo: str | None = None):
        token = token or os.getenv("GITHUB_TOKEN")
        owner = owner or os.getenv("GITHUB_OWNER")
        repo = repo or os.getenv("GITHUB_REPO")
        if not token:
            raise ValueError("Miss field GITHUB_TOKEN")
        if not owner:
            raise ValueError("Miss field GITHUB_OWNER")
        if not repo:
            raise ValueError("Miss field GITHUB_REPO")

        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}
        client = Client(headers=headers, follow_redirects=True, base_url="https://api.github.com")
        return cls(token=token, owner=owner, repo=repo, client=client)

    def __del__(self):
        try:
            if self.client:
                self.client.close()
        except AttributeError:
            pass

    def create_issue(
        self,
        title: str,
        *,
        body: str | None = "",
        labels: List[str] | None = None,
        assignees: List[str] | None = None,
    ) -> Dict[str, str]:
        """
        https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#create-an-issue

        :param title:
        :param body:
        :param labels:
        :param assignees:
        :return:
        """
        api = f"/repos/{self.owner}/{self.repo}/issues"
        data = {"title": title, "body": body, "labels": labels or [], "assignees": assignees or []}
        resp = self.client.post(api, json=data)
        return resp.json()

    def get_issue(self, issue_number: int) -> Dict[str, str]:
        """
        https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#get-an-issue

        :param issue_number:
        :return:
        """
        api = f"/repos/{self.owner}/{self.repo}/issues/{issue_number}"
        resp = self.client.post(api)
        return resp.json()

    def list_repo_issues(
        self,
        *,
        state: Literal["open", "closed", "all"] = "all",
        labels: str = "",
        per_page: int = 10,
    ) -> Dict[str, str]:
        """
        https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#list-repository-issues

        :param per_page: The number of results per page (max 100). Default: 30
        :param state:
        :param labels: A list of comma separated label names. Example: `bug,ui,@high`
        :return:
        """
        api = f"/repos/{self.owner}/{self.repo}/issues"
        query = {"state": state, "labels": labels, "per_page": per_page}
        resp = self.client.get(api, params=query)
        return resp.json()

    def update_issue(
        self,
        url: str,
        *,
        state: Literal["open", "closed"] = "open",
        state_reason: Literal["completed", "not_planned", "reopened"] = None,
    ) -> Dict[str, str]:
        """
        /repos/{self.owner}/{self.repo}/issues/{issue_number}

        https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#update-an-issue
        :param url: 通过 list 方法获取筛选后的 issue 对象，返回值中的 url 参数可以继续使用
        :param state:
        :param state_reason:
        :return:
        """
        body = {"state": state, "state_reason": state_reason}
        res = self.client.post(url, json=body)
        return res.json()
