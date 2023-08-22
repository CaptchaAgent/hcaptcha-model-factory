# -*- coding: utf-8 -*-
# Time       : 2023/3/21 17:50
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description: https://docs.github.com/en/rest/issues/issues
import os
import typing

import httpx

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


class GitHubAPI:
    def __init__(
        self, token: typing.Optional[str] = "", root: typing.Optional[str] = "api.github.com"
    ):
        self.token = os.getenv("GITHUB_TOKEN") or token
        if not self.token:
            raise ValueError("GitHub_TOKEN missing")
        self.root = root


class GitHubIssueAPI(GitHubAPI):
    def __init__(
        self,
        owner: typing.Optional[str] = "",
        repo: typing.Optional[str] = "",
        token: typing.Optional[str] = "",
        *,
        root: typing.Optional[str] = "api.github.com",
    ):
        super().__init__(token=token, root=root)
        self._owner = os.getenv("GITHUB_OWNER") or owner
        self._repo = os.getenv("GITHUB_REPO") or repo
        if not self._owner:
            raise ValueError("GITHUB_OWNER missing")
        if not self._repo:
            raise ValueError("GITHUB_REPO missing")

        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
        }

    def validate(self) -> bool:
        return all([self.token, self._owner, self._repo])

    def create_issue(
        self,
        title: str,
        *,
        body: typing.Optional[str] = "",
        labels: typing.Optional[typing.List[str]] = None,
        assignees: typing.Union[typing.List[str]] = None,
    ) -> typing.Optional[typing.Dict[str, str]]:
        """
        HTTP response status codes for "Create an issue"

        Status code	Description
        ------------------------
        201	Created <<success>>
        ------------------------
        403	Forbidden
        404	Resource not found
        410	Gone
        422	Validation failed, or the endpoint has been spammed.
        503	Service unavailable

        :param title:
        :param body:
        :param labels:
        :param assignees:
        :return:
        """
        url = f"https://{self.root}/repos/{self._owner}/{self._repo}/issues"
        data = {"title": title, "body": body, "labels": labels or [], "assignees": assignees or []}
        resp = httpx.post(url, headers=self._headers, json=data)
        return resp.json()

    def get_issue(self, issue_number: int) -> typing.Optional[typing.Dict[str, str]]:
        """
        HTTP response status codes for "Get an issue"

        Status code	Description
        ------------------------
        200	OK
        ------------------------
        301	Moved permanently
        304	Not modified
        404	Resource not found
        410	Gone
        :param issue_number:
        :return:
        """
        url = f"https://{self.root}/repos/{self._owner}/{self._repo}/issues/{issue_number}"
        resp = httpx.post(url, headers=self._headers)
        return resp.json()

    def list_repo_issues(
        self,
        *,
        state: typing.Literal["open", "closed", "all"] = "all",
        labels: typing.Optional[str] = "",
        per_page: typing.Optional[int] = 10,
    ) -> typing.Optional[typing.Dict[str, str]]:
        """
        HTTP response status codes for "List repository issues"
        ---
        Status code	Description
        ---
        200	OK
        301	Moved permanently
        404	Resource not found
        422	Validation failed, or the endpoint has been spammed.
        ---
        https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#list-repository-issues

        :param per_page: The number of results per page (max 100). Default: 30
        :param state:
        :param labels: A list of comma separated label names. Example: `bug,ui,@high`
        :return:
        """
        api = f"https://{self.root}/repos/{self._owner}/{self._repo}/issues"
        query = {"state": state, "labels": labels, "per_page": per_page}
        resp = httpx.get(api, headers=self._headers, params=query)
        return resp.json()


def create_issue_body_about_general_challenge(
    *, prompt: str, screenshot_url: str, sitekey: str, sitelink: typing.Optional[str] = ""
) -> str:
    """

    :param prompt:
    :param screenshot_url:
    :param sitelink: "https://accounts.hcaptcha.com/demo?sitekey={sitekey}"
    :param sitekey:
    :return:
    """
    sitelink = sitelink or f"https://accounts.hcaptcha.com/demo?sitekey={sitekey}"
    return ISSUE_TEMPLATE_CHALLENGE_GENERAL.format(
        prompt=prompt, sitekey=sitekey, sitelink=sitelink, screenshot_url=screenshot_url
    )


def test_create_issue():
    body = create_issue_body_about_general_challenge(
        prompt="superman",
        screenshot_url="https://i0.hdslb.com/bfs/banner/729322738c403cd67bfbf6a9f3242a759f841815.jpg@1200w_300h_1c.webp",
        sitekey="adafb813-8b5c-473f-9de3-485b4ad5aa09",
    )
    gh = GitHubIssueAPI()
    resp = gh.create_issue(
        "[Challenge] superman", body=body, labels=["🔥 challenge"], assignees=["QIN2DIM"]
    )
    print(resp)


if __name__ == "__main__":
    test_create_issue()
