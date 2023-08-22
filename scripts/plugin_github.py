# -*- coding: utf-8 -*-
# Time       : 2023/8/23 1:29
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from hcaptcha_whistleblower.plugins.github_issues import GitHubAPI, ISSUE_TEMPLATE_CHALLENGE_GENERAL

sitekey = "adafb813-8b5c-473f-9de3-485b4ad5aa09"
focus_cleaning_prompt = "孤注一擲"
screenshot_url = "https://p6.itc.cn/q_70/images03/20230629/a4b2d7cd99e3490a9cc2106cc232100d.jpeg"
sitelink = f"https://accounts.hcaptcha.com/demo?sitekey={sitekey}"
sign_label = "🔥 challenge"
gh = GitHubAPI.from_env()


def create_issue():
    body = ISSUE_TEMPLATE_CHALLENGE_GENERAL.format(
        prompt=focus_cleaning_prompt,
        screenshot_url=screenshot_url,
        sitekey=sitekey,
        sitelink=sitelink,
    )
    resp = gh.create_issue(
        f"[Challenge] {focus_cleaning_prompt}",
        body=body,
        labels=[sign_label],
        assignees=["QIN2DIM"],
    )
    print(resp)

    print(f"View at {resp['html_url']}")


def close_challenge_issue():
    resp = gh.list_repo_issues(labels=sign_label)
    for i in resp:
        url = i["url"]
        gh.update_issue(url=url, state="closed", state_reason="completed")


if __name__ == "__main__":
    close_challenge_issue()
