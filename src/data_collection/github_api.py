"""
Fetch repository metadata and open issues from the GitHub REST API.

Usage:
    python src/data_collection/github_api.py --repos data/processed/repo_list.txt

Requires GITHUB_TOKEN in .env or environment.
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

PROCESSED_DIR = Path("data/processed")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
BASE_URL = "https://api.github.com"


def get_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _get(url: str, params: dict = None) -> dict | list | None:
    headers = get_headers()
    for attempt in range(3):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 1)
            print(f"  Rate limited — sleeping {wait:.0f}s")
            time.sleep(wait)
        elif resp.status_code == 404:
            return None
        else:
            time.sleep(2 ** attempt)
    return None


def fetch_repo_metadata(full_name: str) -> dict | None:
    data = _get(f"{BASE_URL}/repos/{full_name}")
    if not data:
        return None
    topics_data = _get(f"{BASE_URL}/repos/{full_name}/topics") or {}
    readme_data = _get(f"{BASE_URL}/repos/{full_name}/readme") or {}

    readme_text = ""
    if readme_data.get("content"):
        import base64
        try:
            readme_text = base64.b64decode(readme_data["content"]).decode("utf-8", errors="replace")
        except Exception:
            pass

    return {
        "repo": full_name,
        "language": data.get("language", ""),
        "topics": topics_data.get("names", []),
        "description": data.get("description", "") or "",
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "open_issues_count": data.get("open_issues_count", 0),
        "readme": readme_text[:4000],
    }


def fetch_open_issues(full_name: str, label: str = "good first issue", per_page: int = 50) -> list[dict]:
    params = {"state": "open", "per_page": per_page}
    if label:
        params["labels"] = label
    data = _get(f"{BASE_URL}/repos/{full_name}/issues", params=params)
    if not data:
        return []
    return [
        {
            "repo": full_name,
            "number": issue["number"],
            "title": issue.get("title", ""),
            "body": (issue.get("body") or "")[:1000],
            "labels": [l["name"] for l in issue.get("labels", [])],
            "url": issue.get("html_url", ""),
        }
        for issue in data
        if "pull_request" not in issue
    ]


def fetch_user_pr_history(username: str, max_prs: int = 30) -> tuple[list[str], list[str]]:
    """Return (pr_titles, pr_bodies) for a user's recently merged PRs via search API."""
    params = {
        "q": f"type:pr author:{username} is:merged",
        "per_page": max_prs,
        "sort": "updated",
    }
    data = _get(f"{BASE_URL}/search/issues", params=params)
    if not data or "items" not in data:
        return [], []
    titles = [item.get("title", "") for item in data["items"]]
    bodies = [(item.get("body") or "")[:300] for item in data["items"]]
    return titles, bodies


def collect_repo_metadata(repo_list_path: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    repos = Path(repo_list_path).read_text().strip().splitlines()

    out_path = PROCESSED_DIR / "repo_metadata.jsonl"
    with open(out_path, "w") as f:
        for repo in tqdm(repos, desc="Fetching repo metadata"):
            meta = fetch_repo_metadata(repo.strip())
            if meta:
                f.write(json.dumps(meta) + "\n")

    print(f"Wrote metadata for repos to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", required=True, help="Path to file with one repo per line (owner/name)")
    args = parser.parse_args()
    collect_repo_metadata(args.repos)
