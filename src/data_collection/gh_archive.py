"""
Download and parse GH Archive hourly event dumps.

Usage:
    python src/data_collection/gh_archive.py --start 2024-01-01 --end 2024-03-31
"""

import argparse
import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path("data/raw/gh_archive")
PROCESSED_DIR = Path("data/processed")

RELEVANT_EVENT_TYPES = {
    "PullRequestEvent",
    "IssuesEvent",
    "IssueCommentEvent",
    "PullRequestReviewEvent",
}

EVENT_WEIGHTS = {
    "PullRequestEvent_closed_merged": 5,
    "PullRequestEvent_opened": 2,
    "IssuesEvent_opened": 1,
    "IssueCommentEvent_created": 0.5,
}


def date_range(start: str, end: str):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    current = start_dt
    while current <= end_dt:
        for hour in range(24):
            yield current.strftime("%Y-%m-%d"), hour
        current += timedelta(days=1)


def download_hour(date: str, hour: int, out_dir: Path) -> Path:
    url = f"https://data.gharchive.org/{date}-{hour}.json.gz"
    out_path = out_dir / f"{date}-{hour}.json.gz"
    if out_path.exists():
        return out_path
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return out_path


def parse_event(event: dict) -> dict | None:
    event_type = event.get("type")
    if event_type not in RELEVANT_EVENT_TYPES:
        return None

    actor = event.get("actor", {}).get("login", "")
    repo = event.get("repo", {}).get("name", "")

    if not actor or not repo or actor.endswith("[bot]"):
        return None

    payload = event.get("payload", {})
    weight_key = None

    if event_type == "PullRequestEvent":
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        if action == "closed" and pr.get("merged"):
            weight_key = "PullRequestEvent_closed_merged"
        elif action == "opened":
            weight_key = "PullRequestEvent_opened"
    elif event_type == "IssuesEvent" and payload.get("action") == "opened":
        weight_key = "IssuesEvent_opened"
    elif event_type == "IssueCommentEvent" and payload.get("action") == "created":
        weight_key = "IssueCommentEvent_created"

    if weight_key is None:
        return None

    return {
        "user": actor,
        "repo": repo,
        "event": weight_key,
        "weight": EVENT_WEIGHTS[weight_key],
        "created_at": event.get("created_at", ""),
    }


def process_archive(start: str, end: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    interactions: dict[tuple, float] = {}

    for date, hour in tqdm(list(date_range(start, end)), desc="Processing GH Archive"):
        try:
            gz_path = download_hour(date, hour, RAW_DIR)
        except Exception as e:
            print(f"  Skip {date}-{hour}: {e}")
            continue

        try:
            with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parsed = parse_event(event)
                    if parsed:
                        key = (parsed["user"], parsed["repo"])
                        interactions[key] = interactions.get(key, 0.0) + parsed["weight"]
        except EOFError:
            # File was truncated (download interrupted) — delete and retry once
            print(f"  Truncated file {gz_path.name}, re-downloading ...")
            gz_path.unlink()
            try:
                gz_path = download_hour(date, hour, RAW_DIR)
                with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        parsed = parse_event(event)
                        if parsed:
                            key = (parsed["user"], parsed["repo"])
                            interactions[key] = interactions.get(key, 0.0) + parsed["weight"]
            except Exception as e:
                print(f"  Skip {date}-{hour} after retry: {e}")

    # Write out as TSV
    out_path = PROCESSED_DIR / "interactions_raw.tsv"
    with open(out_path, "w") as f:
        f.write("user\trepo\tscore\n")
        for (user, repo), score in interactions.items():
            f.write(f"{user}\t{repo}\t{score:.4f}\n")

    print(f"Wrote {len(interactions):,} user-repo pairs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()
    process_archive(args.start, args.end)
